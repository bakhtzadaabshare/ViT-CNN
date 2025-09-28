import torch
#from dataLoader import BrainTumorDataset, train_transform, test_transform
#from cnnModel import CNNModel
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
#from otherModels import vgg_model, resNet_model, mobileNet_model
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import seed_everything

from models import CNNModel, CNNWithViTCross, vit_model
from visualization import test_model, dataVisualization
from configs import config
from dataLoader import train_loader, test_loader, BR35HDataset, transform, test_transform
from torch.utils.data import random_split

class MyModel(pl.LightningModule):
    def __init__(self, model, criterion, optimizer_class, learning_rate=1e-3):
        super(MyModel, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate

        # Attributes to hold metrics
        #self.train_acc_history = []
        #self.valid_acc_history = []
        #self.y_score_history = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        if config.binary_dataset:
            labels = labels.to(config.device).float().unsqueeze(1)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            preds = torch.sigmoid(outputs) > 0.5
            acc = (preds == labels).float().mean()
        else:
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            acc = (predicted == labels).float().mean()
        
        #self.train_acc_history.append(acc.item())  # Store training accuracy
        #self.y_score_history.append(outputs.cpu().detach().numpy())  # Store logits
        
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch 
        if config.binary_dataset:
            labels = labels.to(config.device).float().unsqueeze(1)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            preds = torch.sigmoid(outputs) > 0.5
            acc = (preds == labels).float().mean()
        else:
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            acc = (predicted == labels).float().mean()

        #self.valid_acc_history.append(acc.item())  # Store validation accuracy
        #self.y_score_history.append(outputs.cpu().detach().numpy())  # Store logits
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.model.parameters(), lr=self.learning_rate)
                # Define scheduler
        scheduler = {
            'scheduler':  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50),
            'interval': 'epoch',  # Metric to monitor for reducing LR
            #'scheduler': ReduceLROnPlateau(optimizer, mode='min', patience=6, verbose=True, factor=0.1),
            #'monitor': 'val_loss',  1# Metric to monitor for reducing LR
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
class BestMetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

def on_validation_epoch_end(self, trainer, pl_module):
    val_loss = trainer.callback_metrics['val_loss']
    val_acc = trainer.callback_metrics['val_acc']

    if val_loss < self.best_val_loss:
        self.best_val_loss = val_loss
        trainer.save_checkpoint('best_val_loss_checkpoint.ckpt')

    if val_acc > self.best_val_acc:
        self.best_val_acc = val_acc
        trainer.save_checkpoint('best_val_acc_checkpoint.ckpt')

    print(f"\nEpoch {trainer.current_epoch}: Best Val Loss: {self.best_val_loss:.4f}, Best Val Acc: {self.best_val_acc:.4f}")

#customizing the callback class to update user after each epoch.
class LearningRateLogger(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
    # Log the learning rate for each optimizer
        for i, optimizer in enumerate(trainer.optimizers):
            current_lr = optimizer.param_groups[0]['lr']
            pl_module.log(f'learning_rate_optimizer_{i}', current_lr, prog_bar=True, logger=True)
            print(f"Learning Rate for Optimizer {i}: {current_lr}")

    
def main():
    #customizing the callback class to update user after each epoch.
    print("Loading datasets...")
    print("Which dataset to use?")
    print("1. Multi-class dataset")
    print("2. Binary-class dataset (BR35H)")
    choice = input("Enter your choice (1 or 2): ")
    if choice == '1':
        config.binary_dataset = False
        num_classes = 4
    elif choice == '2':
        # Load full dataset (without transformation yet)
        full_dataset = BR35HDataset(root_dir=config.binary_dataset_path, mode='train')

        # Define the split ratio
        train_size = int(0.8 * len(full_dataset))  # 80% training
        val_size = len(full_dataset) - train_size  # 20% validation

        # Perform the split
        train_indices, val_indices = random_split(full_dataset, [train_size, val_size])

        # Apply different transforms after splitting
        train_br35h = BR35HDataset(root_dir=config.br35h_train, transform=transform, mode='train')
        test_br35h = BR35HDataset(root_dir=config.br35h_test, transform=test_transform, mode='train')

        # Overwrite dataset with split indices
        train_br35h.image_paths = [full_dataset.image_paths[i] for i in train_indices.indices]
        train_br35h.labels = [full_dataset.labels[i] for i in train_indices.indices]

        test_br35h.image_paths = [full_dataset.image_paths[i] for i in val_indices.indices]
        test_br35h.labels = [full_dataset.labels[i] for i in val_indices.indices]
        config.binary_dataset = True
        #intializing the dataLoader for train and test dataset
        train_dataset = DataLoader(train_br35h, batch_size=config.batch_size, shuffle=True, num_workers=3)
        test_dataset = DataLoader(test_br35h, batch_size=config.batch_size, shuffle=False, num_workers=3)
        num_classes = 1 # Binary classification with BCEWithLogitsLoss
        print(f"Training samples: {len(train_br35h)}, Validation samples: {len(test_br35h)}")
    else:
        print("Invalid choice. Exiting.")
        return
    
    seed_everything(42, workers=True)
    # Callbacks
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    best_metric_callback = BestMetricsCallback()
    save_path = config.checkpoint_path #the path where the trained will save

    #model training

    model = CNNWithViTCross(num_classes=num_classes)
    #model = CNNModel()
    #model = vit_model()

    #model = Res_Model(input_dim = (3, 256, 256), output_dim = 4)
    #vit.to(device)
    if config.binary_dataset:
        criterion = nn.BCEWithLogitsLoss()  #loss function for the binary classification
    else:
        criterion = nn.CrossEntropyLoss()  #loss function for the categorical classification
    #criterion = nn.BCEWithLogitsLoss()
    #optimizer = optim.Adam(vit_model.head.parameters(), lr=0.001)   #definging the adam optimizer by setting some basic parameters
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer = optim.Adam
    #initializing torch model
    lightning_model = MyModel(model, criterion, optimizer, config.learning_rate)
    #ensure that the directory is already present
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #Defining checkpoint callback
    checkpoint_callback = ModelCheckpoint(dirpath=save_path, save_top_k=1, monitor='val_loss', mode='min')
    # Trainer
    trainer = pl.Trainer(max_epochs=config.num_epochs, 
                            callbacks=[early_stopping_callback, checkpoint_callback, best_metric_callback, LearningRateLogger()],
                        accelerator=config.device.type,  # Use 'gpu' or 'cpu' based on your setup
                        devices='auto',
                            #strategy='xla',
                        #strategy = 'ddp',
                        #precision = 16)
                            deterministic=True,
                        )
    # Train the model
    trainer.fit(lightning_model, train_loader, test_loader)

    print(f'Device detected for training: {config.device}')

    #model prediction
    true_labels, predictions, y_score = test_model(model, test_loader)
    #evaluate_and_plot_confusion_matrix(lightning_model, test_loader)
    # Data visualization using confusion materix
    dataVisualization(true_labels=true_labels, predictions=predictions, y_score=y_score)

if __name__ == "__main__":
    main()