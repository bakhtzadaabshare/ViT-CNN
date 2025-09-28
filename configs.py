import torch
class config:
    def __init__(self):
        self.image_size = 256
        self.batch_size = 16
        self.num_epochs = 100
        self.learning_rate = 1e-3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dir = 'E:\PhD Research\My previous Research\multiclass_dataset\Training'
        self.test_dir = 'E:\PhD Research\My previous Research\multiclass_dataset\Testing'
        self.br35h_train = r'E:\PhD Research\My previous Research\binary_class_dataset'
        self.br35h_test = r'E:\PhD Research\My previous Research\binary_class_dataset'
        self.binary_dataset_path = r"E:\PhD Research\My previous Research\binary_class_dataset"
        self.checkpoint_path = '/cnn_vit_checkpoints'
        self.binary_dataset = False
  
config = config()
