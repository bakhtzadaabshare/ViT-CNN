import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataLoader import BrainTumorDataset, test_transform
from models import CNNWithViTCross
from visualization import test_model, dataVisualization
from configs import config
from dataLoader import test_loader

def loadPretrainedModel(model, checkpoint_path, device='cpu'):
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Print checkpoint keys
    #print("Checkpoint keys:", checkpoint.keys())

    # Function to remove prefix from state dict keys
    def remove_prefix_from_keys(state_dict, prefix):
        return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}

    # Load model state dict
    model_state_dict = checkpoint['state_dict']
    model_state_dict = remove_prefix_from_keys(model_state_dict, "model.")  # Fix key names
    model.load_state_dict(model_state_dict, strict=False)

    # Load optimizer state if available
    if 'optimizer_states' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_states'][0])

    # Move optimizer state tensors to correct device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # Set model to evaluation mode
    model.eval()
    
    return model

def main():
    # Instantiate the model
    model = CNNWithViTCross()

    # Specify device (CPU or GPU)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    model.to(device)  # Move model to the correct device
    # checkpoint_path
    checkpoint_path = r'D:\epoch=32-step=5907.ckpt'
    loaded_model = loadPretrainedModel(model, checkpoint_path, device)

    # Perform model prediction
    true_labels, predictions, y_score = test_model(loaded_model, test_loader)

    # Data visualization
    dataVisualization(true_labels=true_labels, predictions=predictions, y_score=y_score)

if __name__ == "__main__":
    main()