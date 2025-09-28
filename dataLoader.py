# loading data from dataset
import os
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from configs import config
from torch.utils.data import DataLoader

# Custom skull-stripping function
def skull_strip(image):
    # Convert PIL image to numpy array
    img_array = np.array(image)

    # Convert to grayscale if necessary
    if len(img_array.shape) == 3:  # Check if it's a color image
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array

    # Thresholding to create a binary mask for the brain region
    _, mask = cv2.threshold(img_gray,20, 255, cv2.THRESH_BINARY)

    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Apply the mask to the original image
    img_masked = cv2.bitwise_and(img_array, img_array, mask=mask)

    # Convert back to PIL Image
    return Image.fromarray(img_masked)


# Add the custom skull-stripping function to your transformations
train_transform = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    #transforms.Lambda(skull_strip),  # Add the skull-stripping step
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#resizing the size of the test set
test_transform = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#Defining the Brain Tumor Dataset by inhereting from the pytorch dataset.
class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.image_paths = []
        self.labels = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        #image = cv2.imread(img_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
class BR35HDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        """
        Args:
            root_dir (string): Directory with subdirectories 'yes', 'no', and 'pred'.
            transform (callable, optional): Optional transform to be applied on an image.
            mode (string): 'train' for labeled data, 'pred' for unlabeled prediction images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode  # 'train' (labeled) or 'pred' (unlabeled)
        
        self.image_paths = []
        self.labels = []  # Only used for training

        if mode == 'train':  # Load labeled data from 'yes' and 'no'
            for class_name, label in [('yes', 1), ('no', 0)]:
                class_dir = os.path.join(root_dir, class_name)
                for img_name in os.listdir(class_dir):
                    if img_name.lower().startswith(('y', 'no')):  # Ensuring correct labeling
                        img_path = os.path.join(class_dir, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(label)
        
        elif mode == 'pred':  # Load images for inference (without labels)
            pred_dir = os.path.join(root_dir, 'pred')
            for img_name in os.listdir(pred_dir):
                img_path = os.path.join(pred_dir, img_name)
                self.image_paths.append(img_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.mode == 'train':  # Return image and label
            return image, self.labels[idx]
        else:  # Return only image for prediction
            return image

# Define transformations for BR35H dataset
transform = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.Lambda(skull_strip),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


#calling the BrainTumorDataset function for train and test
train_dataset = BrainTumorDataset(root_dir=config.train_dir, transform=train_transform)
test_dataset = BrainTumorDataset(root_dir=config.test_dir, transform=test_transform)
# Load the multi-class dataset
#intializing the dataLoader for train and test dataset
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=3)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=3)

