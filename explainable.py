import torch
import numpy as np
from configs import config
import matplotlib.pyplot as plt 
import seaborn as sns
#from dataLoader import test_loader
from inferences import loadPretrainedModel
from models import CNNWithViTCross
from torch.utils.data import DataLoader
from dataLoader import test_dataset
from lime import lime_image
from skimage.segmentation import mark_boundaries
#from skimage.segmentation import felzenswalb
import numpy as np
from skimage.segmentation import felzenszwalb


def denormalization(image):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    denormalized_image = image * std[:, None, None] + mean[:, None, None]
    # Convert the denormalized image to numpy and clip the values to [0, 1]
    denormalized_image = denormalized_image.cpu().numpy().transpose(1, 2, 0)  # Convert to (H, W, C)
    denormalized_image = np.clip(denormalized_image, 0, 1)  # Ensure values are within [0, 1]
    return denormalized_image

def visualize_saliency(model, images, labels, title):
    
    # Assuming 'model' is your trained CNN+ViT-Tiny model and 'test_loader' is your DataLoader
    #device = 'cpu'
    model = model.to(config.device)
    model.eval()
    
    # Select the first 6 images
    num_images = 4
    #images, labels = images[:num_images], labels[:num_images]
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    # Create a figure for displaying the saliency maps
    fig, axes = plt.subplots(2, num_images, figsize=(10, 6))
    
    num_index = []
    j = 0
        
    
    for i in range(len(images)):
        image = images[i].unsqueeze(0).to(config.device)  # Add batch dimension
        image.requires_grad = True
    
        # Forward pass
        output = model(image)
        probs = torch.nn.functional.softmax(output, dim=1)
        predicted_class_index = torch.argmax(probs, dim=1).item()
        if predicted_class_index not in num_index:
            num_index.append(predicted_class_index)
            predicted_class_name = class_names[predicted_class_index]  # Get class name
            true_class_name = class_names[labels[i].item()]  # Get true class name
            #print(f"Image {i+1} Predicted class index:", predicted_class_index)
        
            class_output = output[0, predicted_class_index]
        
            # Backpropagation to get gradients
            model.zero_grad()
            class_output.backward()
        
            # Compute saliency map
            saliency = image.grad.abs().squeeze(0)  # Remove batch dimension
            saliency = saliency.max(dim=0)[0].cpu().detach().numpy()  # Take max across channels
        
            # Normalize saliency values
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    
            d_image = denormalization(images[i])
        
            # Display the original image
            axes[0, j].imshow(d_image)  # Convert to (H, W, C) format
            axes[0, j].axis("off")
            axes[0, j].set_title(f"True: {true_class_name}\nPred: {predicted_class_name}")
        
            # Display the saliency map
            sns.heatmap(saliency, cmap='hot', cbar=False, square=True, xticklabels=False, yticklabels=False, ax=axes[1, j])
            axes[1, j].set_title(f"Saliency ({title})")
            j = j+1
    plt.tight_layout()
    plt.savefig(f"Saliency_{title}.png", dpi=1000, bbox_inches='tight' )
    plt.show()

#LIME explaination

def limeExplaination(model, images, labels):
    model.to(config.device)
    # Convert PyTorch model to a function compatible with LIME
    def pytorch_model_predict(image):
        # Convert images to torch tensor
        image = torch.tensor(image).permute(0, 3, 1, 2).float()
        image = image.contiguous()
        image = image.to(config.device)  # Send to device (CPU/GPU)
        outputs = model(image)  # Forward pass
        probs = torch.nn.functional.softmax(outputs, dim=1)
        return probs.cpu().detach().numpy()

    # Instantiate the explainer
    explainer = lime_image.LimeImageExplainer()

    # Generate explanation for a single image
    #image_np = images[6].squeeze(0).permute(1, 2, 0).detach().numpy()  # Convert to NumPy format
    d_image = denormalization(images[17])
    #image_np2 = d_image.squeeze(0).permute(1, 2, 0).detach().numpy()

    explanation = explainer.explain_instance(
        d_image, 
        pytorch_model_predict, 
        top_labels=4, 
        hide_color=0, 
        num_samples=100,  # Number of perturbed samples
    )

    # After getting the output from the model
    image = images[17].unsqueeze(0).to(config.device)
    model.eval()
    output = model(image)
    probs = torch.nn.functional.softmax(output, dim=1)
    predicted_class_index = torch.argmax(probs, dim=1)

    #segments = slic(d_image, n_segments=3, compactness=3)
    #plt.imshow(mark_boundaries(d_image, segments))
    #plt.title("Superpixels")
    #plt.axis("off")
    #plt.show()

    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    true_label = class_names[labels[17].item()]
    print(f"True Label: {true_label}")
    predicted_class_name = class_names[predicted_class_index.item()]

    temp, mask = explanation.get_image_and_mask(
        label=predicted_class_index.item(),
        positive_only=True,
        num_features=2,  # Number of superpixels to show
        hide_rest=False
    )
    plt.imshow(mark_boundaries(temp, mask))
    plt.title(f"\nClass: {true_label}")
    plt.axis("off")
    plt.savefig(f"LIME_Explaination_{true_label}.png", dpi=1000, bbox_inches='tight' )
    plt.show()

def main():
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    # Load the model
    model = CNNWithViTCross()
    checkpoint_path = r'D:\epoch=32-step=5907.ckpt'
    loaded_model = loadPretrainedModel(model, checkpoint_path)
    # Visualize saliency maps for the first 6 images
    visualize_saliency(loaded_model, images, labels, title='ViT-CNN')
    # Perform LIME explanation
    limeExplaination(loaded_model, images, labels)

if __name__ == "__main__":
    main()