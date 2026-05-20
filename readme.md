# Brain Tumor Classification Using MRI data

This project is designed to detect brain tumors using a novel technique titled:

**"ViT-CNN: Explainable Dual-Stream Cross-Attention Framework for Brain Tumor Screening on Consumer Devices"**

This framework introduces an innovative approach to brain tumor detection by leveraging a dual-stream architecture that combines Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs). The model incorporates cross-attention mechanisms to enhance feature fusion and improve classification accuracy. Additionally, it emphasizes explainability by utilizing techniques such as Saliency Maps and Local Interpretable Model-agnostic Explanations (LIME) to provide insights into the decision-making process of the model.

The proposed framework aims to advance the field of medical imaging by providing a robust and interpretable solution for brain tumor classification, making it a valuable tool for healthcare professionals and researchers.

Follow the instructions below to set up and use the project.

## Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- Required Python libraries: 
 1. pytorch version 2.3.0 or above: for the core model development
 2. pytorch-lightning: for the model training
 3. timm: for the ViT-tiny 
 4. lime: for the model explainability

These libraries can be installed easily by running:

```
pip install -r requirements.txt
```

This will install all the necessary dependencies listed in the `requirements.txt` file.

## Usage

1. Prepare your dataset:
    - The current model is designed for the [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and [BR35H dataset](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection).
    - If you want to train it on your own dataset that make changes in the dataLoader.py file accordingly.
    - Ensure the images are properly labeled according to the dataset structure.

2. Run the training script:
    python train.py: It will automatically store the checkpoint with best acccuracy

3. To test the model on test portion of the dataset:
    python inferences.py: This will load the already store model and validate it on test set of the dataset and visualize the result.

## File Structure

- `config.py`: This script contain the basic configuration of the model such is batch_size, dataset_path, model_saving_path, num_epoch, etc.
- `train.py`: Script to train the model.
- `inferences.py`: Script that load the stored model and make predictions.
- `visualization.py`: Script to visualize the result of the model.
- `models.py`: Script that define the overall architecture of the model including, CNN, ViT, ViT-CNN with different feature fusion techniques.
- `explainable.py`: Script that explain the model result by using Salinecy maps and LIME explaination.



