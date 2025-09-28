import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_model(model, test_loader):
    print('Model test started...')
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    y_score = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            y_score.extend(outputs.cpu().numpy())

    accuracy = correct / total
    print(f"Accuracy on test set: {accuracy:.4f}")
    return true_labels, predictions, y_score

def dataVisualization(true_labels, predictions, y_score):
    target_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

    # Print classification report
    print("Classification Report:")
    print(classification_report(true_labels, predictions, target_names=target_names, digits=4))

    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Greens')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=target_names,
           yticklabels=target_names,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')

    # Rotate tick labels for better visibility
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=14)

    # Loop over data dimensions and create text annotations with increased font size
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    fontsize=14,  # Increased font size
                    color="white" if cm[i, j] > thresh else "black")

    # Increase font size for labels and title
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_title('ViT-CNN: Confusion Matrix', fontsize=14)

    fig.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=1000, bbox_inches='tight')
    plt.show()

    # Binarize the output
    y_true = label_binarize(true_labels, classes=[0, 1, 2, 3])
    n_classes = y_true.shape[1]

    # Convert y_score to numpy array
    y_score = np.array(y_score)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curve with larger fonts and bolder lines
    plt.figure(figsize=(8, 6))
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'green']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=3,  # Increased line width
                 label='{0} (AUC = {1:0.2f})'.format(target_names[i], roc_auc[i]))

    plt.plot(fpr["micro"], tpr["micro"],
             label='Micro-average (area = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)  # Thicker micro-average line

    plt.plot([0, 1], [0, 1], 'k--', lw=2.5)  # Thicker diagonal reference line

    # Increase font sizes
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ViT-CNN: ROC Curve', fontsize=12)

    # Increase legend font size
    plt.legend(loc="lower right", fontsize=12)

    plt.savefig("RoC.png", dpi=1000, bbox_inches='tight')
    plt.show()
