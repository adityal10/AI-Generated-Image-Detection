import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import torch

def compute_accuracy(outputs, labels):
    preds = outputs.argmax(dim=1)  # predicted class
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)

def compute_ap(outputs, labels):
    # Convert labels to CPU numpy
    labels_np = labels.cpu().numpy()
    # Get predicted probability for class 1 (real)
    probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
    return average_precision_score(labels_np, probs)