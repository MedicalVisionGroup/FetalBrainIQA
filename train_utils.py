import numpy as np
from torch.utils.data import Subset
import torch

def conf_matrix(y_true, y_pred, threshold=0.5) -> np.array:
    """
    Compute confusion matrix for binary classification.
    
    Args:
        y_true (Tensor): Ground truth labels (0 or 1), shape (N,)
        y_pred (Tensor): Predicted probabilities or logits, shape (N,)
        threshold (float): Threshold to convert probabilities into class labels. Irrelevant
                            here since all values either 0 or 1 
    
    Returns:
        [tn, fp, fn, tp]: Confusion matrix counts.
    """
    # If predictions are logits/probabilities, threshold them
    y_pred_label = (y_pred >= threshold).int()
    y_true = y_true.int()

    # Compute values
    tp = ((y_true == 1) & (y_pred_label == 1)).sum().item()
    tn = ((y_true == 0) & (y_pred_label == 0)).sum().item()
    fp = ((y_true == 0) & (y_pred_label == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred_label == 0)).sum().item()
    
    return np.array([tn, fp, fn, tp])

def evaluate(model, loader, device):
    """Evaluate accuracy on a given dataset loader. 
    Return (pos_accuracy, neg_accuracy)"""
    model.eval()

    accuracies = np.zeros(4)
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)
            accuracies += conf_matrix(labels, preds)

    model.train()

    return accuracies

def summarize_dataset(subset: Subset, name = "subset"):
    result = f"{name}:\n"
    result += f"Image Size: {subset[0][0].shape}\n"
    result += f"Size: {len(subset)}\n"
    pos_samples = len([label for fpath, label in subset if label == 1])
    result += f"Number of Pos Samples: {pos_samples}\n"
    result += f"Number of Neg Samples: {len(subset) - pos_samples}\n"

    print(result)

def display_accuracies(array: np.array) -> str:
    # input (tn, fp, fn, tp)
    perc = array / sum(array) * 100
    return f"tn={perc[0]:.2f}%,fp={perc[1]:.2f}%,fn={perc[2]:.2f}%,tp={perc[3]:.2f}%"

