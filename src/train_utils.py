import numpy as np
from torch.utils.data import Subset
import torch
import matplotlib.pyplot as plt

def conf_matrix(y_true, y_pred_label) -> np.array:
    """
    Compute confusion matrix for binary classification.
    
    Returns:
        [tn, fp, fn, tp]: Confusion matrix counts.
    """
    # If predictions are logits/probabilities, threshold them
    y_true = y_true.int()
    y_pred_label = y_pred_label.int()

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

def save_accuracies(epoch: int, num_epochs: int, loss: float, train_cfvalues, val_cfvalues, fname: str | None = None):
     text = (
         f"Epoch {epoch+1}/{num_epochs}\n" +
         f"  Loss = {loss:.4f}\n" + 
         f"  Train: {display_accuracies(train_cfvalues)}\n" + 
         f"  Val:   {display_accuracies(val_cfvalues)}\n"
        )
     
     if fname is not None:
        write_type = 'w' if epoch == 0 else 'a' # append or rewrite

        with open(fname, write_type) as f:  
            f.write(text)
        
def display_accuracies(array: np.array) -> str:
    # input (tn, fp, fn, tp)
    perc = array / sum(array) * 100
    basics = f"tn={perc[0]:.2f}%,fp={perc[1]:.2f}%,fn={perc[2]:.2f}%,tp={perc[3]:.2f}%\t"

    fpr = (array[1] / (array[1] + array[0])) * 100
    fnr = (array[2] / (array[2] + array[3])) * 100
    acc = perc[0] + perc[3]

    rates = f"fpr={fpr:.2f}%, fnr={fnr:.2f}, acc={acc:.2f}%"


    return basics + rates

import numpy as np
import matplotlib.pyplot as plt

def display_curve(train_cfvalues: list[np.ndarray], val_cfvalues: list[np.ndarray], fname: str = "output.png"):
    """
    Plots Accuracy, FPR, FNR over epochs.
    Each entry in train_cfvalues / val_cfvalues is [TN, FP, FN, TP].
    """
    # Stack arrays
    ta = np.vstack(train_cfvalues)  # shape (epochs x 4)
    va = np.vstack(val_cfvalues)

    # Normalize to fractions
    ta = ta / ta.sum(axis=1, keepdims=True)
    va = va / va.sum(axis=1, keepdims=True)

    # Compute metrics
    # Accuracy = (TP + TN)
    # FPR = FP / (FP + TN)
    # FNR = FN / (FN + TP)
    train_acc = ta[:, 0] + ta[:, 3]
    train_fpr = ta[:, 1] / (ta[:, 1] + ta[:, 0])
    train_fnr = ta[:, 2] / (ta[:, 2] + ta[:, 3])

    val_acc = va[:, 0] + va[:, 3]
    val_fpr = va[:, 1] / (va[:, 1] + va[:, 0])
    val_fnr = va[:, 2] / (va[:, 2] + va[:, 3])

    colors = ['red', 'green', 'blue']

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot
    epochs = range(1, len(train_cfvalues) + 1) 
    ax.plot(epochs, train_acc, label="Train Acc", color=colors[0], linestyle='-')
    ax.plot(epochs, val_acc, label="Val Acc", color=colors[0], linestyle='--')

    ax.plot(epochs, train_fpr, label="Train FPR", color=colors[1], linestyle='-')
    ax.plot(epochs, val_fpr, label="Val FPR", color=colors[1], linestyle='--')

    ax.plot(epochs, train_fnr, label="Train FNR", color=colors[2], linestyle='-')
    ax.plot(epochs, val_fnr, label="Val FNR", color=colors[2], linestyle='--')

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fraction")
    ax.set_title("Metrics Over Epochs")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)



