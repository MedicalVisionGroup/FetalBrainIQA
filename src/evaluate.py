from pathlib import Path
import numpy as np
import json

from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from model import DiagnosticModel
from train_utils import conf_matrix, generate_roc


def evaluate(model: DiagnosticModel, loader: DataLoader, 
             device, 
             criterion: nn.Module = None,
             roc_path: Path = None, 
             ckpt_path: Path | None = None, 
             save_path: Path | None = None):
    """
    Runs the model on the validation set and returns 
    the cfvalues for the run
    """    

    print("Evaluating")
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    val_cfvalues = np.zeros(4)
    all_probs = []
    all_labels = []
    loss = 0.0
    total_items = 0.0

    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)

            _, preds = torch.max(outputs, dim=1)

            val_cfvalues += conf_matrix(preds, labels)

            loss += (criterion(outputs, labels).item() * data.size(0))
            total_items += data.size(0)

            # Get probabilities for positive class (class 1)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())     

    loss /= total_items

    # Calculate ROC Graph
    all_probs_tensor = torch.cat(all_probs)
    all_labels_tensor = torch.cat(all_labels)
    auc = generate_roc(all_probs_tensor, all_labels_tensor, fpath = roc_path, title="Full Dataset")

    # Dump Values
    if save_path is not None:
        with open(save_path, 'w') as f:
            json.dump({'auc': auc, 'val_cfvalues': list(val_cfvalues)}, f)

    model.train()

    return val_cfvalues, loss, auc
