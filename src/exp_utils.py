import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from torch.utils.data import DataLoader
import torch

from src.model import DiagnosticModel

def save_bad_examples(model: DiagnosticModel, data_loader: DataLoader, output_dir: Path, ckpt_path: Path = None):
    """
    Takes in a model and a data_loader, and it saves a file that shows mis-classifications
    """
    device = next(model.parameters()).device
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    all_fp = []
    all_fn = []

    for data, labels in data_loader:
        data, labels = data.to(device), labels.to(device)

        outputs = model(data) 
        _, preds = torch.max(outputs, dim = 1)

        false_pos = (labels == 0) & (preds == 1)
        false_negs = (labels == 1) & (preds == 0)

        # move channels to the end; extend the list by batch
        all_fp.extend(data[false_pos].cpu().permute(0, 2, 3, 1))  # each img is (W, H, 3)
        all_fn.extend(data[false_negs].cpu().permute(0, 2, 3, 1))

    model.train()
    np.random.shuffle(all_fp)
    np.random.shuffle(all_fn)

    print(f"There are {len(all_fp)} false positives")
    fig, axes = plt.subplots(nrows=3, ncols = 5, figsize = (15, 9))
    for i, ax in enumerate(axes.flat):
        if i >= len(all_fp):
            break
        ax.imshow(all_fp[i][:, :, 0], cmap = 'grey')
        ax.set_title(f"Image {i+1}")
        ax.axis('off')
    fig.suptitle("False Positive Examples", fontsize=20)
    plt.tight_layout()
    fig.savefig(output_dir / "false_positives.png")
    plt.close(fig)

    print(f"There are {len(all_fn)} false negatives")
    fig, axes = plt.subplots(nrows=3, ncols = 5, figsize = (15, 9))
    for i, ax in enumerate(axes.flat):
        if i >= len(all_fn):
            break
        ax.imshow(all_fn[i][:, :, 0], cmap = 'grey')
        ax.set_title(f"Image {i+1}")
        ax.axis('off')
    fig.suptitle("False Negative Examples", fontsize=20)
    plt.tight_layout()
    fig.savefig(output_dir / "false_negatives.png")
    plt.close(fig)




