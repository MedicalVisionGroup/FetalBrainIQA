import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from torch.utils.data import DataLoader
import torch

from model import DiagnosticModel

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

    fp_indices = []
    fn_indices = []

    global_idx = 0   # <-- Track index in dataset

    for data, labels in data_loader:
        # Dealing w/ idxs
        batch_size = labels.size(0)
        batch_indices = np.arange(global_idx, global_idx + batch_size)
        global_idx += batch_size

        # Running the Model
        data, labels = data.to(device), labels.to(device)

        outputs = model(data) 
        _, preds = torch.max(outputs, dim = 1)

        false_pos_mask = (labels == 0) & (preds == 1)
        false_neg_mask = (labels == 1) & (preds == 0)

        # Save images (move channels to the end; extend the list by batch)
        all_fp.extend(data[false_pos_mask].cpu().permute(0, 2, 3, 1))  # each img is (W, H, 3)
        all_fn.extend(data[false_neg_mask].cpu().permute(0, 2, 3, 1))

        # Save idxs
        fp_indices.extend(batch_indices[false_pos_mask.cpu().numpy()])
        fn_indices.extend(batch_indices[false_neg_mask.cpu().numpy()])


    model.train()

    # ---- SHUFFLE PAIRS TOGETHER ----

    fp_perm = np.random.permutation(len(all_fp))
    fn_perm = np.random.permutation(len(all_fn))

    all_fp = [all_fp[i] for i in fp_perm]
    fp_indices = [fp_indices[i] for i in fp_perm]

    all_fn = [all_fn[i] for i in fn_perm]
    fn_indices = [fn_indices[i] for i in fn_perm]

    # ---- DISPLAY FALSE POSITIVES ----

    print(f"There are {len(all_fp)} false positives")
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 9))
    for i, ax in enumerate(axes.flat):
        if i >= len(all_fp):
            break
        ax.imshow(all_fp[i][:, :, 0], cmap='grey', vmin=0, vmax=1)
        ax.set_title(f"idx={fp_indices[i]}")
        ax.axis('off')
    fig.suptitle("False Positives (good, but predicted bad)", fontsize=20)
    plt.tight_layout()
    fig.savefig(output_dir / "false_positives.png")
    plt.close(fig)

    # ---- DISPLAY FALSE NEGATIVES ----
    
    print(f"There are {len(all_fn)} false negatives")
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 9))
    for i, ax in enumerate(axes.flat):
        if i >= len(all_fn):
            break
        ax.imshow(all_fn[i][:, :, 0], cmap='grey', vmin=0, vmax=1)
        ax.set_title(f"idx={fn_indices[i]}")
        ax.axis('off')
    fig.suptitle("False Negatives (bad, but predicted good)", fontsize=20)
    plt.tight_layout()
    fig.savefig(output_dir / "false_negatives.png")
    plt.close(fig)

    return fp_indices, fn_indices



