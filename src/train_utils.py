from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


def generate_roc(outputs: torch.Tensor, labels: torch.Tensor):
    """
    Outputs are the generated outputs thus far:
    1 = good
    0 = bad

    This generates the ROC curve and relevant statistics
    """
    labels = labels.int()
    
    xs = []
    ys = []

    for t in np.append(np.linspace(0, 1, 100), 0.5):
        preds = (outputs >= t).int()
    
        tn, fp, fn, tp = conf_matrix(preds, labels)
    
        TPR = tp / (tp + fn) if (tp + fn) > 0 else 0 # given positive label, Prob correct?
        FPR = fp / (fp + tn) if (fp + tn) > 0 else 0 # given negative label, Prob wrong?

        xs.append(FPR)
        ys.append(TPR)
    
    return (xs,ys)
    

def conf_matrix(preds, labels) -> np.array:
    """
    Compute confusion matrix for binary classification.
    
    Returns:
    tn, fp, fn, tp: Confusion matrix counts.
    """
    # If predictions are logits/probabilities, threshold them
    labels = labels.int()
    preds = preds.int()

    # Compute values
    tp = ((labels == 1) & (preds == 1)).sum().item()
    tn = ((labels == 0) & (preds == 0)).sum().item()
    fp = ((labels == 0) & (preds == 1)).sum().item()
    fn = ((labels == 1) & (preds == 0)).sum().item()
    
    return np.array([tn, fp, fn, tp])

def evaluate(model, loader, device):
    """
    Runs the model on the validation set and returns a list of 
    (outputs, labels) for all the runs
    
    """
    model.eval()

    val_cfvalues = np.zeros(4)

    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            
            _, preds = torch.max(outputs, dim=1)
            val_cfvalues += conf_matrix(preds, labels)
            
    model.train()

    return val_cfvalues

def print_accuracies(epoch: int, num_epochs: int, loss: float, train_cfvalues, val_cfvalues, fname: str | None = None):
     text = (
         f"Epoch {epoch+1}/{num_epochs}\n" +
         f"  Loss = {loss:.4f}\n" + 
         f"  Train: {get_info_str(train_cfvalues)}\n" + 
         f"  Val:   {get_info_str(val_cfvalues)}\n"
        )
     
     if fname is not None:
        write_type = 'w' if epoch == 0 else 'a' # append or rewrite

        with open(fname, write_type) as f:  
            f.write(text)

def get_info(cf_values: list):
    cfv = cf_values / sum(cf_values)

    info = {}
    info['tn'] = cfv[0]
    info['fp'] = cfv[1]
    info['fn'] = cfv[2]
    info['tp'] = cfv[3]

    info['prec'] = info['tp'] / (info['tp'] + info['fp'])
    info['recall'] = info['tp'] / (info['tp'] + info['fn'])
    info['f1'] = 2 * (info['recall'] * info['prec']) / (info['recall'] + info['prec'])
    info['acc'] = info['tn'] + info['tp']

    return info

def get_info_str(cf_values: list) -> str:
    """
    results: list of (outputs, labels)
    """

    info = get_info(cf_values)
    return f"""
        tn = {info['tn']*100:4.2f}, tp = {info['tp']*100:4.2f}
        fn = {info['fn']*100:4.2f}, fp = {info['fp']*100:4.2f}

        prec = {info['prec']*100:4.2f}, recall = {info['recall']*100:4.2f}
        f1 = {info['f1']*100:4.2f}, acc = {info['acc']*100:4.2f}
    """

import numpy as np
import matplotlib.pyplot as plt

def display_curve(train_full: list[np.ndarray], val_full: list[np.ndarray], loss_full: list[float], fname: Path):
    """
    Plots Accuracy, FPR, FNR over epochs.
    Each entry in train_cfvalues / val_cfvalues is [TN, FP, FN, TP].
    """
    metrics = ['acc', 'prec', 'recall', 'loss']
    colors = ['red', 'green', 'blue', 'yellow', 'orange']

    train_info = [get_info(cfv) for cfv in train_full]
    val_info = [get_info(cfv) for cfv in val_full]

    # Plot
    epochs = range(1, len(train_full) + 1) 
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # ---- Left axis: fractions ----
    for metric, color in zip(metrics, colors):
        if metric == 'loss':
            continue
        ax1.plot(epochs, [dic[metric] for dic in train_info],
                 label=f"Train {metric.capitalize()}",
                 color=color, linestyle="--")
        ax1.plot(epochs, [dic[metric] for dic in val_info],
                 label=f"Val {metric.capitalize()}",
                 color=color, linestyle="-")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Metric")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xticks(epochs)
    ax1.grid(True)

    # ---- Right axis: loss ----
    ax2 = ax1.twinx()
    ax2.plot(epochs, loss_full, label="Loss", color="orange", linewidth=2, linestyle='--')
    ax2.set_ylabel("Loss")

    # ---- Combine legends ----
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax1.set_title(f"Metrics for {fname.name}")

    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)



