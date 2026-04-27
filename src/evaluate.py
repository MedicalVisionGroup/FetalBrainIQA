import numpy as np
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import json
import time

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve

from src.model import DiagnosticModel
from src.data import DicomDataset

class ValidationTracker:
    """
    Used to track validation metrics
    """

    def __init__(self, save_dir: Path, model: nn.Module):
        self.save_dir = save_dir
        self.save_dir.mkdir(exist_ok=True)

        self.model = model

        self.metric_opt = {}
        self.metric_best = {}
        self.metric_best_epoch = {}

    def add_tracker(self, metric_name: str, opt: str = 'min'):
        self.metric_opt[metric_name] = opt
        self.metric_best[metric_name] = float('inf') if opt == 'min' else -float('inf')

    def update_best(self, metric_info: dict, epoch: int):
        for metric_name, best in self.metric_best.items():
            new_value = metric_info[metric_name]

            to_save = (self.metric_opt[metric_name] == 'min' and new_value < best)  or (self.metric_opt[metric_name] == 'max' and new_value > best)
            if to_save:
 
                self.metric_best[metric_name] = new_value
                self.metric_best_epoch[metric_name] = epoch

                torch.save(self.model.state_dict(), self.save_dir / f"model_{metric_name}.pth")
                print(f"Saved new best model for {metric_name} {new_value:.4f}")

        with open(self.save_dir / "info.json", 'w') as f:
            json.dump({
                'best_epoch': self.metric_best_epoch,
                'best_value': self.metric_best,
            }, f, indent = 2)


    def yield_best_models(self):
        for metric_name in self.metric_best.keys():
            self.model.load_state_dict(
                torch.load(self.save_dir / f"model_{metric_name}.pth")
            )
            self.model.eval()
            yield f'model_{metric_name}', self.model


def save_metric_info_epoch(save_path: Path, train_metric_info: dict, val_metric_info: dict):
    with open(save_path, "a") as f:
        f.write(json.dumps({'train': train_metric_info, 'val': val_metric_info}) + "\n")

def save_metric_info_test(save_path: Path, test_metric_info: dict):
    with open(save_path / 'metric_info.json', "w") as f:
        json.dump(test_metric_info, f, indent = 2)

def evaluate_metrics(raw_info: dict, loss: float, epoch: int):
    """
    Takes in a dict with keys mapping to lists of same size:
    - preds
    - probs
    - labels
    - idxs

    Returns dictionary mapping metric_name -> value 
    """
    
    assert {'preds', 'probs', 'labels', 'idxs'} == set(raw_info.keys())
    assert len(raw_info['preds']) == len(raw_info['probs']) == len(raw_info['labels']) == len(raw_info['idxs'])

    # List -> Numpy Arrays
    preds = np.array(raw_info['preds'], dtype = int)
    probs = np.array(raw_info['probs'], dtype = np.float32)
    labels = np.array(raw_info['labels'], dtype = int)

    tp = ((labels == 1) & (preds == 1)).sum().item()
    tn = ((labels == 0) & (preds == 0)).sum().item()
    fp = ((labels == 0) & (preds == 1)).sum().item()
    fn = ((labels == 1) & (preds == 0)).sum().item()

    tpr = tp / (tp + fn) if (tp + fn) != 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) != 0 else float("nan")

    prec = tp / (tp + fp) if (tp + fp) != 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) != 0 else float("nan")
    f1 = 2 * (recall * prec) / (recall + prec) if (recall + prec) != 0 else float("nan")
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else float("nan")

    auc = roc_auc_score(y_true = labels, y_score = probs)
    fpr_curve, tpr_curve, _ = roc_curve(y_true = labels, y_score = probs)
    
    def get_tpr_at(fpr: float):
        return np.interp(fpr, fpr_curve, tpr_curve)  
          
    return {
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tpr': round(float(tpr), 3),
        'fpr': round(float(fpr), 3),
        'prec': round(float(prec), 3),
        'recall': round(float(recall), 3),
        'f1': round(float(f1), 3),
        'acc': round(float(acc), 3),
        'auc': round(float(auc), 3),
        'loss': round(float(loss), 3),
        'epoch': int(epoch),
        'tpr_at_10%': round(float(get_tpr_at(.10)), 3),
        'tpr_at_20%': round(float(get_tpr_at(.20)), 3),
        'tpr_at_30%': round(float(get_tpr_at(.30)), 3),
    }

def get_inference_speed(model: DiagnosticModel, dataset: DicomDataset, device: str,
                        num_examples: int = 100, warmup: int = 100):
    """
    Performs a single inference step many times and averages the total time elapsed. 
    Returns the mean and variance of the times. Note that inference speed is only the time
    it takes the model to produce an output.  
    """
    times = []
    idxs = np.random.choice(len(dataset), size=num_examples, replace=False)

    # Warm-up (not timed)
    for i in range(min(warmup, len(idxs))):
        data, _, _, _ = dataset[idxs[i]]
        data = data.unsqueeze(0).to(device)

        with torch.no_grad():
            model(data)

    # Benchmark
    for idx in idxs:
        data, _, _, _ = dataset[idx]
        data = data.unsqueeze(0).to(device)

        if data.is_cuda:
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.no_grad():
            model(data)

        if data.is_cuda:
            torch.cuda.synchronize()

        times.append(time.perf_counter() - start)

    return np.mean(times), np.var(times)

def evaluate(model: DiagnosticModel, loader: DataLoader, device, criterion: nn.Module = None, 
             save_path: Path | None = None, epoch: int = -1, use_tqdm: bool = True):
    """
    Runs the model on the validation set and returns 
    the metric info 
    """    

    print("Evaluating")

    model.eval()

    raw_info = {
        'preds': [],
        'probs': [],
        'labels': [],
        'idxs': [],
    }

    with torch.no_grad():
        total_loss = 0
        total_samples = 0
        for data, _, labels, idxs in tqdm(loader, "Validating", disable = not use_tqdm):
            data, labels = data.to(device), labels.to(device)

            logits = model(data)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim = -1)
            _, preds = torch.max(probs, dim=1)

            raw_info['preds'].extend(preds.detach().cpu().numpy().tolist())
            raw_info['labels'].extend(labels.detach().cpu().numpy().tolist())
            raw_info['probs'].extend(probs[:, 1].detach().cpu().numpy().tolist())
            raw_info['idxs'].extend(idxs.detach().cpu().numpy().tolist())

            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)

        total_loss /= total_samples

    if save_path is not None: 
        used_dataset: DicomDataset = loader.dataset
        extra_info = used_dataset.get_extra_info(raw_info['idxs'], info = ['path', 'mask_path', 'slice_num'])
        save_info = raw_info | extra_info

        pd.DataFrame(save_info).to_csv(save_path)

    model.train()
    return evaluate_metrics(raw_info, total_loss, epoch)

