from tqdm import tqdm
from pathlib import Path
import json
import time
import os, shutil
import numpy as np
from torch.optim import Adam
import torch.nn as nn
import torch
from torch.utils.data import DataLoader

from model import DiagnosticModel
from data import split_people, get_sample_dataframe
from parse_args import parse_args
from train_setup import setup
from evaluate import evaluate, ValidationTracker, evaluate_metrics
from evaluate import save_metric_info_epoch, save_metric_info_test
from display_utils import display_metrics

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# FIX WEIRD SETUP GETTING OUTPUT_DIR???

def train_and_test(model: DiagnosticModel, 
          train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
          args_dict: dict, 
          run_dir: Path, 
          device, 
          criterion: nn.Module):
    
    print("Beginning Train")
    test_dir = run_dir / 'test_info'
    test_dir.mkdir(exist_ok=True)

    epoch_metric_dir = run_dir / 'epoch_metrics'
    epoch_metric_dir.mkdir(exist_ok=True)

    start_time = time.time()

    num_epochs = args_dict['epochs']

    optimizer = Adam(model.parameters(), lr=args_dict['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,   # total epochs
        eta_min=1e-6
    )

    val_tracker = ValidationTracker(test_dir, model)
    val_tracker.add_tracker('f1', 'max')
    val_tracker.add_tracker('loss', 'min')
    val_tracker.add_tracker('auc', 'max')

    for epoch in range(num_epochs):
        train_raw_info = {
            'preds': [],
            'probs': [],
            'labels': [],
            'idxs': [],
        }

        epoch_loss = 0
        total_samples = 0
        for data, _, labels, idxs in tqdm(train_loader, desc=f"Epoch {epoch}", disable=not args_dict['use_tqdm']):
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            probs = model(data)
            loss = criterion(probs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(probs, dim=1)

            train_raw_info['preds'].extend(preds.detach().cpu().numpy().tolist())
            train_raw_info['labels'].extend(labels.detach().cpu().numpy().tolist())
            train_raw_info['probs'].extend(probs[:, 1].detach().cpu().numpy().tolist()) # prob of positive
            train_raw_info['idxs'].extend(idxs.detach().cpu().numpy().tolist())

            epoch_loss += loss.item() * data.size(0)
            total_samples += data.size(0)

        epoch_loss /= total_samples
        scheduler.step()

        train_metric_info = evaluate_metrics(train_raw_info, epoch_loss, epoch = epoch)

        # Validation
        val_metric_info  = evaluate(model, val_loader, device, criterion=criterion, epoch=epoch)
        val_tracker.update_best(val_metric_info, epoch)
        
        # Save results (metric_info & raw_info for train)
        save_metric_info_epoch(epoch_metric_dir / "data.jsonl", train_metric_info, val_metric_info)
        
        display_metrics(args_dict['output_dir'], metrics = ['auc', 'tpr', 'fpr'], name = "epoch_metrics")
        display_metrics(args_dict['output_dir'], metrics = ['loss'], name = "loss_curve")

    # Testing
    all_test_metric_info = {}
    for model_name, test_model in val_tracker.yield_best_models():
        all_test_metric_info[model_name] = evaluate(test_model, test_loader, device, criterion=criterion, save_path = test_dir / f"{model_name}_raw.csv")
    
    save_metric_info_test(test_dir, all_test_metric_info)


    return time.time() - start_time

def run_experiments(args_dict: dict):
    all_runtimes = []
    
    num_runs = args_dict['num_runs']

    data_samples_df, person_ids = get_sample_dataframe(args_dict['data_path'], dataset_types = ['R', 'BCH'])
    people_groups = split_people(person_ids, fractions = args_dict['split_fracs'], 
                                 seed = args_dict['data_split_seed'], num_runs = num_runs)

    os.makedirs(args_dict['output_dir'], exist_ok=False)

    for i in range(num_runs):
        print(f"Starting run {i}")
        run_output_dir = args_dict['output_dir'] / f'run{i}'
        run_output_dir.mkdir(parents=True, exist_ok=True)

        model, loaders, criterion = setup(args_dict, people_groups[i], run_output_dir, data_samples_df)
        train_loader, val_loader, test_loader = loaders
        
        device = args_dict['device']

        # Train & Track Time
        time_to_train = train_and_test(model, train_loader, val_loader, test_loader, 
                              args_dict, device = device, run_dir = run_output_dir, criterion=criterion)
        all_runtimes.append(time_to_train)

        # Dump Experiment-Level Info
        with open(args_dict['output_dir'] / 'info.json', 'w') as f:
            json.dump({
                "times": all_runtimes,
                "avg time": np.mean(all_runtimes),
                "train_val_test_people": people_groups,
            }, f, indent = 2)

if __name__ == '__main__':
    args_dict = parse_args()
    run_experiments(args_dict)
