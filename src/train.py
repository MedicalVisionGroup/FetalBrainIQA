from tqdm import tqdm
from pathlib import Path
import json
import time

import numpy as np
from torch.optim import Adam
import torch.nn as nn
import torch
from torch.utils.data import DataLoader

from data import get_people_groups
from model import DiagnosticModel

from train_setup import setup, parse_args
from train_utils import conf_matrix, get_info
from train_utils import print_accuracies, display_curve
from exp_utils import save_bad_examples
from evaluate import evaluate

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def train(model: DiagnosticModel, 
          ckpt_path: Path,
          train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
          args_dict: dict, 
          run_dir: Path, 
          device, 
          criterion: nn.Module):
    
    print("Beginning Train")
    start_time = time.time()

    start_epoch = 0
    num_epochs = args_dict['epochs']

    optimizer = Adam(model.parameters(), lr=args_dict['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,   # total epochs
        eta_min=1e-6
    )

    val_metric = args_dict['val_metric']
    full_train = []
    full_train_loss = []

    full_val = []
    full_val_loss = []
    full_val_auc = []

    full_test = []
    full_test_loss = []
    full_test_auc = []

    best_val_metric_score = 0.0

    # If there's already a model saved, start from there
    # if ckpt_path.exists():
    #     checkpoint = torch.load(ckpt_path, map_location=device)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     start_epoch = checkpoint['epoch']
    #     best_val_metric_score = checkpoint[val_metric]
    #     full_train = checkpoint['full_train']
    #     full_val = checkpoint['full_val']
    #     full_loss = checkpoint['full_loss']

    for epoch in range(start_epoch, start_epoch + num_epochs):
        train_cfvalues = np.zeros(4)
        epoch_loss = 0
        total_samples = 0

        for data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch + num_epochs}", disable=not args_dict['use_tqdm']):
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
            
            _, preds = torch.max(outputs, dim=1)
            train_cfvalues += conf_matrix(preds, labels)

        epoch_loss /= total_samples

        # Validation
        val_cfvalues, val_loss, val_auc  = evaluate(model, val_loader, device, criterion=criterion)
        test_cfvalues, test_loss, test_auc = evaluate(model, test_loader, device, criterion=criterion)

        # Collecting All Results
        full_train.append(train_cfvalues)
        full_train_loss.append(epoch_loss)

        full_val.append(val_cfvalues)
        full_val_loss.append(val_loss)
        full_val_auc.append(val_auc)

        full_test.append(test_cfvalues)
        full_test_loss.append(test_loss)
        full_test_auc.append(test_auc)

        # Compare to Current Best Model
        # val_metric_score = get_info(val_cfvalues)[val_metric]
        # if val_metric_score >= best_val_metric_score or epoch == start_epoch:
        #     best_val_metric_score = val_metric_score
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #          val_metric: best_val_metric_score,
        #         'full_train': full_train,
        #         'full_val': full_val,
        #         'full_loss': full_loss,
        #     }, ckpt_path)
        #     print(f"Saved new best model at epoch {epoch+1} with {val_metric} {val_metric_score:.4f}")

        scheduler.step()
        
        print_accuracies(epoch, num_epochs, epoch_loss, train_cfvalues, val_cfvalues, fname=run_dir/"accuracies.txt")
        display_curve(full_train, full_val, full_train_loss, full_val_loss, full_val_auc,
                      run_dir, title = f'Val ({run_dir})',
                      metrics = ['acc', 'tpr', 'fpr', 'loss', 'f1', 'auc'],
                      colors = ['red', 'green', 'blue', 'black', 'orange', 'pink'],
                      val_metric = val_metric)
        
        display_curve(full_train, full_test, full_train_loss, full_test_loss, full_test_auc,
                      run_dir, title = f'Test ({run_dir})',
                      metrics = ['acc', 'tpr', 'fpr', 'loss', 'f1', 'auc'],
                      colors = ['red', 'green', 'blue', 'black', 'orange', 'pink'],
                      val_metric = val_metric)

    return time.time() - start_time

def run_experiments(args_dict: dict, use_k_fold: bool = False):
    all_runtimes = []

    n_rounds = args_dict['num_runs']
    seed = args_dict['data_split_seed']
    num_train, num_val, num_test = args_dict['dataset_cnts']

    people_groups = get_people_groups(num_train, num_val, num_test, 
                                      n_rounds = n_rounds, seed = seed, use_k_fold=use_k_fold)

    for i in range(n_rounds):
        run_output_dir = args_dict['output_dir'] / f'run{i}'
        run_output_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = run_output_dir / 'best_model.pth'

        model, loaders, criterion = setup(args_dict, people_groups[i], run_output_dir)
        train_loader, val_loader, test_loader = loaders
        
        device = next(model.parameters()).device

        # Train & Track Time
        time_to_train = train(model, ckpt_path, train_loader, val_loader, test_loader, 
                              args_dict, device = device, run_dir = run_output_dir, criterion=criterion)
        all_runtimes.append(time_to_train)

        # Test Set
        evaluate(model, test_loader, device=device, 
                 criterion = criterion,
                 roc_path = run_output_dir / 'final_roc.png', 
                 ckpt_path=ckpt_path,
                 save_path = run_output_dir / 'test_results.json')

        # Save Bad Examples!
        fp_idxs, fn_idxs = save_bad_examples(model, test_loader, run_output_dir, ckpt_path = ckpt_path)
        with open(run_output_dir / 'info.json', 'r') as f:
            run_info_dict = json.load(f)
        run_info_dict["fp_idxs"] = [int(idx) for idx in fp_idxs]
        run_info_dict["fn_idxs"] = [int(idx) for idx in fn_idxs]
        run_info_dict['ppl_ids'] = {
            "train": people_groups[i][0],
            "val": people_groups[i][1],
            "test": people_groups[i][2],
        }
        with open(run_output_dir / 'info.json', 'w') as f:
            json.dump(run_info_dict, f, indent = 2)

        # Dump Experiment-Level Info
        with open(args_dict['output_dir'] / 'info.json', 'w') as f:
            json.dump({
                "times": all_runtimes,
                "avg time": np.mean(all_runtimes),
                "train_val_test_people": people_groups,
            }, f, indent = 2)

if __name__ == '__main__':
    args_dict = parse_args()
    run_experiments(args_dict, use_k_fold=args_dict['k_fold'])
