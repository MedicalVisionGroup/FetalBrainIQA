from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path
import os
import argparse
import json
import time

import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import Adam
import torch.nn as nn
import torch

from data import DicomDataset, BalancedBatchSampler, split
from model import DiagnosticModel

from train_utils import conf_matrix, generate_roc, get_info
from train_utils import print_accuracies, display_curve
from exp_utils import save_bad_examples
from augs_list import get_color_transform_list, get_spatial_transform_list

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ----- PARAMETERS ----------
batch_size = 32
num_workers = 8
lr = 1e-4

# The amount of people in train/val/test datasets
train_ppl_cnt = 20
val_ppl_cnt = 5
test_ppl_cnt = 5

# How to choose best validation score
val_metric = 'f1'

def parse_args():
    #1) Load .env for data/output directories
    load_dotenv("/data/vision/polina/users/marcusbl/bin_class/.env")
    data_dir = Path(os.environ["DATA_DIR"])
    output_root = Path(os.environ["OUTPUT_DIR_ROOT"])

    # 2) Parse the args
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs",       # default if not given
        help="Subdirectory inside of outputs to save results"
    )
    parser.add_argument(
        "--aug",
        type=str,
        default='',
        help='Augmentation types: s-spatial, c-color'
    )
    parser.add_argument(
        "--use_tqdm",
        action="store_true",    # becomes True if flag is present
        help="Enable tqdm progress bars"
    )
    parser.add_argument(
        "--balance", 
        type=str,
        help= """
                1) w - resample w/ inv freq weights
                2) o - update the objective w/ the weights
                3) b - balance the training process exactly to 50-50
              """
    )
    parser.add_argument(
        "--model",
        type=str,
        default='resnet18',
        help='Specify the type of model using for classification'
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help='Specify the # of epochs using in training'
    )
    parser.add_argument(
        "--use_weights",
        action="store_true",
        help = "If true, downloads the weights for the model you're using from PyTorch"
    )
    parser.add_argument(
        "--mask_method",
        type=str,
        help = """ 
                1) stack - make input to model scan + mask = 2 input channel
                2) mask - actually apply the mask [dataset limited to images w/ actual masks]
               """,
    )
    parser.add_argument(
        "--norm_method",
        type=str,
        help = "Type of normalization method: min-max or peak-squash"
    )
    parser.add_argument(
        "--masked_norm",
        action="store_true",
        help = "True/False should normalize wrt the mask and not the entire image"
    )
    parser.add_argument(
        "--perc_norm",
        type=float,
        default=0,
        help = "The quantile for min-max normalization"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help = "The # of runs to run the training experimentation "
    )
    parser.add_argument(
        "--balance_val",
        action="store_true",
        help = "Will balance validation set to be 50-50 at each sampling"
    )
    

    args = parser.parse_args()
    args_dict = vars(args)
    args_dict['batch_size'] = batch_size
    args_dict['num_workers'] = num_workers
    args_dict['lr'] = lr
    args_dict['dataset_cnts'] = [train_ppl_cnt, val_ppl_cnt, test_ppl_cnt]
    args_dict['val_metric'] = val_metric
    args_dict['in_channels'] =  2 if args.mask_method == 'stack' else 3
    args_dict['data_dir'] = data_dir
    args_dict['output_dir'] = output_root / args.out_dir

    return args_dict

def setup(args_dict: dict):
    print("Beginning Setup")

    # Create Output Directory
    output_dir = args_dict['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in: {output_dir}")

    with open(output_dir / 'params.json', 'w') as f:
        args_copy = args_dict.copy()
        args_copy['output_dir'] = str(args_copy['output_dir'])
        args_copy['data_dir'] = str(args_copy['data_dir'])
        json.dump(args_copy, f, indent=2)

    # 2) Create Dataset for Train/Validation & Apply Augmentations
    dataset = DicomDataset(args_dict['data_dir'])
    dataset.set_norm(mask_method = args_dict['mask_method'], 
                     norm_method = args_dict['norm_method'], 
                     masked_norm = args_dict['masked_norm'],
                     perc_norm   = args_dict['perc_norm'])
    
    train_dataset, val_dataset, test_dataset = split(dataset, train_ppl_cnt, val_ppl_cnt, 
                                                   test_ppl_cnt, seed = 42)
    augmentation_list = []
    if 's' in args_dict['aug']:
        print("Applying Spatital Augmentations")
        augmentation_list.extend(get_spatial_transform_list())
    if 'c' in args_dict['aug']:
        print("Applying Color Augmentations")
        augmentation_list.extend(get_color_transform_list(args_dict['mask_method']))

    train_dataset.set_aug(augmentation_list)
    
    # train_dataset.save_examples(output_dir, num_examples = 10)
    dataset.test_data_collect(output_dir = output_dir) # testing the scans / masks align

    dataset.summarize(name = "Original")
    train_dataset.summarize(name = "Train")
    val_dataset.summarize(name = "Val")
    test_dataset.summarize(name = "Test")

    # Get Class Weights
    class_weights, sample_weights = train_dataset.get_class_weights()

    # 2a) Re-Sampling for Training
    shuffle = False
    if args_dict['balance'] == 'w':
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    elif args_dict['balance'] == 'b':
        sampler = BalancedBatchSampler(train_dataset.get_labels(), batch_size = batch_size)
    elif args_dict['balance'] == 'o':
        sampler = None
        shuffle = True
    else: 
        raise ValueError(f"Unknown balance method: {args_dict['balance']}. Acceptable are one of w,b,o")
    
    val_sampler = None
    if args_dict['balance_val']:
        print('Will do validation balancing')
        val_sampler = BalancedBatchSampler(val_dataset.get_labels(), batch_size = batch_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler = sampler, num_workers=num_workers, shuffle = shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler = val_sampler, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, num_workers = num_workers, shuffle=False)

    # 3) Device
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device("cuda")  # GPU -- should be used if using cluster

    print(f"Using device {device}")

    #4) Create Model
    model = DiagnosticModel(model_name = args_dict['model'], 
                            in_channels = args_dict['in_channels'], 
                            include_weights = args_dict['use_weights'])
    model = model.to(device)

    # 5) Create Loss Function
    if args_dict['balance'] == 'o':
        criterion = nn.CrossEntropyLoss(weight = class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    return model, train_loader, val_loader, test_loader, criterion

def train(model: nn.Module, train_loader, val_loader, args_dict: dict, run_dir: Path, device, criterion):
    print("Beginning Train")
    start_time = time.time()

    num_epochs = args_dict['epochs']
    ckpt_path = run_dir / 'best_model.pth'

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,   # total epochs
        eta_min=1e-6
    )

    full_train = []
    full_val = []
    full_loss = []
    full_val_loss = []
    best_val_metric_score = 0.0
    start_epoch = 0

    # If there's already a model saved, start from there
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_metric_score = checkpoint[val_metric]
        full_train = checkpoint['full_train']
        full_val = checkpoint['full_val']
        full_loss = checkpoint['full_loss']

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
        val_cfvalues, val_loss  = evaluate(model, val_loader, device, criterion=criterion)

        full_val_loss.append(val_loss)
        val_metric_score = get_info(val_cfvalues)[val_metric]
        if val_metric_score >= best_val_metric_score or epoch == start_epoch:
            best_val_metric_score = val_metric_score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                 val_metric: best_val_metric_score,
                'full_train': full_train,
                'full_val': full_val,
                'full_loss': full_loss,
            }, ckpt_path)
            print(f"Saved new best model at epoch {epoch+1} with {val_metric} {val_metric_score:.4f}")

        scheduler.step()
        
        # Tracking Results & Displaying
        full_train.append(train_cfvalues)
        full_val.append(val_cfvalues)
        full_loss.append(epoch_loss)
        print_accuracies(epoch, num_epochs, epoch_loss, train_cfvalues, val_cfvalues, fname=run_dir/"accuracies.txt")
        display_curve(full_train, full_val, full_loss, full_val_loss, run_dir, title = run_dir.name,
                      metrics = ['acc', 'tpr', 'fpr', 'loss', 'f1'],
                      colors = ['red', 'green', 'blue', 'black', 'orange'],
                      val_metric = val_metric)
        
    return time.time() - start_time

def evaluate(model: torch.nn.Module, loader, device, 
             criterion = None,
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

            if roc_path is not None:
                # Get probabilities for positive class (class 1)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())     

    loss /= total_items

    # Save ROC Graph
    if roc_path is not None:
        all_probs_tensor = torch.cat(all_probs)
        all_labels_tensor = torch.cat(all_labels)
        auc = generate_roc(all_probs_tensor, all_labels_tensor, fpath = roc_path, title="Full Dataset")

    # Dump Values
    if save_path is not None:
        with open(save_path, 'w') as f:
            json.dump({'auc': auc, 'val_cfvalues': list(val_cfvalues)}, f)

    model.train()

    return val_cfvalues, loss

def run_experiments(args_dict: dict):
    all_times = []

    for i in range(args_dict['num_runs']):
        model, train_loader, val_loader, test_loader, criterion = setup(args_dict)
        device = next(model.parameters()).device

        run_output_dir = args_dict['output_dir'] / f'run{i}'
        ckpt_path = run_output_dir / 'best_model.pth'

        run_output_dir.mkdir(parents=True, exist_ok=True)

        # Train & Track Time
        time_to_train = train(model, train_loader, val_loader, args_dict, device = device, run_dir = run_output_dir, criterion=criterion)
        all_times.append(time_to_train)

        # Test Set
        evaluate(model, test_loader, device=device, 
                 criterion = criterion,
                 roc_path = run_output_dir / 'final_roc.png', 
                 ckpt_path=ckpt_path,
                 save_path = run_output_dir / 'test_results.json')

        # Save Bad Examples!
        save_bad_examples(model, val_loader, run_output_dir, ckpt_path = ckpt_path)

    with open(args_dict['output_dir'] / 'info.json', 'w') as f:
        json.dump({
            "times": all_times,
            "avg time": np.mean(all_times),
            "indexes": []
        }, f, indent = 2)

def run_experiments_k_fold():
    pass

if __name__ == '__main__':
    args_dict = parse_args()
    run_experiments(args_dict)
