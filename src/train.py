from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path
import os
import argparse
import json

import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import Adam
import torch.nn as nn
import torch

from data import DicomDataset, BalancedBatchSampler, split_and_augment
from model import DiagnosticModel

from train_utils import conf_matrix, generate_roc, get_info
from train_utils import print_accuracies, display_curve
from exp_utils import save_bad_examples

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ----- PARAMETERS ----------
batch_size = 16
num_workers = 8
lr = 1e-4

# The amount of people in train/val/test datasets
train_ppl_cnt = 21
val_ppl_cnt = 7
test_ppl_cnt = 2

# How to choose best validation score
val_metric = 'f1'

def setup():
    print("Beginning Setup")

    #1) Load .env and argparse variables
    load_dotenv("/data/vision/polina/users/marcusbl/bin_class/.env")
    data_dir = os.environ["DATA_DIR"]
    output_root = os.environ["OUTPUT_DIR_ROOT"]

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
        "--mask_method",
        type=str,
        help = """ 
                1) stack - make input to model scan + mask = 2 input channel
                2) mask - actually apply the mask [dataset limited to images w/ actual masks]
               """,
    )
    parser.add_argument(
        "--use_weights",
        action="store_true",
        help = "If true, downloads the weights for the model you're using from PyTorch"
    )
    
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict['batch_size'] = batch_size
    args_dict['num_workers'] = num_workers
    args_dict['lr'] = lr
    args_dict['dataset_cnts'] = [train_ppl_cnt, val_ppl_cnt, test_ppl_cnt]
    args_dict['val_metric'] = val_metric
    args_dict['in_channels'] =  2 if args.mask_method == 'stack' else 3

    # Create Output Directory
    output_dir = Path(output_root) / Path(args.out_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in: {output_dir}")

    with open(output_dir / 'params.json', 'w') as f:
        json.dump(args_dict, f, indent = 2)

    # 2) Create Dataset for Train/Validation 
    dataset = DicomDataset(data_dir, mask_method = args.mask_method)
    train_dataset, val_dataset, test_dataset = split_and_augment(dataset, train_ppl_cnt, val_ppl_cnt, 
                                                   test_ppl_cnt, aug_method=args.aug, seed = 42)
    
    train_dataset.save_examples(output_dir, num_examples = 10)
    dataset.test_data_collect(output_dir = output_dir) # testing the scans / masks align

    dataset.summarize(name = "Original")
    train_dataset.summarize(name = "Train")
    val_dataset.summarize(name = "Val")
    test_dataset.summarize(name = "Test")

    # Get Class Weights
    class_weights, sample_weights = train_dataset.get_class_weights()

    # 2a) Re-Sampling for Training
    shuffle = False
    if args.balance == 'w':
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    elif args.balance == 'b':
        sampler = BalancedBatchSampler(train_dataset.get_labels(), batch_size = batch_size)
    elif args.balance == 'o':
        sampler = None
        shuffle = True
    else: 
        raise ValueError(f"Unknown balance method: {args.balance}. Acceptable are one of w,b,o")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler = sampler, num_workers=num_workers, shuffle = shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers = num_workers)

    # 3) Device
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device("cuda")  # GPU -- should be used if using cluster

    print(f"Using device {device}")

    #4) Create Model
    model = DiagnosticModel(model_name = args.model, in_channels = args_dict['in_channels'], include_weights = args.use_weights)
    model = model.to(device)

    return model, train_loader, val_loader, test_loader, args, output_dir, class_weights

def train(model: nn.Module, train_loader, val_loader, args, output_dir: Path, device, class_weights):
    print("Beginning Train")
    num_epochs = args.epochs
    ckpt_path = output_dir / 'best_model.pth'

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,   # total epochs
        eta_min=1e-6
    )

    if args.balance == 'o':
        criterion = nn.CrossEntropyLoss(weight = class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()


    full_train = []
    full_val = []
    full_loss = []
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

        for data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch + num_epochs}", disable=not args.use_tqdm):
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
        val_cfvalues = evaluate(model, val_loader, device)
        val_metric_score = get_info(val_cfvalues)[val_metric]
        if val_metric_score > best_val_metric_score:
            best_val_metric_score = val_metric_score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                f'val_{val_metric}': best_val_metric_score,
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
        print_accuracies(epoch, num_epochs, epoch_loss, train_cfvalues, val_cfvalues, fname=output_dir/"accuracies.txt")
        display_curve(full_train, full_val, full_loss, output_dir, title = output_dir.name,
                      metrics = ['acc', 'tpr', 'fpr', 'loss', 'f1'],
                      colors = ['red', 'green', 'blue', 'black', 'orange'])

def evaluate(model: torch.nn.Module, loader, device, roc_path: Path = None, 
             ckpt_path: Path | None = None, save_path: Path | None = None):
    """
    Runs the model on the validation set and returns a list of 
    (outputs, labels) for all the runs
    
    """    
    print("Evaluating")
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    val_cfvalues = np.zeros(4)
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            
            _, preds = torch.max(outputs, dim=1)
            val_cfvalues += conf_matrix(preds, labels)

            if roc_path is not None:
                # Get probabilities for positive class (class 1)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())     

    if roc_path is not None:
        all_probs_tensor = torch.cat(all_probs)
        all_labels_tensor = torch.cat(all_labels)
        auc = generate_roc(all_probs_tensor, all_labels_tensor, fpath = roc_path, title="Full Dataset")

    model.train()

    if save_path is not None:
        with open(save_path, 'w') as f:
            json.dump({'auc': auc, 'val_cfvalues': list(val_cfvalues)}, f)

    return val_cfvalues

if __name__ == '__main__':
    model, train_loader, val_loader, test_loader, args, output_dir, class_weights = setup()
    device = next(model.parameters()).device
    ckpt_path = output_dir/'best_model.pth'

    train(model, train_loader, val_loader, args, output_dir, device, class_weights = class_weights)
    evaluate(model, test_loader, device=device, roc_path = output_dir / 'final_roc.png', 
             ckpt_path=ckpt_path, save_path = output_dir / 'test_results.json')
    save_bad_examples(model, val_loader, output_dir, ckpt_path = ckpt_path)
