from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path
import os
import argparse

import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import Adam
import torch.nn as nn
import torch

from src.data import DicomDataset, subject_split
from src.augs.apply_augs import apply_augs
from src.model import DiagnosticModel

from src.train_utils import conf_matrix, generate_roc, get_info
from src.train_utils import print_accuracies, display_curve
from src.exp_utils import save_bad_examples

# ----- PARAMETERS ----------
batch_size = 16
num_workers = 8
lr = 1e-4
val_ratio = 0.25 # % of people, not actual images

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
        "--resample", 
        action="store_true",
        help="Resamples the training process to be balanced between both classes"
    )
    parser.add_argument(
        "--reweight",
        action="store_true",
        help = "Reweights the training objective function to account for class imbalance "
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
    
    args = parser.parse_args()
    output_dir = Path(output_root) / Path(args.out_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in: {output_dir}")

    # 2) Create Dataset & DataLoader
    dataset = DicomDataset(data_dir)
    dataset.summarize(name = "original")
    dataset.test_data_collect(output_dir = output_dir)

    train_dataset, val_dataset = subject_split(dataset, val_ratio=val_ratio)
    apply_augs(train_dataset, val_dataset, method = args.aug)
    
    train_dataset.save_examples(output_dir, num_examples = 10)
    
    train_dataset.summarize(name = "Train")
    val_dataset.summarize(name = "Val")

    # Get Class Weights
    class_weights, sample_weights = train_dataset.get_class_weights()

    # 2a) Re-Sampling for Training

    if args.resample:
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    else:
        sampler = None
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler = sampler, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(dataset, batch_size = batch_size, shuffle=False, num_workers = num_workers)

    # 3) Device
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device("cuda")  # GPU -- should be used if using cluster

    print(f"Using device {device}")

    #4) Create Model
    model = DiagnosticModel(model_name = args.model)
    model = model.to(device)

    return model, train_loader, val_loader, test_loader, args, output_dir, class_weights

def train(model: nn.Module, train_loader, val_loader, args, output_dir: Path, device, class_weights):
    print("Beginning Train")
    num_epochs = args.epochs
    ckpt_path = output_dir / 'best_model.pth'

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=60,   # total epochs
        eta_min=1e-6
    )

    if args.reweight:
        criterion = nn.CrossEntropyLoss(weight = class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()


    full_train = []
    full_val = []
    full_loss = []
    best_val_acc = 0.0
    start_epoch = 0

    # If there's already a model saved, start from there
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['val_acc']
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
        val_acc = get_info(val_cfvalues)['acc']
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'full_train': full_train,
                'full_val': full_val,
                'full_loss': full_loss,
            }, ckpt_path)
            print(f"Saved new best model at epoch {epoch+1} with val_acc {val_acc:.4f}")

        scheduler.step()
        
        # Tracking Results & Displaying
        full_train.append(train_cfvalues)
        full_val.append(val_cfvalues)
        full_loss.append(epoch_loss)
        print_accuracies(epoch, num_epochs, epoch_loss, train_cfvalues, val_cfvalues, fname=output_dir/"accuracies.txt")
        display_curve(full_train, full_val, full_loss, output_dir, title = output_dir.name)

def evaluate(model: torch.nn.Module, loader, device, roc_path: Path = None, ckpt_path: Path | None = None):
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
        generate_roc(all_probs_tensor, all_labels_tensor, fpath = roc_path, title="Full Dataset")

    model.train()

    return val_cfvalues

if __name__ == '__main__':
    model, train_loader, val_loader, test_loader, args, output_dir, class_weights = setup()
    device = next(model.parameters()).device
    ckpt_path = output_dir/'best_model.pth'

    train(model, train_loader, val_loader, args, output_dir, device, class_weights = class_weights)
    evaluate(model, test_loader, device=device, roc_path = output_dir / 'final_roc.png', ckpt_path=ckpt_path)
    save_bad_examples(model, val_loader, output_dir, ckpt_path = ckpt_path)


# prec = # correct / predicted positives
# recall = # correct / true positive

# prec < recall -> overpredicting positive! 