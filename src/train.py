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

from src.data import DicomDataset, subject_split, apply_augs
from src.model import DiagnosticModel

from src.train_utils import conf_matrix, generate_roc, get_info
from src.train_utils import print_accuracies, display_curve


# ----- PARAMETERS ----------
batch_size = 16
num_workers = 8
lr = 1e-4
num_epochs = 20
val_ratio = 0.25 # % of people, not actual images

def setup():
    print("Beginning Training")

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
    
    args = parser.parse_args()
    output_dir = Path(output_root) / Path(args.out_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in: {output_dir}")

    # 2) Create Dataset & DataLoader
    dataset = DicomDataset(data_dir)
    dataset.summarize(name = "original")
    train_dataset, val_dataset = subject_split(dataset, val_ratio=val_ratio)
    apply_augs(train_dataset, val_dataset, method = args.aug)
    
    train_dataset.save_examples(output_dir)

    train_dataset.summarize(name = "Train")
    val_dataset.summarize(name = "Val")

    # 2a) Re-Sampling for Training
    labels = [sample[1] for sample in train_dataset]  # extract labels
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()   # inverse frequency
    sample_weights = class_weights[torch.tensor(labels)]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler = sampler, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(dataset, batch_size = batch_size, shuffle=False, num_workers = num_workers)

    # 3) Device
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device("cuda")  # GPU -- should be used if using cluster

    print(f"Using device {device}")

    #4) Create Model
    model = DiagnosticModel()
    model = model.to(device)

    return model, train_loader, val_loader, test_loader, args, output_dir, device

def train(model: nn.Module, train_loader, val_loader, args, output_dir: Path, device):
    checkpoint_path = output_dir / 'best_model.pth'

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    full_train = []
    full_val = []
    full_loss = []
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        train_cfvalues = np.zeros(4)
        total_loss = 0
        total_samples = 0

        for data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not args.use_tqdm):
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
            
            _, preds = torch.max(outputs, dim=1)
            train_cfvalues += conf_matrix(preds, labels)

        epoch_loss = total_loss / total_samples

        # Validation
        val_cfvalues = evaluate(model, val_loader, device)
        val_acc = get_info(val_cfvalues)['acc']
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc
            }, checkpoint_path)
            print(f"Saved new best model at epoch {epoch+1} with val_acc {val_acc:.4f}")


        # Tracking Results & Displaying
        full_train.append(train_cfvalues)
        full_val.append(val_cfvalues)
        full_loss.append(epoch_loss)
        print_accuracies(epoch, num_epochs, epoch_loss, train_cfvalues, val_cfvalues, fname=output_dir/"accuracies.text")
        display_curve(full_train, full_val, full_loss, output_dir/"learning_curve.png")

def evaluate(model: torch.nn.Module, loader, device, roc_path: Path = None, ckpt_path: Path | None = None):
    """
    Runs the model on the validation set and returns a list of 
    (outputs, labels) for all the runs
    
    """    
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
        generate_roc(all_probs_tensor, all_labels_tensor, fpath = roc_path)

    model.train()

    return val_cfvalues

if __name__ == '__main__':
    model, train_loader, val_loader, test_loader, args, output_dir, device = setup()
    evaluate(model, test_loader, device=device, roc_path = output_dir / 'roc1.png')
    train(model, train_loader, val_loader, args, output_dir, device)
    evaluate(model, test_loader, device=device, roc_path = output_dir / 'roc2.png', ckpt_path=output_dir/'best_model.pth')
