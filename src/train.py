from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path
import os
import argparse
import shutil

import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch

from src.data import DicomDataset, subject_split, apply_transforms
from src.model import DiagnosticModel

from src.train_utils import evaluate, conf_matrix
from src.train_utils import save_accuracies, display_curve


# ----- PARAMETERS ----------
batch_size = 16
num_workers = 8
lr = 1e-4
num_epochs = 20
val_ratio = 0.25 # % of people, not actual images

def train():
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

    #2) Create Dataset & DataLoader
    dataset = DicomDataset(data_dir)
    dataset.summarize(name = "original")
    train_dataset, val_dataset = subject_split(dataset, val_ratio=val_ratio)
    apply_transforms(train_dataset, val_dataset, method = args.aug)
    
    train_dataset.save_example(output_dir)
    return

    train_dataset.summarize(name = "Train")
    val_dataset.summarize(name = "Val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    #3) Create Model & Train it
    model = DiagnosticModel()
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device("cuda")  # GPU -- should be used if using cluster

    print(f"Using device {device}")

    model = model.to(device)

    train_results = []
    val_results = []

    for epoch in range(num_epochs):
        train_cfvalues = np.zeros(4)

        for data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not args.use_tqdm):
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # track train accuracy
            preds = torch.argmax(outputs, dim=1) # returning 0 or 1 for whichever higher
            train_cfvalues += conf_matrix(labels, preds)
        
        val_cfvalues = evaluate(model, val_loader, device)

        train_results.append(train_cfvalues)
        val_results.append(val_cfvalues)

        save_accuracies(epoch, num_epochs, loss.item(), train_cfvalues, val_cfvalues, fname=output_dir/"accuracies.text")
        display_curve(train_results, val_results, output_dir/"learning_curve.png")


if __name__ == '__main__':

    train()