from dotenv import load_dotenv
from tqdm import tqdm
import os

import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch

from data import DicomDataset, subject_split, apply_transforms
from model import DiagnosticModel

from train_utils import evaluate, conf_matrix
from train_utils import display_accuracies, display_curve


# 2) Run & compare the different approaches? 
# 3) Ensure things are happening the way we want it to?  -- show pictures of the augmentations?


# ----- PARAMETERS ----------
batch_size = 16
num_workers = 8
lr = 1e-4
num_epochs = 15
val_ratio = 0.2 # % of people, not actual images

def train():
    
    #1) Load .env variables
    load_dotenv("/data/vision/polina/users/marcusbl/bin_class/.env")
    data_dir = os.environ["DATA_DIR"]

    #2) Create Dataset & DataLoader
    dataset = DicomDataset(data_dir)
    dataset.summarize(name = "original")
    train_dataset, val_dataset = subject_split(dataset, val_ratio=val_ratio)
    apply_transforms(train_dataset, val_dataset, method = 'cs')

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
    elif torch.backends.mps.is_available():
        device = torch.device('mps')

    print(f"Using device {device}")

    model = model.to(device)

    train_results = []
    val_results = []

    for epoch in range(num_epochs):
        train_cfvalues = np.zeros(4)

        # for data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        for data, labels in train_loader:
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

        print(
            f"Epoch {epoch+1}/{num_epochs}\n"
            f"  Loss = {loss.item():.4f}\n"
            f"  Train: {display_accuracies(train_cfvalues)}\n"
            f"  Val:   {display_accuracies(val_cfvalues)}"
        )

        display_curve(train_results, val_results, "learning_curve.png")

    


if __name__ == '__main__':
    train()