from dotenv import load_dotenv
from tqdm import tqdm
import os

import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import torch.nn as nn
import torch
from torchvision import transforms


from data import DicomDataset, subject_split
from model import DiagnosticModel

from train_utils import summarize_dataset, evaluate, conf_matrix, display_accuracies

# ----- PARAMETERS ----------
batch_size = 16
num_workers = 4
lr = 1e-4
num_epochs = 10
val_ratio = 0.2 # % of people, not actual images

train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 244)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 1→3 channels
])

def train():

    #1) Load .env variables
    load_dotenv("/data/vision/polina/users/marcusbl/bin_class/.env")
    data_dir = os.environ["DATA_DIR"]

    #2) Create Dataset & DataLoader
    dataset = DicomDataset(data_dir, transform=train_transforms)
    train_dataset, val_dataset = subject_split(dataset, val_ratio=val_ratio)
    summarize_dataset(train_dataset, "Train")
    summarize_dataset(val_dataset, "Val")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

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

    for epoch in range(num_epochs):
        train_accuracies = np.zeros(4)

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
            train_accuracies += conf_matrix(labels, preds)

        val_accuracies = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{num_epochs}\n"
            f"  Loss = {loss.item():.4f}\n"
            f"  Train: {display_accuracies(train_accuracies)}\n"
            f"  Val:   {display_accuracies(val_accuracies)}"
        )


if __name__ == '__main__':
    train()