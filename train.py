import os
from dotenv import load_dotenv
from tqdm import tqdm

from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim import Adam
import torch.nn as nn
import torch

from data import DicomDataset
from model import DiagnosticModel

train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 244)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 1→3 channels

])

batch_size = 16
num_workers = 4
lr = 1e-4
num_epochs = 10
val_split = 0.2

def evaluate(model, loader, device):
    """Evaluate accuracy on a given dataset loader"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    model.train()
    return correct / total if total > 0 else 0

def summarize_dataset(subset: Subset, name = "subset"):
    result = f"{name}:\n"
    result += f"Image Size: {subset[0][0].shape}\n"
    result += f"Size: {len(subset)}\n"
    pos_samples = len([label for fpath, label in subset if label == 1])
    result += f"Number of Pos Samples: {pos_samples}\n"
    result += f"Number of Neg Samples: {len(subset) - pos_samples}\n"

    print(result)

def main():

    #1) Load .env variables
    load_dotenv("/data/vision/polina/users/marcusbl/bin_class/.env")
    data_dir = os.environ["DATA_DIR"]

    #2) Create Dataset & DataLoader
    dataset = DicomDataset(data_dir, transform=train_transforms)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
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
        correct, total = 0, 0
        # for data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        for data, labels in train_loader:

            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # track train accuracy
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total if total > 0 else 0
        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Loss={loss.item():.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")


if __name__ == '__main__':
    main()

