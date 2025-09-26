from collections import defaultdict

from pathlib import Path
import pydicom

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

class DicomDataset(Dataset):
    """
    Parses the Dicom dataset stored at DATA_DIR in .env

    Notes:
        1. skips labels that are {'roi': 'no'}
        2. skips stacks that don't have associated csv labels
    """

    def __init__(self, root_dir_str: str, samples:list = None):
        self.root_dir = Path(root_dir_str)

        self.label_map = {
            '{"image_quality":"bad"}': 0,
            '{"image_quality":"good"}': 1
        }

        self.total_stacks = 0
        self.unlabeled_stacks = []

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((244, 244)),
        ])

        if not samples:
            samples = self._load_samples() # (fpath, label, person)
            
        self.samples = samples
           
    def _load_samples(self):
        samples = []
        # Dataset/ -> Person/ -> Stack/ -> CSV & Dicoms/ -> Dicom Files 
        for person_path in Path(self.root_dir).iterdir():
            if not person_path.is_dir():
                continue

            for stack_path in person_path.iterdir():
                if not stack_path.is_dir():
                    continue

                self.total_stacks += 1

                csv_file = next(stack_path.glob("*.csv"), None)
                if csv_file is None:
                    self.unlabeled_stacks.append(stack_path)
                    continue

                label_df = pd.read_csv(csv_file)

                dicoms_dir_path = stack_path / "dicoms"
                for fpath in dicoms_dir_path.glob("*.dcm"):
                    row = label_df.loc[label_df["External ID"] == (fpath.stem + ".png")]
                    label_str = row["Label"].values[0]

                    if label_str in self.label_map:
                        samples.append((fpath, self.label_map[label_str], person_path.stem))
            
        return samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        fpath, label, _ = self.samples[idx]

        img = pydicom.dcmread(fpath).pixel_array.astype(dtype=np.float32)

        if self.transform:
            img = self.transform(img)

        return img, label
    
    def set_transform(self, transform):
        self.transform = transform       

    def show(self, idx, file_name):
        img = self[idx][0][0, :, :]
        plt.imshow(img, cmap="gray")   # show as grayscale
        plt.axis("off")                # remove axes
        plt.savefig(f"{file_name}", bbox_inches="tight", pad_inches=0)
        plt.close()
            
    def summarize(self, name = "subset"):
        result = f"\n{name}:\n"
        result += f"Image Size: {self[0][0].shape}\n"
        result += f"Size: {len(self)}\n"
        pos_samples = len([label for _, label, _ in self.samples if label == 1])
        result += f"Number of Pos Samples: {pos_samples}\n"
        result += f"Number of Neg Samples: {len(self) - pos_samples}\n"
        result += f"e.g. Max Value: {torch.max(self[0][0])}"

        print(result)

    def get_subset(self, indices):
        select_samples = [self.samples[i] for i in indices]
        subset = DicomDataset(self.root_dir, samples = select_samples)

        return subset

def subject_split(dataset: DicomDataset, val_ratio:float=0.2):
    """
    Split dataset into train/val subsets by person.
    Ensures all images of a person are in the same subset.
    """
    # Group indices by person
    person_to_indices = defaultdict(list)
    for idx, (_, _, person) in enumerate(dataset.samples):
        person_to_indices[person].append(idx)

    unique_people = list(person_to_indices.keys())

    # Split people
    n_val = max(1, int(len(unique_people) * val_ratio))
    val_people = set(unique_people[:n_val])
    train_people = set(unique_people[n_val:])

    # Flatten indices
    train_indices = [idx for p in train_people for idx in person_to_indices[p]]
    val_indices   = [idx for p in val_people   for idx in person_to_indices[p]]

    train_dataset = dataset.get_subset(indices = train_indices)
    val_dataset   = dataset.get_subset(indices = val_indices)

    return train_dataset, val_dataset

# ------- APPLYING TRASNFORMATIONS -----------
class MinMaxNormalize:
    def __init__(self, min_val=0.0, max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, img):
        # img assumed to be a torch.Tensor (C, H, W)
        img_min = img.min()
        img_max = img.max()
        img = (img - img_min) / (img_max - img_min)  # scale to 0-1
        img = img * (self.max_val - self.min_val) + self.min_val
        return img
    
def apply_transforms(train_dataset: DicomDataset, val_dataset: DicomDataset, method = '') -> None:
    """
    Applies a series of transformations

    1) Calculate mean/std from train_dataset & applies normalization
    2) Spatial augmentations to train if 's' in method
    3) Color   augmentations to train if 'c' in method
    4) Duplicates the img to 3D for the ResNet18

    """

    basics = [
        transforms.ToTensor(), # doesn't scale 
        transforms.Resize((244, 244)),
        MinMaxNormalize(0.0, 1.0),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ]

    spatial_transform = [
        transforms.RandomHorizontalFlip(p=0.5),  
        transforms.RandomVerticalFlip(p=0.5),    
        transforms.RandomRotation(degrees=15),       
        transforms.RandomAffine(
            degrees = 0,
            translate = (0.05, 0.05), # 5 percent in both directions
            scale = (0.9, 1.1)        # 10% scale in either direction 
        )
    ]

    color_transform = [
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2
        )                                   # random color changes
    ]

    augmentations = []
    if 's' in method:
        print("Applying Spatital Augmentations")
        augmentations.extend(spatial_transform)
    if 'm' in method:
        print("Applying Color Augmentations")
        augmentations.extend(color_transform)

    train_transform = basics[:-1] + augmentations + basics[-1:]
    val_transform = basics

    train_dataset.transform = transforms.Compose(train_transform)
    val_dataset.transform = transforms.Compose(val_transform)



        
        
# Image Size: torch.Size([3, 224, 244])
# Size: 6817
# Number of Pos Samples: 5105
# Number of Neg Samples: 1712

# Val:
# Image Size: torch.Size([3, 224, 244])
# Size: 1704
# Number of Pos Samples: 1261
# Number of Neg Samples: 443
