from collections import defaultdict

from pathlib import Path
import pydicom

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, Subset

class DicomDataset(Dataset):
    """
    Parses the Dicom dataset stored at DATA_DIR in .env

    Notes:
        1. skips labels that are {'roi': 'no'}
        2. skips stacks that don't have associated csv labels
    """

    def __init__(self, root_dir_str: str, transform = None):
        self.root_dir = Path(root_dir_str)

        self.label_map = {
            '{"image_quality":"bad"}': 0,
            '{"image_quality":"good"}': 1
        }

        self.total_stacks = 0
        self.unlabeled_stacks = []

        self.transform = transform

        self.samples = [] # (fpath, label, person)
           
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
                        self.samples.append((fpath, self.label_map[label_str], person_path.stem))

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

    def display(self, idx, file_name):
        img = self[idx][0][0, :, :]
        plt.imshow(img, cmap="gray")   # show as grayscale
        plt.axis("off")                # remove axes
        plt.savefig(f"{file_name}", bbox_inches="tight", pad_inches=0)
        plt.close()
            

def subject_split(dataset: DicomDataset, val_ratio=0.2):
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

    train_subset = Subset(dataset, train_indices)
    val_subset   = Subset(dataset, val_indices)

    return train_subset, val_subset

        
        
# Image Size: torch.Size([3, 224, 244])
# Size: 6817
# Number of Pos Samples: 5105
# Number of Neg Samples: 1712

# Val:
# Image Size: torch.Size([3, 224, 244])
# Size: 1704
# Number of Pos Samples: 1261
# Number of Neg Samples: 443
