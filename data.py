from pathlib import Path
import pydicom

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import torch

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

        self.samples = []

        def get_label_df(stack_path: Path) -> pd.DataFrame | None:
            """
            Gets the DataFrame that stores the labels for 
            each of the dicoms in the stack
            """
            for f in stack_path.glob("*.csv"):
                return pd.read_csv(f)
            
            self.unlabeled_stacks.append(stack_path)
            return None     
           
        # Dataset/ -> Person/ -> Stack/ -> CSV & Dicoms/ -> Dicom Files 
        for person_path in Path(self.root_dir).iterdir():
            if not person_path.is_dir():
                continue

            for stack_path in person_path.iterdir():
                if not stack_path.is_dir():
                    continue

                self.total_stacks += 1

                label_df = get_label_df(stack_path)
                if label_df is None:
                    continue

                dicoms_dir_path = stack_path / "dicoms"
                for fpath in dicoms_dir_path.glob("*.dcm"):
                    row = label_df.loc[label_df["External ID"] == (fpath.stem + ".png")]
                    label_str = row["Label"].values[0]

                    if label_str in self.label_map:
                        self.samples.append((fpath, self.label_map[label_str]))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        fpath, label = self.samples[idx]

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
            
