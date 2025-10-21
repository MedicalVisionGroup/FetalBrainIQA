from collections import defaultdict

from pathlib import Path
import pydicom
import nibabel as nib
import nibabel.nicom.dicomreaders as dcmreaders

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torch

from src.augs.augs_list import default_img_transform
 
class DicomDataset(Dataset):
    """
    Parses the Dicom dataset stored at DATA_DIR in .env

    Notes:
        1. skips labels that are {'roi': 'no'}
        2. skips stacks that don't have associated csv labels

    Nifti files gives the masks 
    Dicom files give the images
    CSV file give the labels for the dicom files
    """

    def __init__(self, root_dir_str: str, samples:list = None, max_samples = None):
        self.root_dir = Path(root_dir_str)

        self.label_map = {
            '{"image_quality":"bad"}': 0,
            '{"image_quality":"good"}': 1
        } # ignore the roi label

        self.total_stacks = 0
        self.unlabeled_stacks = []
        self.max_samples = max_samples

        self.use_transform = False
        self.transform = None
        self.default_transform = default_img_transform

        self.scan_dict = {}
        if not samples:
            samples = self._load_samples() # (fpath, label, person)
            
        self.samples = samples

           
    def _load_samples(self):
        samples = []
        # Dataset/ -> Person/ -> Stack/ -> CSV, Niftis, & Dicoms/ -> Dicom Files 
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

                # Dealing w/ Nifti Files (W, H, num_images)
                scan3d = nib.load(stack_path / "converted.nii.gz").get_fdata()
                scan3d_mask = nib.load(stack_path / "converted_mask.nii.gz").get_fdata() 

                # Dicom Folder
                dicoms_dir_path = stack_path / "dicoms"

                assert len(list(dicoms_dir_path.glob("*.dcm"))) == scan3d.shape[-1]
                # sort the files so they match the order of the nifti
                sorted_dicom_files = sorted(dicoms_dir_path.glob("*.dcm"), key = lambda s: s.name[:4])
                for scan_num, fpath in enumerate(sorted_dicom_files):
                    row = label_df.loc[label_df["External ID"] == (fpath.stem + ".png")]
                    label_str = row["Label"].values[0]

                    if label_str in self.label_map:
                        samples.append((fpath, self.label_map[label_str], person_path.stem))
                        dicom = pydicom.dcmread(fpath)
                        csa = dcmreaders.read_csa_header(dicom, 'image')
                        print(csa.keys())


                        # break if reached max samples
                        if self.max_samples is not None and len(samples) >= self.max_samples:
                            return samples
                    

        return samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        fpath, label, _ = self.samples[idx]

        dicom = pydicom.dcmread(fpath)
        img = dicom.pixel_array.astype(dtype=np.float32)

        if self.use_transform:
            img = self.transform(img)
        else:
            img = self.default_transform(img)

        return img, label
    
    def set_transform(self, transform):
        self.transform = transform     
        self.use_transform = True  

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

    def save_examples(self, dir_path: Path, num_examples: int = 3, num_augs: int = 5):
        dir_path.mkdir(parents=True, exist_ok=True)

        ncols = num_augs + 1  # base + augmentations
        nrows = num_examples

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(3 * ncols, 3 * nrows)
        )

        base_idxs = np.random.randint(0, len(self)-1, size = num_examples)

        for row, base_idx in enumerate(base_idxs):
            # Base image (no augmentation)
            self.use_transform = False
            base_img, base_label = self[base_idx]

            # Augmented images
            self.use_transform = True
            aug_imgs = [self[base_idx][0] for _ in range(num_augs)]

            # Collect
            all_imgs = [base_img] + aug_imgs
            titles = ["base"] + [f"aug {i}" for i in range(num_augs)]

            for col, (img, title) in enumerate(zip(all_imgs, titles)):
                ax = axes[row, col] if nrows > 1 else axes[col]
                ax.imshow(F.to_pil_image(img))
                if row == 0:  # only add column titles on top row
                    ax.set_title(title)
                ax.axis("off")

            # Add y-label to show class (once per row)
            label_str = "GOOD" if base_label == 1 else "BAD"
            axes[row, 0].set_title(f"Idx ({base_idx}) = {label_str}")

        plt.suptitle(f"{dir_path.name}", fontsize=18, weight="bold")
        plt.tight_layout()
        plt.savefig(dir_path / "aug_examples.png")
        plt.close()

    def get_class_weights(self):
        labels = [sample[1] for sample in self.samples]  # extract labels
        class_counts = torch.bincount(torch.tensor(labels))
        class_weights = 1.0 / class_counts.float()   # inverse frequency
        sample_weights = class_weights[torch.tensor(labels)]

        return class_weights, sample_weights
    
    def test_data_collect(self):
        size = 10
        idxs = np.random.randint(0, len(self)-1, size = size)
        error = 0
        for idx in idxs:
            fpath, _, _ = self.samples[idx]
            dicom = pydicom.dcmread(fpath)
            print(dicom)
            break
            img_dicom = dicom.pixel_array.astype(dtype=np.float32)
            img_nifti = self.scan_dict[fpath]
            
            print("Nifti:", np.min(img_nifti), np.max(img_nifti))
            print("Dicom:", np.min(img_dicom), np.max(img_dicom))

            print(np.mean((img_dicom - img_nifti) ** 2))
            error += np.mean((img_dicom - img_nifti) ** 2)

        assert (error / size) < .01

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
    # np.random.shuffle(unique_people) # shuffles the list of people for true random selection

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



        
        
# Image Size: torch.Size([3, 224, 244])
# Size: 6817
# Number of Pos Samples: 5105
# Number of Neg Samples: 1712

# Val:
# Image Size: torch.Size([3, 224, 244])
# Size: 1704
# Number of Pos Samples: 1261
# Number of Neg Samples: 443
