from collections import defaultdict
from pathlib import Path
import os
from tqdm import tqdm

import json

import math
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision import transforms
import torch

from src.augs.augs_list import get_default_transform_list
 
class DicomDataset(Dataset):
    """
    Parses the Dicom dataset stored at DATA_DIR in .env

    Notes:
        1. skips labels that are {'roi': 'no'}
        2. skips stacks that don't have associated csv labels

    Nifti files gives the masks 
    Dicom files give the images
    CSV file give the labels for the dicom files

    NOTE: 
    - Both Nifit Scans/Masks are rotated by 90 degrees (counterclockwise) from Dicom
    - An affine-transformation must be applied between scan & mask
    """

    def __init__(self, root_dir_str: str, samples:list = None, max_samples = None):
        self.root_dir = Path(root_dir_str)

        # self.label_map = {
        #     '{"image_quality":"bad"}': 0,
        #     '{"image_quality":"good"}': 1
        # } # ignore the roi label
        # self.total_stacks = 0
        # self.unlabeled_stacks = []

        self.max_samples = max_samples

        self.use_transform = False
        self.transform = None
        self.default_transform = transforms.Compose(get_default_transform_list())

        self.cached_masks = {} # maps stack_path -> 3D Volume of Transformed Mask

        if not samples:
            samples = self._load_samples() # (fpath, label, person)
            
        self.samples = samples

           
    def _load_samples(self):
        samples = []
        # Dataset/ -> Person/ -> Stack/ -> Clean/ -> CSV, Niftis, & Dicoms/ -> Dicom Files 
        for person_path in tqdm(list(Path(self.root_dir).iterdir()), "Loading People Data: "):

            for stack_path in (person_path).iterdir():
                info_dir = stack_path / 'clean'

                # 1) Load Labels
                with open(info_dir / 'labels.json', 'r') as f:
                    label_map = json.load(f) # scan_num -> label

                # 2) Get Dicoms, Niftis, Masks (Width, Height, Scans)
                dicom_stack_path = info_dir / 'dicoms.npy'
                nifti_stack_path = info_dir / 'niftis.npy'
                mask_stack_path  = info_dir / 'masks.npy'
                
                for scan_num in label_map.keys():
                    samples.append({
                        "dicom_path": dicom_stack_path,
                        "nifti_path": nifti_stack_path,
                        "mask_path": mask_stack_path,
                        "scan_num": int(scan_num),
                        "label": label_map[scan_num],
                        "person": person_path.stem,
                    })
                    
                    # Break if reached max samples
                    if self.max_samples is not None and len(samples) >= self.max_samples:
                        return samples    
                # Compare Order
                # sorted_dicom_files = sorted(dicoms_dir_path.glob("*.dcm"), key = lambda s: s.name[:4], reverse=True)
                # first_dicom = pydicom.dcmread(sorted_dicom_files[0]).pixel_array.astype(dtype=np.float32)
                # first_nifti =  np.rot90(np.asarray(nifti_scan.dataobj[:, :, 0]), k = 1, axes = (0,1))

                # if not np.allclose(first_dicom, first_nifti): # don't match -> reverse order
                #     sorted_dicom_files.reverse()

                # 3) Parse Each Dicom for (fpath, label, person, (mask_path, scan_num))

                # for scan_num, fpath in enumerate(sorted_dicom_files):
                #     # Get the label
                #     row = label_df.loc[label_df["External ID"] == (fpath.stem + ".png")]
                #     label_str = row["Label"].values[0]

                #     # If unknown label -> skip
                #     if label_str not in self.label_map: 
                #         continue
                    
                #     # If no mask -> skip
                #     if not has_mask[scan_num]:
                #         continue

                #     # Add the sample
                #     samples.append((fpath, 
                #                     self.label_map[label_str], 
                #                     person_path.stem, 
                #                     (nifti_mask_path, nifti_scan_path, scan_num))
                #                     )


                    
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # dicom_stack, label, _, _ = self.samples[idx]

        # dicom = pydicom.dcmread(fpath)
        # img = dicom.pixel_array.astype(dtype=np.float32)

        # mask = self.get_mask(idx)
        
        img = self.get_img(idx)
        mask = self.get_mask(idx)
        label = self.samples[idx]['label']

        if np.sum(mask) > 0: # if there's a mask
            img = img * mask
        
        if self.use_transform:
            img = self.transform(img)
        else:
            img = self.default_transform(img)

        return img, label
    
    def get_img(self, idx, img_type='dicom'):
        """
        img_type is either nifti or dicom. They should produce the same thing...
        """

        scan_num = self.samples[idx]['scan_num']
        img_path = self.samples[idx][f'{img_type}_path']

        return np.load(img_path)[:, :, scan_num]
    
    def get_mask(self, idx):
        """
        Load the mask from file; note that we have to affine transform (resmaple) + rot90
        to align with the dicom & nifti
        """
        scan_num = self.samples[idx]['scan_num']
        mask_path = self.samples[idx]['mask_path']

        return np.load(mask_path)[:, :, scan_num]

        # _, _, _, (mask_path, scan_path, scan_num) = self.samples[idx]

        # if mask_path not in self.cached_masks: # cache the transformation
        #     nifti_mask = nib.load(mask_path)
        #     nifti_scan = nib.load(scan_path)
        #     self.cached_masks[mask_path] = resample_from_to(nifti_mask, nifti_scan, order = 0)

        # mask_resampled = self.cached_masks[mask_path]

        # mask2d = np.array(mask_resampled.dataobj[:, :, scan_num], dtype=np.float32)

        # return np.rot90(mask2d, k=1, axes=(0,1))

    
    # def get_niftislice(self, idx):
    #     _, _, _, (mask_path, scan_path, scan_num) = self.samples[idx]

    #     scan2d = nib.load(scan_path).dataobj[:, :, scan_num]

    #     return np.rot90(scan2d, k = 1, axes = (0, 1))

    def get_person_map(self) -> dict[str, list[int]]:
        """
        Returns a dictionary, which maps each person to the associated idxs into samples attribute
        """

        person_to_idxs = defaultdict(list)
        for idx, sample in enumerate(self.samples):
            person_to_idxs[sample['person']].append(idx)

        return person_to_idxs

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
        pos_samples = len([1 for sample in self.samples if sample['label'] == 1])
        result += f"Number of Pos Samples: {pos_samples}\n"
        result += f"Number of Neg Samples: {len(self) - pos_samples}\n"
        result += f"e.g. Max Value: {torch.max(self[0][0])}\n"

        print(result)

    def get_subset(self, indices):
        select_samples = [self.samples[i] for i in indices]
        subset = DicomDataset(self.root_dir, samples = select_samples)

        return subset

    def save_examples(self, dir_path: Path, num_examples: int = 3, num_augs: int = 5):
        dir_path.mkdir(parents=True, exist_ok=True)

        ncols = num_augs + 1  # base + augmentations
        nrows = num_examples

        _, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(3 * ncols, 2 + 3 * nrows)
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
        labels = [sample['label'] for sample in self.samples]  # extract labels
        class_counts = torch.bincount(torch.tensor(labels))
        class_weights = 1.0 / class_counts.float()   # inverse frequency
        sample_weights = class_weights[torch.tensor(labels)]

        return class_weights, sample_weights
    
    def test_data_collect(self, output_dir: Path):
        size = 5
        idxs = np.random.randint(0, len(self)-1, size = size)
        
        # dicoms = np.stack([pydicom.dcmread(self.samples[idx][0]).pixel_array.astype(dtype=np.float32) for idx in idxs], axis = -1)
        # masks = np.stack([self.get_mask(idx) for idx in idxs], axis = -1)
        # niftis = np.stack([self.get_niftislice(idx) for idx in idxs], axis = -1)

        dicoms = np.stack([self.get_img(idx, img_type = 'dicom') for idx in idxs], axis = -1)
        masks = np.stack([self.get_mask(idx) for idx in idxs], axis = -1)
        niftis = np.stack([self.get_img(idx, img_type = 'nifti') for idx in idxs], axis = -1)

        os.makedirs(output_dir / 'check_masks', exist_ok=True)
        save_image3d(niftis, output_dir / 'check_masks/niftis.png', mask = masks)
        save_image3d(dicoms, output_dir / 'check_masks/dicoms.png', mask = masks)
        save_image3d(masks, output_dir / 'check_masks/masks.png')

        assert np.allclose(dicoms, niftis)

def subject_split(dataset: DicomDataset, val_ratio:float=0.2):
    """
    Split dataset into train/val subsets by person.
    Ensures all images of a person are in the same subset.
    """
    # Group indices by person
    person_to_idxs = dataset.get_person_map()

    unique_people = list(person_to_idxs.keys())
    np.random.shuffle(unique_people) # shuffles the list of people for true random selection

    # Split people
    n_val = max(1, int(len(unique_people) * val_ratio))
    val_people = set(unique_people[:n_val])
    train_people = set(unique_people[n_val:])

    # Flatten indices
    train_indices = [idx for p in train_people for idx in person_to_idxs[p]]
    val_indices   = [idx for p in val_people   for idx in person_to_idxs[p]]

    train_dataset = dataset.get_subset(indices = train_indices)
    val_dataset   = dataset.get_subset(indices = val_indices)

    return train_dataset, val_dataset

def save_image3d(array, fpath: Path, mask: np.array = None):
    """
    Takes a 3D Volume and displays them all in a grid pattern;
    row-major order
    """
    # Save image
    H, W, Z = array.shape

    cols = math.ceil(math.sqrt(Z))
    rows = math.ceil(Z / cols)

    # Create figure
    _, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    # Handle if axes is 1D
    axes = np.array(axes).reshape(rows, cols)

    # Plot each slice
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < Z:
            ax.imshow(array[:, :, i], cmap='gray')
            if mask is not None:
                ax.imshow(mask[:, :, i], cmap='Blues', alpha=0.3)
            ax.set_title(f"z={i}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(fpath, dpi=300)
    plt.close()
        