from collections import defaultdict
from pathlib import Path
import os
from tqdm import tqdm

import json

import math
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, Sampler
import torchvision.transforms.functional as F
from torchvision import transforms
import torch

from augs_list import get_default_transform_list, get_color_transform_list, get_spatial_transform_list

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

    def __init__(self, root_dir_str: str, samples:list = None, max_samples = None, 
                 mask_method: str = ' '):
        
        self.root_dir = Path(root_dir_str)

        self.max_samples = max_samples

        self.use_transform = True
        self.transform = None
        self.default_transform = None

        self.mask_method = mask_method

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

                # 3) Check if it has masks
                with open(info_dir / 'has_mask.json') as f:
                    has_mask_map = json.load(f)
                
                for scan_num in label_map.keys():
                    # skip if we need a mask and its mask is empty
                    if self.mask_method == 'mask' and not has_mask_map[scan_num]:
                        continue

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
                    
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        
        img = self.get_img(idx)
        mask = self.get_mask(idx)
        label = self.samples[idx]['label']

        if self.mask_method == 'mask': # if there's a mask
            img = img * mask
        
        if self.mask_method == 'stack': # stack the image and mask!
            img = np.stack([img, mask], axis = -1)

        if self.transform is None and self.default_transform is None:
            raise ValueError("You can't get an item before setting the transforms")
        
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

    def get_person_map(self) -> dict[str, list[int]]:
        """
        Returns a dictionary, which maps each person to the associated idxs into samples attribute
        """

        person_to_idxs = defaultdict(list)
        for idx, sample in enumerate(self.samples):
            person_to_idxs[sample['person']].append(idx)

        return person_to_idxs

    def set_transforms(self, default_transform_list, transform_list):
        self.default_transform = transforms.Compose(default_transform_list) 
        self.transform = transforms.Compose(transform_list) 

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
        subset = DicomDataset(self.root_dir, samples = select_samples, 
                              mask_method = self.mask_method)

        return subset

    def save_examples(self, dir_path: Path, num_examples: int = 3, num_augs: int = 5):
        dir_path.mkdir(parents=True, exist_ok=True)

        ncols = num_augs + 1  # base + augmentations
        nrows = num_examples

        _, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(3 * ncols, 4 + 3 * nrows)
        )

        base_idxs = np.random.randint(0, len(self), size = num_examples)

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
                ax.imshow(F.to_pil_image(img[0, :, :]), cmap='grey')
                if row == 0:  # only add column titles on top row
                    ax.set_title(title)
                ax.axis("off")

            # Add y-label to show class (once per row)
            label_str = "GOOD" if base_label == 0 else "BAD"
            axes[row, 0].set_title(f"Idx ({base_idx}) = {label_str}")

        plt.suptitle(f"{dir_path.name}", fontsize=18, weight="bold")
        plt.tight_layout()
        plt.savefig(dir_path / "aug_examples.png")
        plt.close()

    def get_labels(self):
        return [sample['label'] for sample in self.samples]

    def get_class_weights(self):
        labels = self.get_labels() # extract labels
        class_counts = torch.bincount(torch.tensor(labels))
        class_weights = 1.0 / class_counts.float()   # inverse frequency
        sample_weights = class_weights[torch.tensor(labels)]

        return class_weights, sample_weights
    
    def test_data_collect(self, output_dir: Path):
        size = 5
        idxs = np.random.randint(0, len(self)-1, size = size)
        
        dicoms = np.stack([self.get_img(idx, img_type = 'dicom') for idx in idxs], axis = -1)
        masks = np.stack([self.get_mask(idx) for idx in idxs], axis = -1)
        niftis = np.stack([self.get_img(idx, img_type = 'nifti') for idx in idxs], axis = -1)

        os.makedirs(output_dir / 'check_masks', exist_ok=True)
        save_image3d(niftis, output_dir / 'check_masks/niftis.png', mask = masks)
        save_image3d(dicoms, output_dir / 'check_masks/dicoms.png', mask = masks)
        save_image3d(masks, output_dir / 'check_masks/masks.png')

        assert np.allclose(dicoms, niftis)

def split_and_augment(dataset: DicomDataset, train_cnt: int, val_cnt: int, test_cnt: int, 
                      aug_method:str = '', seed: None | int = None) -> tuple[DicomDataset, DicomDataset, DicomDataset]:
    """
    Split dataset into train/val subsets by person.
    Ensures all images of a person are in the same subset.
    """
    if seed is not None:
        np.random.seed(seed)

    # Group indices by person
    person_to_idxs = dataset.get_person_map()

    unique_people = list(person_to_idxs.keys())
    np.random.shuffle(unique_people) # shuffles the list of people for true random selection

    # Split people
    assert (train_cnt + val_cnt + test_cnt) <= len(unique_people), f"There are only {len(unique_people)}, but {train_cnt, val_cnt, test_cnt} requested"
    print(train_cnt, val_cnt, test_cnt)
    train_people = set(unique_people[:train_cnt])
    val_people = set(unique_people[train_cnt : train_cnt + val_cnt])
    test_people = set(unique_people[train_cnt+val_cnt : train_cnt + val_cnt + test_cnt])

    # Flatten indices
    train_indices = [idx for p in train_people  for idx in person_to_idxs[p]]
    val_indices   = [idx for p in val_people    for idx in person_to_idxs[p]]
    test_indices  = [idx for p in test_people   for idx in person_to_idxs[p]]

    # Get Subsets
    train_dataset = dataset.get_subset(indices = train_indices)
    val_dataset   = dataset.get_subset(indices = val_indices)
    test_dataset = dataset.get_subset(indices = test_indices)

    # Now Augment (Sets the .transform & .default_transform arguments appropriately for all 3 datasets)
    apply_augs(dataset, train_dataset, val_dataset, test_dataset, method = aug_method)

    return train_dataset, val_dataset, test_dataset

def apply_augs(dataset: DicomDataset, train_dataset: DicomDataset, val_dataset: DicomDataset, test_dataset: DicomDataset,
               method = '', perc = .02) -> None:
    """
    Applies a series of transformations

    1) Calculate mean/std from train_dataset & applies normalization
    2) Spatial augmentations to train if 's' in method
    3) Color   augmentations to train if 'c' in method
    4) Duplicates the img to 3D for the ResNet

    """
    basics = get_default_transform_list(perc=perc, mask_method=dataset.mask_method)
    spatial_transform = get_spatial_transform_list()
    color_transform = get_color_transform_list(mask_method=dataset.mask_method)

    augmentations = []
    if 's' in method:
        print("Applying Spatital Augmentations")
        augmentations.extend(spatial_transform)
    if 'c' in method:
        print("Applying Color Augmentations")
        augmentations.extend(color_transform)

    train_transform = basics[:-1] + augmentations + basics[-1:]
    val_transform = basics
    test_transform = basics

    # Set transforms (default, actual)
    dataset.set_transforms(basics, basics)
    train_dataset.set_transforms(basics, train_transform)
    val_dataset.set_transforms(basics, val_transform)
    test_dataset.set_transforms(basics, test_transform)

class BalancedBatchSampler(Sampler):
    """
    Sampler that ensures each batch has exactly 50% positive and 50% negative samples.
    Yields individual indices; DataLoader handles batching.
    """

    def __init__(self, labels, batch_size):
        assert batch_size % 2 == 0, "Batch size must be even."

        self.labels = np.array(labels)
        self.pos_indices = np.where(self.labels == 1)[0]
        self.neg_indices = np.where(self.labels == 0)[0]

        self.batch_size = batch_size
        self.half = batch_size // 2

        # Number of batches we can make with full 50/50 balance
        self.num_batches = min(len(self.pos_indices), len(self.neg_indices)) // self.half

    def __iter__(self):
        # Shuffle the indices each epoch
        pos_perm = np.random.permutation(self.pos_indices)
        neg_perm = np.random.permutation(self.neg_indices)

        for i in range(self.num_batches):
            pos_batch = pos_perm[i*self.half : (i+1)*self.half]
            neg_batch = neg_perm[i*self.half : (i+1)*self.half]
            batch = np.concatenate([pos_batch, neg_batch])
            np.random.shuffle(batch)
            # Yield indices one by one for DataLoader
            for idx in batch:
                yield int(idx)

    def __len__(self):
        return self.num_batches * self.batch_size

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
        