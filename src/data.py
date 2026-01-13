from collections import defaultdict
from pathlib import Path
import os
from tqdm import tqdm

import json

import math
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, Sampler
from torchvision import transforms
import torch

from brain_transforms import CustomNormalize, CustomTransform

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

class SyntheticDataset(Dataset):
    """
    Synthetic dataset generated via augmentations/corruptions of scans.
    - Initialized by referencing the directory where they are located
    - Directory has a list of img_num.png and mask_num.png
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass


class DicomDataset(Dataset):
    """
    Parses the Dicom dataset stored at DATA_DIR in .env

    Nifti files gives the masks 
    Dicom files give the images
    CSV file give the labels for the dicom files

    Parameters:
     1. samples     = pre-specified list of samples
     2. max_samples = upper bound on # of samples
     3. mask_method = 'mask' or 'stack' or None
     4. norm_method = 'min-max' or 'squash-peak' or None
     5. masked_norm = True / False (use mask distribution for normalization)
    """

    def __init__(self, root_dir_str: str, samples:list = None, max_samples = None, 
                 mask_method: str = ' ', norm_method: str | None = None, 
                 masked_norm: bool = False, perc_norm: float = 0.2, check_bounds: bool = False):
        
        self.root_dir = Path(root_dir_str)
        self.max_samples = max_samples

        self.augmentations: list[CustomTransform] | None = None

        self.require_mask = False   # True: dataset only includes scans w/ masks
        self.norm_skip_last = False # True: skip the last channel when normalizing

        self.set_norm(mask_method=mask_method, norm_method=norm_method, masked_norm=masked_norm, perc_norm=perc_norm, check_bounds = check_bounds)
        
        if not samples:
            samples = self._load_samples() # (fpath, label, person)
        self.samples = samples

        self.additional_datasets: list[tuple[SyntheticDataset, float]] = [] # additional dataseights that we add on for sampling w/ their weights

        # Get masked & unmasked idxs
        self.masked_idxs = [] # idicies that have masks
        self.unmasked_idxs = [] # indicies that dont have masks
        for i, sample in enumerate(self.samples):
            if sample['has_mask']:
                self.masked_idxs.append(i)
            else:
                self.unmasked_idxs.append(i)
           
    def _load_samples(self):
        samples = []
        # Dataset/ -> Person/ -> Stack/ -> Clean/ -> CSV, Niftis, & Dicoms/ -> Dicom Files 
        for person_path in tqdm(list(Path(self.root_dir).iterdir()), "Loading People Data: "):
            
            if not person_path.is_dir():
                    continue 

            for stack_path in (person_path).iterdir():
                info_dir = stack_path / 'clean'

                if not info_dir.is_dir():
                    continue 

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
                    samples.append({
                        "dicom_path": dicom_stack_path,
                        "nifti_path": nifti_stack_path,
                        "mask_path": mask_stack_path,
                        "scan_num": int(scan_num),
                        "label": int(label_map[scan_num]),
                        "person": person_path.stem,
                        "has_mask": bool(has_mask_map[scan_num]),
                    })     
                    # Break if reached max samples
                    if self.max_samples is not None and len(samples) >= self.max_samples:
                        return samples    
                    
        return samples
    
    def add_customized_data(self, syn_dataset: SyntheticDataset, weight: float):
        """
        Adds a customized dataset (one that we curated via augmentations); it will be
        sampled from according to its relative weight to the other datasets   

        (note that the original dataset has weight 1)
        """

        self.additional_datasets.append((syn_dataset, weight))

    def __len__(self):
        """
        Length of dataset 
        - takes into account whether we are only using masked images or not
        """
        if self.require_mask:
            return len(self.masked_idxs)
        else:
            return len(self.samples)
        
    def _get_sample_idx(self, external_idx: int):
        """
        external_idx runs between 0 and len(dataset)

        it is turned into a sample_idx that accesses the sample array
        """
        if self.require_mask:
            assert external_idx < len(self.masked_idxs), "Trying to idx into unmasked territory while requiring masks"

        if external_idx < len(self.masked_idxs):
            sample_idx = self.masked_idxs[external_idx]
        elif external_idx < len(self.samples):
            sample_idx = self.unmasked_idxs[external_idx - len(self.masked_idxs)]

        return sample_idx
    
    def _get_sample(self, idx) -> dict:
        """
        Internal Method used to access the samples array. 

        The dataset is organized by ---- masked_idxs ----- unmasked_idxs ----,
        so we sample from the first 
        """
        sample_idx = self._get_sample_idx(idx)

        return self.samples[sample_idx]
    
    def _get_samples(self) -> list[dict]:
        """
        Returns list of all samples, taking into account maskign
        """

        return [self._get_sample(idx) for idx in range(len(self))] 
        

    def __getitem__(self, idx):
        
        img = torch.tensor(self.get_img(idx))                 # (W, H)
        mask = torch.tensor(self.get_mask(idx), dtype=bool)   # (W, H)
        label = self._get_sample(idx)['label']


        # Apply Mask Method [mask placed at the end if stacked]
        if self.mask_method == 'mask': # apply the mask!
            img = img * mask
        elif self.mask_method == 'stack': # stack the image and mask!
            img = torch.stack([img, mask], dim = 0) # (2, W, H)
        elif self.mask_method == 'stack2':
            img = torch.stack([img, img, mask], dim = 0)

        # Ensure image is (2 or 3, W, H)
        if img.ndim == 2:
            img = img.unsqueeze(0)         # (1, H, W)
            img = img.repeat(3, 1, 1)      # (3, H, W)
        
        # Resize Image
        img = transforms.Resize((244, 244))(img)

        # Apply Normalization
        normalizer = CustomNormalize(perc = self.perc_norm, method = self.norm_method, skip_last = self.norm_skip_last)
        if self.masked_norm:
            mask = transforms.Resize((244, 244), interpolation=transforms.InterpolationMode.NEAREST)(mask.unsqueeze(0))
            img = normalizer(img, mask.squeeze(0))
        else:
            img = normalizer(img)

        # Apply Spatial Augmentations & Check for out of bounds

        if self.augmentations is not None:
            for transform in self.augmentations:
                old_mask = img[-1]
                img = transform(img)
                if self.check_bounds and transform.mask_moves_outside(old_mask):
                    # NOTE: this dynamic relabeling corrupts balanced batch sampling b/c get_labels() isn't 100% correct
                    # We assume this shouldn't be a problem on average, though, since these events are rare
                    label = 1 # declare bad

        return img, label
    
    def get_img(self, idx, img_type='dicom'):
        """
        img_type is either nifti or dicom. They should produce the same thing...
        """

        scan_num = self._get_sample(idx)['scan_num']
        img_path = self._get_sample(idx)[f'{img_type}_path']

        return np.load(img_path)[:, :, scan_num]
    
    def get_mask(self, idx) -> torch.Tensor:
        """
        Load the mask from file; note that we have to affine transform (resmaple) + rot90
        to align with the dicom & nifti
        """
        scan_num = self._get_sample(idx)['scan_num']
        mask_path = self._get_sample(idx)['mask_path']

        return np.load(mask_path)[:, :, scan_num]

    def get_person_map(self) -> dict[str, list[int]]:
        """
        Returns a dictionary, which maps each person to the associated idxs into samples attribute
        """

        person_to_idxs = defaultdict(list)
        for idx in range(len(self)):
            sample = self._get_sample(idx)
            person_to_idxs[sample['person']].append(idx)

        return person_to_idxs

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
        pos_samples = len([1 for sample in self._get_samples() if sample['label'] == 1])
        result += f"Number of Pos Samples: {pos_samples}\n"
        result += f"Number of Neg Samples: {len(self) - pos_samples}\n"
        result += f"e.g. Max Value: {torch.max(self[0][0])}\n"

        print(result)

        return {"pos": pos_samples, "neg": len(self) - pos_samples}

    def get_subset(self, indices):
        # Select the sample dicts
        select_samples = [self._get_sample(i) for i in indices]

        # Create new dataset but DO NOT let __init__ load samples from disk
        subset = DicomDataset(
            self.root_dir,
            samples=select_samples,
            mask_method=self.mask_method,
            norm_method=self.norm_method,
            masked_norm=self.masked_norm,
            perc_norm=self.perc_norm
        )

        return subset

    def get_labels(self):
        return [sample['label'] for sample in self._get_samples()]

    def get_class_weights(self):
        labels = self.get_labels() # extract labels
        class_counts = torch.bincount(torch.tensor(labels))
        class_weights = 1.0 / class_counts.float()   # inverse frequency
        sample_weights = class_weights[torch.tensor(labels)]

        return class_weights, sample_weights
    
    def test_data_collect(self, output_dir: Path, idxs: list[int] | None = None):
        if idxs is None:
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

    def set_aug(self, augmenation_list: list[CustomTransform]):
        if augmenation_list is None:
            self.augmentations = None
        else:
            self.augmentations = augmenation_list

    def set_norm(self, 
                 mask_method: str   | None = None, 
                 norm_method: str   | None = None, 
                 masked_norm: bool  | None = None,
                 perc_norm:   float | None = None,
                 check_bounds: bool | None = None):
        
        self.mask_method = mask_method
        self.norm_method = norm_method
        self.masked_norm = masked_norm
        self.perc_norm = perc_norm
        self.check_bounds = check_bounds

        self.require_mask = self.mask_method in ('stack', 'stack2', 'mask') or self.masked_norm or self.check_bounds
        self.norm_skip_last = self.mask_method in ('stack','stack2')

    def get_scans_without_mask(self) -> set[int]:
        return [idx for idx in range(len(self)) if not self._get_sample(idx)['has_mask']]

    def get_idxs_of_stack(self, target_idx: int) -> tuple[list[int], Path]:
        """
        Returns the idxs of other samples in the same stack; and the stack path

        (same stack share the same mask_path)
        """

        mask_path = self._get_sample(target_idx)['mask_path']
        all_idxs = []

        for idx in range(len(self)):
            sample_mask_path = self._get_sample(idx)['mask_path']
            if sample_mask_path == mask_path:
                all_idxs.append(idx)
        
        return all_idxs, mask_path.parent


# ----------------------- SPLITTING THE DATA ------------------------------------------------------------------------------------

def get_people_groups(num_train: int, num_val: int, num_test: int, 
                 n_rounds: int, use_k_fold: bool = False,
                 seed: int = None):
    """
    Returns a list of lists, such that:
    [ [ [train_people], [val_people], [test_people] ] for each round]

    If k_fold is specified, total_people = num_train + num_val + num_test. And dividing occurs as follows:
    We have n_rounds rounds & groups, each with num_people / folds people. Must be divisible.
    One group is test; one is val; rest are train. 
    """

    if seed > 0:
        np.random.seed(seed)

    num_people = num_train + num_val + num_test
    all_people = list(range(num_people))   
    num_folds = n_rounds

    if not use_k_fold:
        result = []
        for _ in range(n_rounds):
            np.random.shuffle(all_people)
            result.append(
                [
                    all_people[                     : num_train],
                    all_people[num_train            : num_train + num_val], 
                    all_people[num_train + num_val  : num_train + num_val + num_test]
                ]
            )
        return result
    else:
        assert num_people % num_folds == 0, f"k_fold {num_folds} does not divide num_people {num_people}"
        group_size = num_people // num_folds

        np.random.shuffle(all_people)
        groups = [all_people[i*group_size: (i+1)*group_size] for i in range(num_folds)]

        result = []
        for i in range(num_folds):
            test_idx = i
            val_idx = (i + 1) % num_folds
            train_idxs = set(range(len(groups))) - {test_idx, val_idx}

            result.append(
                [
                    [x for j in train_idxs for x in groups[j]],
                    groups[val_idx],
                    groups[test_idx]
                ]
            )
        return result

def split_dataset(dataset: DicomDataset, people: list):
    """
    Splits a dataset according to the people list [[train_people], [val_people], [test_people]]

    Returns train_dataset, val_dataset, test_dataset   
    """    

    person_id_to_idxs = dataset.get_person_map()           # person_id -> idxs into dataset
    person_ids = sorted(list(person_id_to_idxs.keys())) # [person_ids in order]

    train_people, val_people, test_people = people

    train_indices = [idx for p in train_people  for idx in person_id_to_idxs[person_ids[p]]]
    val_indices   = [idx for p in val_people    for idx in person_id_to_idxs[person_ids[p]]]
    test_indices  = [idx for p in test_people   for idx in person_id_to_idxs[person_ids[p]]]

    train_dataset = dataset.get_subset(indices = train_indices)
    val_dataset   = dataset.get_subset(indices = val_indices)
    test_dataset = dataset.get_subset(indices = test_indices)

    return train_dataset, val_dataset, test_dataset

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
        

if __name__ == '__main__':
    splits = get_people_groups(20, 5, 5, 6, use_k_fold = False, seed = 10)
    print(splits)
    splits2 = get_people_groups(20, 5, 5, 6, use_k_fold = True, seed = 10)
    print()
    print(splits2)