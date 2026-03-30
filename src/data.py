from pathlib import Path
from tqdm import tqdm
from itertools import product
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, Sampler

import nibabel as nib
from brain_transforms import CustomTransform

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


class VisualParams:
    """
    Organizes all of the visual parameters for the DicomDataset class

    display_method:  None / "mask" / "stack2" / "stack3"
    norm_method:     None / "min-max" / "peak-squash" 
    masked_norm:     True / False
    percentile_norm:  float
    """

    def __init__(self, display_method: str | None = None, norm_method: str | None = None, 
                 masked_norm: bool = False, percentile_norm: float = 0.0):
        
        self.set_values(display_method=display_method, norm_method=norm_method, masked_norm=masked_norm, percentile_norm=percentile_norm)
        

    def set_values(self, display_method: str | None = None, norm_method: str | None = None, 
                 masked_norm: bool = False, percentile_norm: float = 0.0):
        
        self.display_method = display_method
        self.norm_method = norm_method
        self.masked_norm = masked_norm
        self.percentile_norm = percentile_norm

        self.norm_skip_last = display_method in ('stack2','stack3')

    def preprocess_scan_and_mask(self, scan: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Modifies a scan according to the visualize parameters
        """
        assert scan.ndim == 2 and mask.ndim == 2 and scan.shape == mask.shape

        # Display Method
        if self.display_method == 'mask': # scan * mask -> (3, W, H)
            scan = (scan * mask).unsqueeze(0).repeat(3, 1, 1)
        elif self.display_method == 'stack2': # [scan, mask] -> (2, W, H)
            scan = torch.stack([scan, mask], dim = 0) 
        elif self.display_method == 'stack3':
            scan = torch.stack([scan, scan, mask], dim = 0) # [scan, scan, mask] -> (3, W, H)
        else:
            scan = scan.unsqueeze(0).repeat(3, 1, 1) # (3, W, H)

        # Resize Image
        scan = transforms.Resize((244, 244))(scan.float())
        mask = transforms.Resize((244, 244), interpolation=InterpolationMode.NEAREST)(mask.unsqueeze(0).float()).squeeze(0).bool()

        # Apply (possibly masked) Normalization
        if not self.masked_norm:
            mask = None
            
        if self.norm_method == 'min-max':
            scan = self._minmax_normalize(scan, mask)
        elif self.norm_method == 'peak-squash': 
            scan = self._peaksquash_normalize(scan, mask)

        return scan, mask

    def _minmax_normalize(self, scan: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        img: (C, H, W)
        mask: (H, W) boolean
        """

        if mask is not None and mask.sum() > 0:
            assert mask.dtype == torch.bool
            assert scan.ndim == 3 and mask.shape == scan.shape[1:]
            nz_values = scan[0][mask]               # 1D
        else:
            nz_values = scan[scan > 0]            # avoid all the zeros

        scan_min = torch.quantile(nz_values, self.percentile_norm)
        scan_max = torch.quantile(nz_values, 1 - self.percentile_norm)

        new_scan = (scan - scan_min) / (scan_max - scan_min + 1e-6)  # scale to 0-1
        new_scan = torch.clip(new_scan, 0, 1)

        if self.norm_skip_last: # Don't normalize last channel
            return torch.cat([new_scan[:-1], scan[-1:]], dim = 0)
        else:
            return new_scan

    def _peaksquash_normalize(self, scan: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Sends most freq -> 1/2 and 0 -> 0; linearly scales the rest.
        img: (C, H, W)
        mask: (H, W) boolean
        """
        # extract non-zero (or masked) values for estimating peak
        if mask is not None:
            assert mask.dtype == torch.bool
            assert scan.ndim == 3 and mask.shape == scan.shape[1:]
            nz_values = scan[0][mask]              
        else:
            nz_values = scan[scan > 0]

        # compute histogram peak
        counts, bin_edges = torch.histogram(nz_values, bins=200)
        peak_bin_index = torch.argmax(counts)
        x_peak = (bin_edges[peak_bin_index] + bin_edges[peak_bin_index + 1]) / 2

        if self.norm_skip_last: # Don't normalize the final channel
            return torch.cat([scan[0:-1] / (2 * x_peak), scan[-1:]], dim = 0)
        else:
            return scan / (2 * x_peak)
        
def split_people(all_ids: list[int], fractions: list[int], seed: int = 42, num_runs: int = 1) -> list[dict[str, list[int]]]:
    """
    Splits up a list of ids according to some fractions. 

    Returns a list of length num_runs, where each element is a dict mapping split_type -> person_ids split 
    randomly according to fractions
    """
    assert np.isclose(np.sum(fractions), 1)

    np.random.seed(seed)

    n = len(all_ids)
    num_train = int(fractions[0] * n)
    num_val = int(fractions[1] * n)

    result = []

    for _ in range(num_runs):
        np.random.shuffle(all_ids)
        result.append(
            {
                'train': all_ids[                     : num_train],
                'val':   all_ids[num_train            : num_train + num_val], 
                'test':  all_ids[num_train + num_val  : ]
            }
        )

    return result


def get_sample_dataframe(data_csv: Path, dataset_types: list[str], limit_to_one_stack: bool = False) -> tuple[pd.DataFrame, list[int]]:
    """
    Returns both the filter dataframe from the file & a list of all people within the df
    """
    full_df = pd.read_csv(data_csv, index_col = 0)
    print(f"Full Dataset has {len(full_df)} samples")

     # samples should only have labels 0/1
    samples_df = full_df.loc[
        (full_df['label'] == 0) | (full_df['label'] == 1)
    ]

    # filter out according to dataset_types if you want
    if dataset_types:
        samples_df = samples_df[
            samples_df['dataset'].isin(dataset_types)
        ]

    # drop duplicates so that each stack is only used once
    print(len(samples_df))
    samples_df = samples_df.drop_duplicates(subset=['path', 'scan_num'])
    print(len(samples_df))

    # drop duplicates so that each person only has one associated stack
    if limit_to_one_stack:
        print(samples_df.groupby('person_id')['path'].nunique())
        samples_df = samples_df[
            samples_df['path'].isin(
                samples_df.groupby('person_id')['path'].first()
            )
        ]

        assert (samples_df.groupby('person_id')['path'].nunique() == 1).all()

    all_people = samples_df['person_id'].unique().tolist()
    return samples_df, all_people

class DicomDataset(Dataset):
    """
    Assumes the existence of pd.DataFrame file which contains all of the relevant information
    about the many samples
    """

    def __init__(self, samples_df: pd.DataFrame, vis_params: VisualParams | None = None, 
                 person_ids: list[int] | None = None, summarize_name: str | None = None, drop_edges: bool = True):
        
        self.samples_df = samples_df
        
        # filter out according to person_ids if you want
        if person_ids:
            self.samples_df = self.samples_df[
                self.samples_df['person_id'].isin(person_ids)
            ].copy()

        # filter out edges if don't want them
        if drop_edges:
            min_labeled = self.samples_df['labeled_scans'].apply(ast.literal_eval).apply(min)
            max_labeled = self.samples_df['labeled_scans'].apply(ast.literal_eval).apply(max)
            self.samples_df['progress'] = (self.samples_df['scan_num'] - min_labeled) / (max_labeled - min_labeled)

            bins = np.linspace(0, 1, 11)
            self.samples_df["bin"] = pd.cut(self.samples_df['progress'], bins = bins, include_lowest=True)
            self.samples_df = self.samples_df[
                (self.samples_df['bin'] != self.samples_df['bin'].cat.categories[0]) & (self.samples_df['bin'] != self.samples_df['bin'].cat.categories[-1]) 
            ].copy()

        # additional display params
        self.vis_params = vis_params
        self.augmentations: list[CustomTransform] = []

        # self.additional_datasets: list[tuple[SyntheticDataset, float]] = [] # additional dataseights that we add on for sampling w/ their weights

        if summarize_name is not None:
            self.summarize(summarize_name)

    # 1. Main Functionality
    # ---------------------

    def __len__(self):
        return len(self.samples_df)
    
    def read_scan(self, idx) -> torch.Tensor:
        """
        Returns np array corresponding to the scan for idx in the samples DataFrame 
        """

        scan_num = self.samples_df.iloc[idx]['scan_num']
        scan_path = Path(self.samples_df.iloc[idx]['path'])

        if scan_path.suffix == '.npy':
            nifti_data = np.load(scan_path)
        else:
            nifti_img = nib.load(scan_path)
            nifti_data = nifti_img.get_fdata().astype(np.float32)

        return torch.from_numpy(nifti_data)[:, :, scan_num]

    def read_mask(self, idx) -> torch.Tensor:
            
        """
        Returns np array corresponding to the mask for idx in the samples DataFrame 
        """
        scan_num = self.samples_df.iloc[idx]['scan_num']
        mask_path = Path(self.samples_df.iloc[idx]['mask_path'])
        
        if mask_path.suffix == '.npy':
            nifti_data = np.load(mask_path)
        else:
            nifti_img = nib.load(mask_path)
            nifti_data = nifti_img.get_fdata().astype(np.bool_)

        return torch.from_numpy(nifti_data)[:, :, scan_num]

    def __getitem__(self, idx):
        scan = self.read_scan(idx)           
        mask = self.read_mask(idx)

        label = self.samples_df.iloc[idx]['label']

        # Apply Normalization
        normalized_scan, normalized_mask = self.vis_params.preprocess_scan_and_mask(scan, mask)

        # Apply Spatial Augmentations
        if self.augmentations is not None:
            for transform in self.augmentations:
                normalized_scan = transform(normalized_scan)

        return normalized_scan, normalized_mask, label, idx
    
    # 2. Extra Functions
    # ---------------------
    def set_vis_params(self, vp: VisualParams):
        self.vis_params = vp

    def set_aug(self, augmentation_list: list[CustomTransform]):
        self.augmentations = augmentation_list 

    def summarize(self, name: str = 'none'):
        """
        Prints relevant information about the DicomDataset
        """
        print(name)
        print('----------------')
        print(self.get_counts())
        print('----------------')
        print()

    def get_counts(self):
        return {
            'pos': int((self.samples_df['label'] == 1).sum()),
            'neg': int((self.samples_df['label'] == 0).sum()),
            'total': len(self.samples_df)
        }
    
    def get_labels(self):
        return self.samples_df['label'].to_numpy(dtype=int)
    
    def get_weights(self):
        """
        Returns two things:
        1. class_weights = Inverse frequency of the labels
        2. per_sample_weights = The weight applied for each sample
        """

        labels = self.get_labels()

        class_counts = torch.bincount(torch.tensor(labels))
        class_weights = 1.0 / class_counts.float()
        per_sample_weights = class_weights[torch.tensor(labels, dtype = int)]

        return class_weights, per_sample_weights

    def show(self, idx: int, path: Path, with_mask: bool = False, display_params: bool = False):
        img, mask = self[idx][0][0, :, :], self[idx][1]

        plt.imshow(img, cmap="gray", vmin=0, vmax=1)

        if with_mask and mask is not None:
            plt.imshow(mask, cmap="Reds", alpha=0.4)

        if display_params:
            plt.text(0, 0, str(display_params))

        plt.axis("off")
        plt.savefig(path)
        plt.close()

    def get_extra_info(self, idxs: list[int], info: list[str] = ['pdf_num', 'stack_num', 'path', 'scan_num']):
        subset = self.samples_df.iloc[idxs]

        return {
            col: subset[col].tolist()
            for col in info
        }


if __name__ == '__main__':
    print("Testing Data Module. Look at outputs_test_dataset directory for images")

    num_indices_to_display = 2

    output_dir = Path('/data/vision/polina/users/marcusbl/bin_class/outputs_test_dataset')
    output_dir.mkdir(exist_ok=True)

    for item in output_dir.iterdir():
        item.unlink()

    # Create Dataset
    dataset_path = Path('/data/vision/polina/users/marcusbl/bin_class/all_data/samples.csv')
    data_samples_df, person_ids = get_sample_dataframe(dataset_path, dataset_types = ['BCH', 'R'])

    dataset = DicomDataset(data_samples_df)
    dataset.summarize()

    groups = split_people(data_samples_df['person_id'].unique().tolist(), fractions = [0.25, 0.25, 0.5], seed = 42, num_runs = 3)
    assert len(groups) == 3

    # Create all possible parameter combinations
    # display_methods = [None, 'mask', 'stack2', 'stack3']
    # norm_methods = [None, 'min-max', 'peak-squash']
    # masked_norm = [True, False]
    # perc = np.linspace(0, 1, 5)

    # all_params = list(product(display_methods, norm_methods, masked_norm, perc))

    # # Choose random idxs & test dataset on each of these
    # idxs = np.random.randint(low=0, high = len(dataset), size = num_indices_to_display)

    # for i, params in tqdm(list(enumerate(all_params)), "Testing all params"):
    #     # print(params)
    #     vp = VisualParams(*params)
    #     dataset.set_vis_params(vp)

    #     for idx in idxs:
    #         dataset.show(idx, output_dir / f'{i}:{idx}.png', with_mask = True, display_params=True)


    

        





