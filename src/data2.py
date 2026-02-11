from collections import defaultdict
from pathlib import Path
import os
from tqdm import tqdm
import json
from itertools import product


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nibabel as nib
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
import torch
from torchvision.transforms import InterpolationMode


from brain_transforms import CustomTransform

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

    def preprocess_scan(self, scan: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Modifies a scan according to the visualize parameters
        """
        assert scan.ndim == 2 and mask.ndim == 2 and scan.shape == mask.shape

        # Display Method
        if self.display_method == 'mask': # scan * mask -> (1, W, H)
            scan = (scan * mask).unsqueeze(0)
        elif self.display_method == 'stack2': # [scan, mask] -> (2, W, H)
            scan = torch.stack([scan, mask], dim = 0) 
        elif self.display_method == 'stack3':
            scan = torch.stack([scan, scan, mask], dim = 0) # [scan, scan, mask] -> (3, W, H)
        else:
            scan = scan.unsqueeze(0) # (1, W, H)

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

        return scan

    def _minmax_normalize(self, scan: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        img: (C, H, W)
        mask: (H, W) boolean
        """

        if mask is not None:
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


class DicomDataset(Dataset):
    """
    Assumes the existence of pd.DataFrame file which contains all of the relevant information
    about the many samples
    """

    def __init__(self, data_csv: Path, vis_params: VisualParams | None = None):
        
        # full df in all of its glory
        self.full_df = pd.read_csv(data_csv, index_col = 0)

        # samples should only have labels 0/1
        self.samples_df = self.full_df.loc[
            (self.full_df['label'] == 0) | (self.full_df['label'] == 1)
        ].copy()


        # additional display params
        self.vis_params = vis_params
        self.augmentations: list[CustomTransform] = []
        # self.additional_datasets: list[tuple[SyntheticDataset, float]] = [] # additional dataseights that we add on for sampling w/ their weights

    # 1. Main Functionality
    # ---------------------

    def __len__(self):
        return len(self.samples_df)
    
    def get_scan(self, idx) -> torch.Tensor:
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

    def get_mask(self, idx) -> torch.Tensor:
            
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
        scan = self.get_scan(idx)           
        mask = self.get_mask(idx)

        label = self.samples_df.iloc[idx]['label']

        # Apply Normalization
        normalized_scan = self.vis_params.preprocess_scan(scan, mask)

        # Apply Spatial Augmentations
        if self.augmentations is not None:
            for transform in self.augmentations:
                normalized_scan = transform(normalized_scan)

        return normalized_scan, label
    
    # 2. Extra Functions
    # ---------------------

    def set_vis_params(self, vp: VisualParams):
        self.vis_params = vp

    def summarize(self):
        """
        Prints relevant information about the DicomDataset
        """
        print(f"There are a total of {len(self.full_df)} in full DataFrame")
        print(f"There are a total of {len(self)} in samples DataFrame")
        print(f"There are {(self.samples_df['label'] == 0).sum()} good & {(self.samples_df['label'] == 1).sum()} bad samples")

    def show(self, idx: int, path: Path, with_mask: bool = False, display_params: bool = False):
        img = self[idx][0][0, :, :]
        mask = self.get_mask(idx)

        plt.imshow(img, cmap="gray", vmin=0, vmax=1)

        if with_mask:
            plt.imshow(
                mask,
                cmap="Reds",      # red overlay
                alpha=0.4         # transparency
            )

        if display_params:
            plt.text(0, 10, str(params))

        plt.axis("off")
        plt.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close()

if __name__ == '__main__':
    print("Testing Data Module. Look at outputs_test_dataset directory for images")

    num_indices_to_display = 5

    output_dir = Path('/data/vision/polina/users/marcusbl/bin_class/outputs_test_dataset')
    output_dir.mkdir(exist_ok=True)

    for item in output_dir.iterdir():
        item.unlink()

    # Create Dataset
    dataset_path = Path('/data/vision/polina/users/marcusbl/all_data/samples.csv')
    dataset = DicomDataset(dataset_path)
    dataset.summarize()

    # Create all possible parameter combinations
    display_methods = [None, 'mask', 'stack2', 'stack3']
    norm_methods = [None, 'min-max', 'peak-squash']
    masked_norm = [True, False]
    perc = np.linspace(0, 1, 5)

    all_params = list(product(display_methods, norm_methods, masked_norm, perc))

    # Choose random idxs & test dataset on each of these
    idxs = np.random.randint(low=0, high = len(dataset), size = num_indices_to_display)

    for i, params in enumerate(all_params):
        print(params)
        vp = VisualParams(*params)
        dataset.set_vis_params(vp)

        for idx in idxs:
            dataset.show(idx, output_dir / f'{i}:{idx}.png', with_mask = True, display_params=True)


        

        





