"""
This should be called AFTER you run Jupyter Notebook: pre_labeling_session.ipynb

It must be run as .py instead of .ipynb b/c it crashes Kernel otherwise

"""

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import json

from tqdm import tqdm
import pandas as pd
import nibabel as nib
from matplotlib.backends.backend_pdf import PdfPages

def minmax(img: np.ndarray, mask: np.ndarray, perc: float = 0.01):
    """
    Percentile-based min–max normalization using a mask.

    img:  (H, W) or (C, H, W)
    mask: (H, W) boolean or 0/1
    perc: lower/upper percentile to clip (e.g., 0.01 → 1%–99%)
    """

    values = img[mask]

    if values.size == 0:
        print('here')
        return np.zeros_like(img)

    img_min = np.quantile(values, perc)
    img_max = np.quantile(values, 1.0 - perc)

    if img_max <= img_min:
        return np.zeros_like(img)

    new_img = (img - img_min) / (img_max - img_min)
    return np.clip(new_img, 0, 1)


def display_slice(stack_data: np.ndarray, mask_data: np.ndarray, slice_num: int, axis):
    """
    Given: 3D stack and 3D Mask; slice_num; axis to display onto

    Displays the slice onto the axis
    """
    mask = mask_data[:, :, slice_num].astype(bool)
    slice = stack_data[:, :, slice_num]

    if mask.sum() > 30:
        img = minmax(slice, mask, perc = .01)
        axis.imshow(img, cmap='grey', vmin=0,vmax = 1)
        # axis.imshow(mask, cmap="Reds", alpha=0.4)
    else:
        axis.imshow(slice, cmap ='grey')
        # axis.imshow(mask, cmap="Reds", alpha=0.4)
        axis.text(
            0.98, 0.98, "⚠",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=14,
            color="red",
            weight="bold"
        )
    axis.text(
        0.02, 0.98, f'{slice_num}; Area = {int(mask.sum())}',
        transform=axis.transAxes,
        ha = "left",
        va = "top", 
        fontsize = 14,
        color='yellow',
        weight='bold'
    )

    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_frame_on(False)


def display_stack(stack_path: Path, mask_path: Path):
    """    
    Given: stack_path & mask_path
    Returns: fig object for the stack
    """

    # PARSE THE STACK
    stack_path = Path(stack_path)
    mask_path = Path(mask_path)

    # Download nifti data and mask data
    if stack_path.suffix == '.npy':
        nifti_data = np.load(stack_path)
    else:
        nifti_img = nib.load(stack_path)
        nifti_data = nifti_img.get_fdata()

    if mask_path.suffix == '.npy':
        mask_data = np.load(mask_path).astype(bool)
    else:
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata().astype(bool)

    # Calculate the # of scans [ignore unmasked beginning and end]
    start, end = 100, 0
    for slice_num in range(nifti_data.shape[-1]):
        mask = mask_data[:, :, slice_num] 

        if mask.sum() > 0:
            start = min(start, slice_num)
            end = max(end, slice_num)

    kept_slices_cnt = end - start + 1
    kept_slices = list(range(start, end + 1)) # [start, end] inclusive 

    if kept_slices_cnt == 0:
        print(f"No masks found for {stack_path}")
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.set_title("NO DATA")
        return fig, [], 0

    # Create the Grid
    nrows = int(np.floor(np.sqrt(kept_slices_cnt)))
    ncols = int(np.ceil(kept_slices_cnt / nrows))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(2 * ncols, 2 * nrows)
    )
    axes = np.atleast_2d(axes)

    # Loop through stack & display
    for i in range(nrows):
        for j in range(ncols):
            cnt = i * ncols + j

            # stop if at the end
            if cnt >= kept_slices_cnt:
                axes[i,j].axis('off')
                continue

            slice_num = kept_slices[cnt]
            display_slice(nifti_data, mask_data, slice_num, axes[i,j])

    # Tight layout to remove extra space
    plt.tight_layout()  # small padding between axes
    plt.subplots_adjust(wspace=0.05, hspace=0.05)  # minimal whitespace
    plt.close(fig)
    
    return fig

if __name__ == '__main__':
    # CHANGE THESE
    pdf_dir = Path('/data/vision/polina/users/marcusbl/bin_class/label_sessions_data/label_session_3-11')
    stacks_per_pdf = 50

    # GET DATA
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Can't find the specified pdf_dir: {pdf_dir}")
    display_df = pd.read_csv(pdf_dir / 'display_df.csv')

    # CREATE MOSAICS
    pdf = None

    all_rows = list(display_df.iterrows())
    all_rows = all_rows + all_rows  # duplicate
    np.random.shuffle(all_rows)     # shuffle in place

    counter_to_path = {} # (pdf_counter, stack_counter) -> stack_path 

    for i, (_, row) in tqdm(list(enumerate(all_rows))):
        stack_counter = i % stacks_per_pdf

        if i % stacks_per_pdf == 0:
            if pdf is not None:
                pdf.close()

            pdf_counter = i // stacks_per_pdf

            # Create Files
            sbdir = (pdf_dir / f'group{pdf_counter}')
            sbdir.mkdir(exist_ok=True)

            # PDF
            pdf = PdfPages(sbdir / f"stacks_{pdf_counter}.pdf")

            # Label Sheet
            tracker_df = pd.DataFrame("", columns=range(1, stacks_per_pdf), index=range(100))

            index_labels = ["start (inc)", "end (inc)"] + [""] * (len(tracker_df) - 2)
            tracker_df.index = index_labels

            # Put dashes in the 3rd row
            tracker_df.iloc[2] = "-"            
            tracker_df.to_csv(sbdir / f'labels_{pdf_counter}.csv')

        fig = display_stack(row['path'], row['mask_path'])
        counter_to_path[str(pdf_counter) + "-" + str(stack_counter)] = row['path']
        with open(pdf_dir / 'lookup.json', 'w') as f:
            json.dump(counter_to_path, f, indent = 2)
            
        fig.text(
            0.99, 0.01,
            f"Stack {stack_counter}",
            color='red',
            fontsize=14,
            weight='bold',
            va='bottom',
            ha='right'
        )

        pdf.savefig(fig)
        plt.close(fig)
