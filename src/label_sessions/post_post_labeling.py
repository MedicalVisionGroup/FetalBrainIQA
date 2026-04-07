# python -m src.label_sessions.post_post_labeling

"""
This script generates the slices that Ramya and I labeled following the 2nd data session where there was disagreement
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import json

from src.label_sessions.generate_mosaics import minmax

df = pd.read_csv('/data/vision/polina/users/marcusbl/bin_class/label_sessions_data/label_session_3-11/post_session.csv', index_col = 0)
relabel_df = df[df['req_relabel'].fillna(False)]

slices_per_pdf = [len(relabel_df) // 2, len(relabel_df) - len(relabel_df) // 2]

def display_slice(stack_path: Path, mask_path: Path, slice_num: int, cnt: int, axis, no_extra: bool = False):
    # Load nifti data
    if stack_path.suffix == '.npy':
        nifti_data = np.load(stack_path)
    else:
        nifti_img = nib.load(stack_path)
        nifti_data = nifti_img.get_fdata()

    # Load mask data
    if mask_path.suffix == '.npy':
        mask_data = np.load(mask_path).astype(bool)
    else:
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata().astype(bool)


    mask = mask_data[:, :, slice_num].astype(bool)
    slice = nifti_data[:, :, slice_num]

    if mask.sum() > 30:
        img = minmax(slice, mask, perc = .01)
        axis.imshow(img, cmap='grey', vmin=0,vmax = 1)
    else:
        axis.imshow(slice, cmap ='grey')
        axis.text(
            0.98, 0.98, "⚠",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=14,
            color="red",
            weight="bold"
        )

    # Draw Top Words & Mask
    expanded_mask = binary_dilation(mask, iterations=4)

    if not no_extra:
        axis.contour(
            expanded_mask,
            levels=[0.5],      # boundary between 0 and 1
            colors='red',
            linewidths=0.25
        )

        axis.text(
            0.02, 0.98,
            f'{cnt}',
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            color='yellow',
            weight='bold',
            bbox=dict(
                facecolor='black',
                alpha=0.7,
                pad=3,
                edgecolor='none'
            )
        )

    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_frame_on(False)


sbdir = Path('/data/vision/polina/users/marcusbl/bin_class/label_sessions_data/label_session_3-11/post_label_session')

# Constants for organization
ROWS, COLS = 5, 4
SLICES_PER_PAGE = ROWS * COLS  # 20
PAGES_PER_PDF = 100
SLICES_PER_PDF = SLICES_PER_PAGE * PAGES_PER_PDF

pdf_counter = 0
pdf = None
fig = None
axes = None

slice_lookup = {} # (pdf_num, idx) -> (path, slice_num)

# Iterate through the dataframe
for idx, (_, row) in enumerate(tqdm(relabel_df.iterrows(), total=len(relabel_df))):
    
    # 1. Start a new PDF every 1000 slices
    if idx % SLICES_PER_PDF == 0:
        pdf_counter += 1
        pdf_path = sbdir / f"temp_slices_{pdf_counter}.pdf"
        pdf = PdfPages(pdf_path)

    # 2. Start a new Page (Figure) every 20 slices
    if idx % SLICES_PER_PAGE == 0:
        fig, axes = plt.subplots(ROWS, COLS, figsize=(8.5, 11))
        axes = axes.flatten()

    # 3. Display the current slice on the current axis
    ax_idx = idx % SLICES_PER_PAGE
    slice_idx = idx - (SLICES_PER_PDF * (pdf_counter-1))
    
    slice_lookup[f"{pdf_counter}-{slice_idx}"] = (row['path'], int(row['slice_num']))
    display_slice(
        Path(row['path']),
        Path(row['mask_path']),
        int(row['slice_num']),
        slice_idx,
        axes[ax_idx],
        # no_extra = True
    )

    # 4. End of Page or End of Data: Save the figure
    is_last_slice_on_page = (idx % SLICES_PER_PAGE == SLICES_PER_PAGE - 1)
    is_final_total_slice = (idx == len(relabel_df) - 1)

    if is_last_slice_on_page or is_final_total_slice:
        # Hide remaining empty subplots on the last page if any
        for j in range(ax_idx + 1, SLICES_PER_PAGE):
            axes[j].axis('off')
            
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # 5. End of PDF or End of Data: Close the PDF
    is_last_slice_in_pdf = (idx % SLICES_PER_PDF == SLICES_PER_PDF - 1)
    
    if is_last_slice_in_pdf or is_final_total_slice:
        pdf.close()

    with open(sbdir / 'lookup.json', 'w') as f:
        json.dump(slice_lookup, f, indent = 2)

print(f"Done! Saved {pdf_counter} PDFs to {sbdir}")

# # Generate CSV files
for i in range(1, pdf_counter+1):
    df = pd.DataFrame({'index': range(1001), 'label': [''] * 1001})
    df.to_csv(sbdir / f'labels_{i}.csv', index=False)