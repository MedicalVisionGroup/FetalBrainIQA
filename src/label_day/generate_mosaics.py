import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from torch.utils.data import DataLoader
import torch

from tqdm import tqdm
import pandas as pd

pd.set_option("display.max_colwidth", None)


bch_path = Path('/data/vision/polina/users/mfirenze/Data_sharing_MIT_Margherita')
bch_info_path = bch_path / 'marcus_info.csv'


bch_df = pd.read_csv(bch_info_path)

# 1. Replace beginning path of data locations
bch_df['path'] = str(bch_path) + bch_df['Data Location'].str.split('mnt1').str[1]
bch_df = bch_df.drop(columns = 'Data Location')

# 2. Get the person
bch_df["person"] = bch_df["path"].str.extract(r"(?:processed|failed)/(.*?)/raw/")

# 3. Get the mask location 
bch_df["mask_path"] = (
    bch_df['path'].str.replace("/raw/", "/masks/", regex=False)
    .str.replace(r"\.nii$", "_mask.nii", regex=True)
)

# 4. Add a flag for dataset
bch_df['dataset'] = 'BCH'

bch_df.head(5)


ramya_path = Path('/data/vision/polina/users/marcusbl/data')


rows = []

for person_dir in ramya_path.iterdir():
    if not person_dir.is_dir():
        continue

    for stack_dir in person_dir.iterdir(): 
        if not stack_dir.is_dir():
            continue

        rows.append({
            'path': stack_dir / 'clean' / 'dicoms.npy',
            'Brain Type': None,
            'person': person_dir.stem,
            'dataset': "R",
            'mask_path': stack_dir / 'clean' / 'masks.npy'
        })

ramya_df = pd.DataFrame(rows)


assert sorted(ramya_df.columns) == sorted(bch_df.columns)
df = pd.concat([ramya_df, bch_df])
df = df.reset_index(drop=True)


all_people = set(df["person"])
person_to_id = {p:i for i,p in enumerate(all_people)}
id_to_person = {i:p for i,p in enumerate(all_people)}
df['person_id'] = df["person"].map(person_to_id).astype(int)

assert df['person_id'].max() == df['person'].nunique() - 1

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


def display_mosaic(scan_path: Path, mask_path: Path, save_path: Path = None):
    """
    Will apply min-max normalization w perc = .01 wrt the mask,
    
    If mask is too small, it'll display it normally and flag it
    """
    # Download nifti data and mask data
    if scan_path.suffix == '.npy':
        nifti_data = np.load(scan_path)
    else:
        nifti_img = nib.load(scan_path)
        nifti_data = nifti_img.get_fdata()

    if mask_path.suffix == '.npy':
        mask_data = np.load(mask_path).astype(bool)
    else:
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata().astype(bool)

    # Calculate the # of scans [ignore unmasked beginning and end]
    start, end = 100, 0
    for scan_num in range(nifti_data.shape[-1]):
        mask = mask_data[:, :, scan_num] 

        if mask.sum() > 0:
            start = min(start, scan_num)
            end = max(end, scan_num)

    total_scan_cnt = int(nifti_data.shape[-1]) # scans should range from 0 to value - 1
    all_scans = list(range(start, end + 1)) # [start, end] inclusive 
    num_scans = len(all_scans)

    # Create the Grid
    if num_scans == 0:
        print(f"No masks found for {scan_path}")
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.set_title("NO DATA")
        return fig, [], 0


    nrows = int(np.floor(np.sqrt(num_scans)))
    ncols = int(np.ceil(num_scans / nrows))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(2 * ncols, 2 * nrows)
    )

    # Ensure axes is always 2D
    axes = np.atleast_2d(axes)

    for i in range(nrows):
        for j in range(ncols):
            cnt = i * ncols + j

            # stop if at the end
            if cnt >= num_scans:
                axes[i,j].axis('off')
                continue
            
            scan_num = all_scans[cnt]

            mask = mask_data[:, :, scan_num].astype(bool)
            scan = nifti_data[:, :, scan_num]

            if mask.sum() > 30:
                img = minmax(scan, mask, perc = .01)
                axes[i,j].imshow(img, cmap='grey', vmin=0,vmax = 1)
                axes[i,j].imshow(mask, cmap="Reds", alpha=0.4)
            else:
                axes[i,j].imshow(scan, cmap ='grey')
                axes[i,j].imshow(mask, cmap="Reds", alpha=0.4)
                axes[i, j].text(
                    0.98, 0.98, "⚠",
                    transform=axes[i, j].transAxes,
                    ha="right",
                    va="top",
                    fontsize=14,
                    color="red",
                    weight="bold"
                )
            axes[i,j].text(
                0.02, 0.98, f'{scan_num}; Area = {int(mask.sum())}',
                transform=axes[i,j].transAxes,
                ha = "left",
                va = "top", 
                fontsize = 14,
                color='yellow',
                weight='bold'
            )

            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j].set_frame_on(False)

    # Tight layout to remove extra space
    plt.tight_layout()  # small padding between axes
    plt.subplots_adjust(wspace=0.05, hspace=0.05)  # minimal whitespace
    plt.close(fig)
    
    return fig, all_scans, total_scan_cnt




num_redundant = 30
num_stacks_per_pdf = 50

listA = []
listB = []

person_map = df.groupby(by='person_id')

for person_id, person_df in person_map:
    num_stacks = len(person_df)

    idxsA = np.random.choice(num_stacks, size=min(2, num_stacks), replace=False)
    idxsB = list(set(range(num_stacks)) - set(idxsA))

    # convert local idx -> global df index
    listA.extend(person_df.iloc[idxsA].index.tolist()) 
    listB.extend(person_df.iloc[idxsB].index.tolist())


# Test mutually exclusive decomposition
assert set(listA) & set(listB) == set()
assert set(listA) | set(listB) == set(range(len(df)))


np.random.shuffle(listA)
listA = listA + listA[:num_redundant]

pdf_dir = Path('/data/vision/polina/users/marcusbl/bin_class/outputs_mosaics_masked')
pdf_dir.mkdir(exist_ok=True)

pdf_counter = 1
stack_counter = 1
figs = []

print(len(listA))
print(len(listB))

rows = []
for idx in tqdm(listA + listB): # do listA before listB
    scan_path = Path(df.loc[idx, 'path'])
    mask_path = Path(df.loc[idx, 'mask_path'])


    
    fig, all_scans, total_scan_cnt = display_mosaic(scan_path, mask_path, save_path = None)   

    rows.append({
        'pdf_num': pdf_counter,
        'stack_num': stack_counter,
        'path': scan_path,
        'mask_path': mask_path,
        'type': df.loc[idx, 'Brain Type'],
        'person': df.loc[idx, 'person'],
        'person_id': df.loc[idx, 'person_id'],
        'dataset': df.loc[idx, 'dataset'],
        'all_scans': all_scans,
        'total_scan_cnt': total_scan_cnt,
    })    

    fig.text(
        0.99, 0.01,               # normalized figure coords (0,0 bottom-left; 1,1 top-right)
        f"Stack {stack_counter}",  
        color='red',
        fontsize=14,
        weight='bold',
        va='bottom',                  # vertical alignment
        ha='right'                  # horizontal alignment
    )

    figs.append(fig)

    stack_counter += 1
    if stack_counter > num_stacks_per_pdf:
        # Make the pdf
        print(f"Starting PDF {pdf_counter}")
        sbdir = (pdf_dir / f'group{pdf_counter}')
        sbdir.mkdir(exist_ok=True)

        with PdfPages(sbdir / f"images_{pdf_counter}.pdf") as pdf:
            for fig in figs:
                pdf.savefig(fig)
                plt.close(fig)

        tracker_df = pd.DataFrame("", columns=range(1, stack_counter), index = range(100))
        tracker_df.to_csv(sbdir / f'labels_{pdf_counter}.csv')

        print(f"Finished PDF {pdf_counter}")

        # Move onto next pdf
        figs = []
        pdf_counter += 1
        stack_counter = 1
    
    pd.DataFrame(rows).to_csv(pdf_dir / 'df.csv')

if figs:
    # Make the pdf
    print(f"Starting PDF {pdf_counter}")
    sbdir = (pdf_dir / f'group{pdf_counter}')
    sbdir.mkdir(exist_ok=True)
    with PdfPages(sbdir / f"images_{pdf_counter}.pdf") as pdf:
        for fig in figs:
            pdf.savefig(fig)
            plt.close(fig)

    tracker_df = pd.DataFrame("", columns=range(1, stack_counter), index = range(100))
    tracker_df.to_csv(sbdir / f'labels_{pdf_counter}.csv')

    print(f"Finished PDF {pdf_counter}")

pd.DataFrame(rows).to_csv(pdf_dir / 'df.csv')