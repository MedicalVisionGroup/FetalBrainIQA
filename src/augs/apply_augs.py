from torchvision import transforms
from src.data import DicomDataset

from src.augs.augs_list import get_default_transform_list, get_color_transform_list, get_spatial_transform_list

def apply_augs(train_dataset: DicomDataset, val_dataset: DicomDataset, method = '',
               perc = .02) -> None:
    """
    Applies a series of transformations

    1) Calculate mean/std from train_dataset & applies normalization
    2) Spatial augmentations to train if 's' in method
    3) Color   augmentations to train if 'c' in method
    4) Duplicates the img to 3D for the ResNet

    """
    basics = get_default_transform_list(perc=perc, inc_mask_channel=train_dataset.inc_mask_channel)
    spatial_transform = get_spatial_transform_list()
    color_transform = get_color_transform_list(inc_mask_channel=train_dataset.inc_mask_channel)

    augmentations = []
    if 's' in method:
        print("Applying Spatital Augmentations")
        augmentations.extend(spatial_transform)
    if 'c' in method:
        print("Applying Color Augmentations")
        augmentations.extend(color_transform)

    train_transform = basics[:-1] + augmentations + basics[-1:]
    val_transform = basics

    train_dataset.transform = transforms.Compose(train_transform)
    val_dataset.transform = transforms.Compose(val_transform)

    
