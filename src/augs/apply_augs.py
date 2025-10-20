from torchvision import transforms
from src.data import DicomDataset

from src.augs.augs_list import default_img_transform_list, spatial_transform_list, color_transform_list

def apply_augs(train_dataset: DicomDataset, val_dataset: DicomDataset, method = '') -> None:
    """
    Applies a series of transformations

    1) Calculate mean/std from train_dataset & applies normalization
    2) Spatial augmentations to train if 's' in method
    3) Color   augmentations to train if 'c' in method
    4) Duplicates the img to 3D for the ResNet18

    """
    basics = default_img_transform_list
    spatial_transform = spatial_transform_list
    color_transform = color_transform_list

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

    
