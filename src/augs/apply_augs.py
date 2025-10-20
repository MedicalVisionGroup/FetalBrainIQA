from torchvision import transforms
from src.data import DicomDataset

from src.augs.augs_basic import default_img_transform_list

def apply_augs(train_dataset: DicomDataset, val_dataset: DicomDataset, method = '') -> None:
    """
    Applies a series of transformations

    1) Calculate mean/std from train_dataset & applies normalization
    2) Spatial augmentations to train if 's' in method
    3) Color   augmentations to train if 'c' in method
    4) Duplicates the img to 3D for the ResNet18

    """
    basics = default_img_transform_list

    spatial_transform = [
        transforms.RandomHorizontalFlip(p=0.5),  
        transforms.RandomVerticalFlip(p=0.5),    
        transforms.RandomAffine(
            degrees = 90,
            translate = (0.2, 0.2), # 30% percent in both directions
            scale = (0.6, 1.4)        # 40% scale in either direction 
        )
    ]

    color_transform = [
        transforms.ColorJitter(
            brightness=0.8, contrast=0.7, saturation=0.7
        )                                   # random color changes
    ]

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

    
