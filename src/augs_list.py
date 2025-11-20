import torch
from torchvision import transforms
from torchvision.transforms import v2

class MinMaxNormalize:
    """
    Normalizes an image so that all values run between min_val and max_val. 
    
    There's an option for choosing min = perc, max = 1 - perc to avoid
    outliers messing up the distribution. This still clips to [0, 1]
    """
    def __init__(self, min_val=0.0, max_val=1.0, perc = 0):
        self.min_val = min_val
        self.max_val = max_val
        self.perc = perc

    def __call__(self, img):
        # img assumed to be a torch.Tensor (C, H, W)
        nz_values = img[img > 0] # ignore the 0 value when normalizing

        img_min = torch.quantile(nz_values, self.perc)
        img_max = torch.quantile(nz_values, 1 - self.perc)

        img = (img - img_min) / (img_max - img_min)  # scale to 0-1
        img = img * (self.max_val - self.min_val) + self.min_val
        img = torch.clip(img, self.min_val, self.max_val)

        return img

def get_default_transform_list(perc = .02, inc_mask_channel: bool = False):
    trfm = [
        transforms.ToTensor(),
        transforms.Resize((244, 244)),
        MinMaxNormalize(0.0, 1.0, perc = perc),
    ]

    if not inc_mask_channel: # if not using mask channel, then repeat the image 3 times
         trfm.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))

    return trfm


def get_spatial_transform_list():
    return [
        transforms.RandomHorizontalFlip(p=0.5),  
        transforms.RandomVerticalFlip(p=0.5),    
        transforms.RandomAffine(
            degrees = 90,
            translate = (0.2, 0.2),   # 20% percent in both directions
            scale = (0.6, 1.4)        # 40% scale in either direction 
        )
    ]

def get_color_transform_list(inc_mask_channel: bool = False):
    if inc_mask_channel:
        return []
    
    return [
            transforms.ColorJitter( # random color changes
                brightness=0.8, contrast=0.7, saturation=0.7
            ),    
            # v2.GaussianNoise(mean = 0, sigma = 0.001, clip = True)  # worse performance based on tests                     
        ]
