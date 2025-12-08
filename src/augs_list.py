import torch
from torchvision import transforms

class CustomNormalize:
    """
    Normalizes an image. 

    1. PERC = quantiles for min/max [choosing min = perc, max = 1 - perc to avoid
    outliers messing up the distribution. This still clips to [0, 1]
    2. MASK = optional mask for using only masked pixels
    3. METHOD: min-max ([0,1]) or peak-squash (squishes most freq -> 1/2 and 0 -> 0; rest linear)
    """
    def __init__(self, perc: float = 0, method: str = None):
        self.perc = perc
        self.method = method

    def __call__(self, img: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:            
        if self.method is None:
            return img
        elif self.method == "min-max":
            return self.minmax(img, mask)
        elif self.method == "peak-squash":
            return self.peaksquash(img, mask)
        else:
            raise ValueError(f"Unknown normalization method specified: {self.method}\nValid are: min-max, peak-squash")
        
    def peaksquash(self, img: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Sends most freq -> 1/2 and 0 -> 0; linearly scales the rest.
        img: (C, H, W)
        mask: (H, W) boolean or 0/1
        """
        # extract non-zero (or masked) values for estimating peak
        if mask is not None:
            assert mask.dtype == torch.bool
            assert img.ndim == 3 and mask.shape == img.shape[1:]
            nz_values = img[0][mask]               # 1D
        else:
            nz_values = img[0].flatten()

        # compute histogram peak
        counts, bin_edges = torch.histogram(nz_values, bins=200)
        peak_bin_index = torch.argmax(counts)
        x_peak = (bin_edges[peak_bin_index] + bin_edges[peak_bin_index + 1]) / 2

        # avoid divide-by-zero
        eps = 1e-6
        return  img / (2 * (x_peak + eps))
            
    def minmax(self, img: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        img: (C, H, W)
        mask: (H, W) boolean or 0/1
        """

        if mask is not None:
            assert mask.dtype == torch.bool
            assert img.ndim == 3 and mask.shape == img.shape[1:]
            nz_values = img[0][mask]               # 1D
        else:
            nz_values = img[img > 0]            # avoid all the zeros

        img_min = torch.quantile(nz_values, self.perc)
        img_max = torch.quantile(nz_values, 1 - self.perc)

        img = (img - img_min) / (img_max - img_min + 1e-6)  # scale to 0-1
        img = torch.clip(img, 0, 1)


        return img

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

def get_color_transform_list(mask_method: str | None = None):
    if mask_method == 'stack':
        return []
    
    return [
            transforms.ColorJitter( # random color changes
                brightness=0.8, contrast=0.7, saturation=0.7
            ),    
            # v2.GaussianNoise(mean = 0, sigma = 0.001, clip = True)  # worse performance based on tests                     
        ]
