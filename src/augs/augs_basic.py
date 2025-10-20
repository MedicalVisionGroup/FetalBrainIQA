import torch
from torchvision import transforms

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
        img_min = torch.quantile(img, self.perc)
        img_max = torch.quantile(img, 1 - self.perc)

        img = (img - img_min) / (img_max - img_min)  # scale to 0-1
        img = img * (self.max_val - self.min_val) + self.min_val
        img = torch.clip(img, self.min_val, self.max_val)

        return img


default_img_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((244, 244)),
        MinMaxNormalize(0.0, 1.0, perc = 0.02),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ]
)