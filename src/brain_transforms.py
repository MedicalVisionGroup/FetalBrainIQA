import torch
from torchvision import transforms
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms.functional as F
import math
import random

class CustomTransform:
    def __init__(self):
        pass

    def __call__(self):
        pass

    def mask_moves_outside(self, mask: torch.Tensor, *params) -> bool:
        pass

def get_spatial_transform_list(trans_perc: float = 0.2, translate_far: bool = False) -> list[CustomTransform]:
    return [
        RandomFlip(p=0.5, dim = 1),  
        RandomFlip(p=0.5, dim = 2),    
        RandomAffineTransform(
            degree_range = 90,
            translate = (trans_perc, trans_perc),   # default 20% in either direction
            scale = (0.8, 1.2),        # 20% scale in either direction,
            translate_far = translate_far,  
        )
    ]

def get_color_transform_list() -> list[CustomTransform]:
    return []

class RandomFlip(CustomTransform):
    """
    Flips a tensor of the form (C, W, H) across any specified dimension
    """
    def __init__(self, p: float = 0.5, dim: int = -1):
        self.p = p
        self.dim = dim

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            return torch.flip(img_tensor, dims=[self.dim])
        return img_tensor

    def mask_moves_outside(self, mask: torch.Tensor):
        return False # mask can't move outside from a horizontal or vertical flip
     
class RandomAffineTransform(CustomTransform):
    """
    Applies an AffineTransform to tensor of form (C, W, H)
    """

    def __init__(self, degree_range: float, translate: tuple[float, float], scale:tuple[float, float], 
                 shear: float = None, translate_far: bool = False):
        self.degree_range = degree_range
        self.translate_bounds = translate
        self.scale_bounds = scale
        self.shear_bound = shear

        # Start as None until a gen_params() is called
        self.angle = None
        self.translate = None
        self.scale = None
        self.shear = None 

        self.translate_far = translate_far # False: uniform; True: close to ends in gen_params

    def gen_params(self, w, h):
        # rotation
        angle = random.uniform(-self.degree_range, self.degree_range)

        # translation (fraction of image size)
        max_dx = self.translate_bounds[0] * w
        max_dy = self.translate_bounds[1] * h

        if not self.translate_far:
            tx = random.uniform(-max_dx, max_dx)
            ty = random.uniform(-max_dy, max_dy)
        else:
            tx = random.choice([random.uniform(-max_dx, 0.95 * -max_dx), random.uniform(max_dx, 0.95 * max_dx)])
            ty = random.choice([random.uniform(-max_dy, 0.95 * -max_dy), random.uniform(max_dy, 0.95 * max_dy)])

        # scale
        scale = random.uniform(self.scale_bounds[0], self.scale_bounds[1])

        # shear
        if self.shear_bound is not None:
            shear_x = random.uniform(-self.shear_bound, self.shear_bound)
            shear_y = random.uniform(-self.shear_bound, self.shear_bound)
            shear = [shear_x, shear_y]
        else:
            shear = [0.0, 0.0]

        # Set the values!
        self.angle = angle
        self.translate = (tx, ty)
        self.scale = scale
        self.shear = shear
    
    def __call__(self, x: torch.Tensor):
        """
        x: Tensor of shape (C, W, H)
        """
        if x.ndim != 3:
            raise ValueError("Expected tensor of shape (C, W, H)")

        C, W, H = x.shape

        self.gen_params(W, H)

        output = F.affine(
            x,
            angle=self.angle,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
        )

        return output

    def _build_affine_matrix(self, W, H):
        """
        Builds a 2x3 affine matrix matching torchvision's F.affine behavior.
        """

        angle = math.radians(self.angle)
        shear_x = math.radians(self.shear[0])
        shear_y = math.radians(self.shear[1])

        # Rotation + scale
        R = torch.tensor([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle),  math.cos(angle)],
        ]) * self.scale

        # Shear
        Sh = torch.tensor([
            [1, math.tan(shear_x)],
            [math.tan(shear_y), 1],
        ])

        A = R @ Sh

        # Image center
        cx = W / 2
        cy = H / 2
        center = torch.tensor([cx, cy])

        # Translation
        t = torch.tensor(self.translate)

        # Centered translation correction
        t_centered = t + center - A @ center

        # 2x3 matrix
        M = torch.cat([A, t_centered.view(2, 1)], dim=1)
        return M
    
    def mask_moves_outside(
        self,
        mask: torch.Tensor,
    ):
        """
        mask: (W, H) binary mask
        returns True/False is pixels move outside fov
        """

        W, H = mask.shape

        # Use first channel (or assert all identical)
        m = mask

        # 1) get ON pixel coordinates
        ys, xs = torch.where(m > 0)

        coords = torch.stack([xs, ys], dim=1).float()  # (N, 2)

        # 2) homogeneous coords
        ones = torch.ones(len(coords), 1)
        coords_h = torch.cat([coords, ones], dim=1)  # (N, 3)

        # 3) affine transform
        M = self._build_affine_matrix(W, H)
        coords_t = (M @ coords_h.T).T  # (N, 2)

        # 4) check bounds
        x, y = coords_t[:, 0], coords_t[:, 1]

        out_of_bounds = (
            (x < 0) | (x >= W) |
            (y < 0) | (y >= H)
        )

        return torch.any(out_of_bounds)