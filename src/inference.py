# python -m src.inference

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.model import DiagnosticModel

model_weights_path = Path(
    "/data/vision/polina/users/marcusbl/bin_class/inference_model/model_auc.pth"
)

def minmax_normalize(
    img: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    img:  shape (C, H, W)
    mask: shape (H, W), boolean
    """

    if img.ndim != 3:
        raise ValueError(f"`img` must have shape (C, H, W), got {tuple(img.shape)}")

    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError(f"`mask` must be bool, got {mask.dtype}")
        if mask.shape != img.shape[1:]:
            raise ValueError(
                f"`mask` shape must match image spatial shape. "
                f"Got mask {tuple(mask.shape)} and image {tuple(img.shape)}"
            )

    channels_to_norm = img[:-1]

    if mask is not None and mask.any():
        values = channels_to_norm[:, mask]
    else:
        values = channels_to_norm[channels_to_norm > 0]

    if values.numel() == 0:
        raise ValueError("No nonzero values found for normalization")

    img_min = torch.quantile(values, 0.0)
    img_max = torch.quantile(values, 1.0)

    normalized = (channels_to_norm - img_min) / (img_max - img_min + 1e-6)
    normalized = torch.clamp(normalized, 0, 1)

    return torch.cat([normalized, img[-1:]], dim=0)

def do_inference(
    slice: torch.Tensor,
    mask: torch.Tensor,
    device: str = "cuda",
) -> float:
    """
    Takes in:
      slice: float tensor of shape (H, W)
      mask:  bool tensor of shape (H, W)

    Returns:
      probability of corruption as a float.
    """

    # ----------------
    # 0. Data checks
    # ----------------

    if not isinstance(slice, torch.Tensor):
        raise TypeError(f"`slice` must be a torch.Tensor, got {type(slice)}")

    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"`mask` must be a torch.Tensor, got {type(mask)}")

    if slice.ndim != 2:
        raise ValueError(f"`slice` must have shape (H, W), got {tuple(slice.shape)}")

    if mask.ndim != 2:
        raise ValueError(f"`mask` must have shape (H, W), got {tuple(mask.shape)}")

    if slice.shape != mask.shape:
        raise ValueError(
            f"`slice` and `mask` must have same shape, "
            f"got slice {tuple(slice.shape)} and mask {tuple(mask.shape)}"
        )

    if not torch.is_floating_point(slice):
        raise TypeError(f"`slice` must be floating point, got {slice.dtype}")

    if mask.dtype != torch.bool:
        raise TypeError(f"`mask` must be bool, got {mask.dtype}")

    # ----------------
    # 1. Device
    # ----------------

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    device = torch.device(device)

    # ----------------
    # 2. Preprocessing
    # ----------------

    slice = slice.float()
    mask = mask.bool()

    resize_img = transforms.Resize((244, 244), antialias=True)
    resize_mask = transforms.Resize(
        (244, 244),
        interpolation=InterpolationMode.NEAREST,
    )

    # Resize image first
    slice = resize_img(slice.unsqueeze(0)).squeeze(0)  # (H, W)

    # Resize mask separately with nearest-neighbor interpolation
    mask = resize_mask(mask.unsqueeze(0).float()).squeeze(0).bool()  # (H, W)

    # Stack channels: image, image, mask
    img = torch.stack(
        [
            slice,
            slice,
            mask.float(),
        ],
        dim=0,
    )  # (3, H, W)

    img = minmax_normalize(img, mask=mask)

    img = img.unsqueeze(0).to(device)  # (1, 3, H, W)

    # ----------------
    # 3. Load model
    # ----------------
    if not model_weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_weights_path}")

    model = DiagnosticModel(model_name = 'resnet50')
    state_dict = torch.load(model_weights_path, map_location=device, weights_only=True) 
    model.load_state_dict(state_dict) 
    model = model.to(device = device)    
    model.eval()

    # ----------------
    # 4. Inference
    # ----------------

    with torch.no_grad():
        logits = model(img)
        probs = F.softmax(logits, dim=1)
        corruption_prob = probs[0, 1]

    return float(corruption_prob.item())


# ---- TESTING ------
# -------------------

if __name__ == "__main__":
    # 1. Choose a slice and mask
    stack_path = Path(
        "/data/vision/polina/users/marcusbl/data/anon-00015/stack_5/clean/dicoms.npy"
    )
    mask_path = Path(
        "/data/vision/polina/users/marcusbl/data/anon-00015/stack_5/clean/masks.npy"
    )
    slice_idx = 16

    # 2. Load arrays
    stack_data = np.load(stack_path)
    mask_data = np.load(mask_path)

    # 3. Convert to tensors
    slice_tensor = torch.tensor(
        stack_data[:, :, slice_idx],
        dtype=torch.float32,
    )

    # FIXED: use mask_data here, not stack_data
    mask_tensor = torch.tensor(
        mask_data[:, :, slice_idx],
        dtype=torch.bool,
    )

    prob = do_inference(slice_tensor, mask_tensor)

    print(f"Probability of corruption: {prob:.6f}")