"""
U²-Net inference for salient object detection.

Downloads pre-trained weights if not already present, then runs the model
on dermoscopy images.  U²-Net produces pixel-level saliency without any
class label — it highlights "what is salient" as a pure vision task,
making it complementary to gradient-based CAM methods.

Qin, X. et al. U2-Net: Going deeper with nested U-structure for salient
object detection. Pattern Recognition, 2020.
"""
import os
import sys
import subprocess

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.models.u2net import U2NET
from src.utils.config import U2NET_CKPT, DEVICE, WEIGHTS_DIR

# U²-Net was trained at 320×320
_U2NET_SIZE = 320

# Google-Drive direct download link for the pre-trained weights
_U2NET_GDRIVE_ID = "1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ"


# ---------------------------------------------------------------------------
# Weight download
# ---------------------------------------------------------------------------

def download_u2net_weights() -> None:
    """Download U²-Net weights from Google Drive via gdown."""
    if os.path.isfile(U2NET_CKPT):
        print(f"U²-Net weights already present: {U2NET_CKPT}")
        return
    try:
        import gdown
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True)
        import gdown

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    url = f"https://drive.google.com/uc?id={_U2NET_GDRIVE_ID}"
    print(f"Downloading U²-Net weights from Google Drive → {U2NET_CKPT} …")
    gdown.download(url, U2NET_CKPT, quiet=False)
    print("Download complete.")


# ---------------------------------------------------------------------------
# Model loader (singleton pattern — expensive to reload)
# ---------------------------------------------------------------------------

_u2net_model: U2NET | None = None


def load_u2net(force_reload: bool = False) -> U2NET:
    """Return a cached U²-Net model.  Downloads weights if missing."""
    global _u2net_model
    if _u2net_model is not None and not force_reload:
        return _u2net_model

    download_u2net_weights()
    model = U2NET(in_ch=3, out_ch=1)
    state = torch.load(U2NET_CKPT, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    _u2net_model = model
    print("U²-Net loaded and ready.")
    return model


# ---------------------------------------------------------------------------
# Preprocessing / postprocessing helpers
# ---------------------------------------------------------------------------

_U2NET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_U2NET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(img_rgb: np.ndarray) -> torch.Tensor:
    """H×W×3 uint8 → 1×3×320×320 float32 tensor on DEVICE."""
    img = cv2.resize(img_rgb, (_U2NET_SIZE, _U2NET_SIZE)).astype(np.float32) / 255.0
    img = (img - _U2NET_MEAN) / _U2NET_STD
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    return tensor


def _normalise(sal: np.ndarray) -> np.ndarray:
    lo, hi = sal.min(), sal.max()
    if hi - lo < 1e-8:
        return np.zeros_like(sal)
    return (sal - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Public inference API
# ---------------------------------------------------------------------------

@torch.no_grad()
def u2net_saliency(img_rgb: np.ndarray) -> np.ndarray:
    """Run U²-Net on a single RGB image.

    Args:
        img_rgb: H×W×3 uint8 numpy array.

    Returns:
        Normalised saliency map float32 [0, 1] same spatial size as input.
    """
    model  = load_u2net()
    h, w   = img_rgb.shape[:2]
    tensor = _preprocess(img_rgb)

    outputs = model(tensor)          # returns tuple: (d0, d1, d2, d3, d4, d5, d6)
    d0      = outputs[0]             # fused output — most accurate single prediction

    sal = d0.squeeze().cpu().numpy()  # H_out × W_out (320×320)
    sal = _normalise(sal)
    sal = cv2.resize(sal, (w, h), interpolation=cv2.INTER_LINEAR)
    return _normalise(sal)


@torch.no_grad()
def u2net_saliency_from_tensor(tensor_chw: torch.Tensor) -> np.ndarray:
    """Convenience wrapper accepting a normalised C×H×W tensor (DataLoader output).

    The tensor is assumed to be ImageNet-normalised.  U²-Net expects its own
    normalisation, so we undo ImageNet stats before re-applying U²-Net stats.
    """
    _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    _IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_01 = (tensor_chw.cpu() * _IMAGENET_STD + _IMAGENET_MEAN).clamp(0, 1)
    img_np = (img_01.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return u2net_saliency(img_np)
