"""
Six gradient-based and perturbation-based CAM saliency methods.

All methods use the pytorch-grad-cam library (Jacobgilpy, 2020–2024).
Reference: https://github.com/jacobgil/pytorch-grad-cam

Methods implemented
-------------------
GradCAM      — Selvaraju et al., ICCV 2017
GradCAM++    — Chattopadhay et al., WACV 2018
ScoreCAM     — Wang et al., CVPR 2020  (perturbation-based, gradient-free)
LayerCAM     — Jiang et al., IEEE TIP 2021  (local + global fusion)
EigenCAM     — Muhammad & Yeasin, arXiv 2020  (PCA, no backprop)
XGradCAM     — Fu et al., BMVC 2020  (axiom-based gradient refinement)
"""
import numpy as np
import torch
import cv2
from torch.cuda.amp import autocast

from pytorch_grad_cam import (
    GradCAM,
    GradCAMPlusPlus,
    ScoreCAM,
    LayerCAM,
    EigenCAM,
    XGradCAM,
)
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from src.utils.config import DEVICE, AMP, IMG_SIZE


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_target_layer(model: torch.nn.Module):
    """Last convolutional block of EfficientNet-B4 (timm)."""
    return [model.blocks[-1]]


def _tensor_to_numpy(tensor_chw: torch.Tensor) -> np.ndarray:
    """C×H×W float32 tensor → H×W×C uint8 numpy array."""
    img = tensor_chw.detach().cpu().numpy().transpose(1, 2, 0)
    # Undo ImageNet normalisation for visualisation purposes
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = (img * std + mean).clip(0, 1)
    return (img * 255).astype(np.uint8)


def _normalise(sal: np.ndarray) -> np.ndarray:
    """Min-max normalise a saliency map to [0, 1]."""
    lo, hi = sal.min(), sal.max()
    if hi - lo < 1e-8:
        return np.zeros_like(sal)
    return (sal - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Single-method runner
# ---------------------------------------------------------------------------

def run_cam(
    method_name: str,
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Run a single CAM method on one image tensor.

    Args:
        method_name:   One of {gradcam, gradcam_pp, scorecam,
                        layercam, eigencam, xgradcam}.
        model:         Classifier (timm EfficientNet) in eval mode on DEVICE.
        input_tensor:  C×H×W float32 tensor (NOT batched).
        target_size:   (H, W) to resize output; defaults to input spatial size.

    Returns:
        Normalised saliency map float32 [0, 1] of shape `target_size`.
    """
    _METHODS = {
        "gradcam":    GradCAM,
        "gradcam_pp": GradCAMPlusPlus,
        "scorecam":   ScoreCAM,
        "layercam":   LayerCAM,
        "eigencam":   EigenCAM,
        "xgradcam":   XGradCAM,
    }
    if method_name not in _METHODS:
        raise ValueError(f"Unknown CAM method: {method_name}. "
                         f"Choose from {list(_METHODS.keys())}")

    cam_cls      = _METHODS[method_name]
    target_layer = _get_target_layer(model)
    targets      = [BinaryClassifierOutputTarget(0)]   # class 0 = malignant probability

    model.eval()
    inp = input_tensor.unsqueeze(0).to(DEVICE)         # 1×C×H×W

    with cam_cls(model=model, target_layers=target_layer) as cam:
        sal = cam(input_tensor=inp, targets=targets)[0]  # H×W

    sal = _normalise(sal.astype(np.float32))

    if target_size is not None:
        h, w = target_size
        sal = cv2.resize(sal, (w, h), interpolation=cv2.INTER_LINEAR)
        sal = _normalise(sal)

    return sal


# ---------------------------------------------------------------------------
# All-methods runner
# ---------------------------------------------------------------------------

def run_all_cams(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_size: tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    """Run all 6 CAM methods on a single image.

    Returns a dict mapping method name → normalised saliency map.
    Errors in individual methods are caught and logged without crashing
    the benchmark (ScoreCAM is sometimes slow / OOM on small GPUs).
    """
    methods = ["gradcam", "gradcam_pp", "scorecam", "layercam",
               "eigencam", "xgradcam"]
    results = {}
    for name in methods:
        try:
            results[name] = run_cam(name, model, input_tensor, target_size)
        except Exception as exc:
            print(f"[WARN] {name} failed: {exc}")
            h = target_size[0] if target_size else input_tensor.shape[-2]
            w = target_size[1] if target_size else input_tensor.shape[-1]
            results[name] = np.zeros((h, w), dtype=np.float32)
    return results


# ---------------------------------------------------------------------------
# Batch inference helper (for benchmark speed measurement)
# ---------------------------------------------------------------------------

@torch.no_grad()
def classifier_confidence(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
) -> float:
    """Return sigmoid confidence for malignant class."""
    model.eval()
    inp = input_tensor.unsqueeze(0).to(DEVICE)
    with autocast(enabled=AMP):
        logit = model(inp).squeeze()
    return float(torch.sigmoid(logit).cpu())
