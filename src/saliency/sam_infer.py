"""
SAM (Segment Anything Model) — point-prompted inference.

We use SAM in POINT-PROMPTED mode, NOT automatic mode.
Automatic mode is slow and produces masks for every object in the scene.
Point-prompted mode is fast, deterministic, and clinically meaningful:

  GradCAM finds WHERE the network looks → that peak point is the prompt →
  SAM draws precise anatomical boundaries around exactly that lesion.

This is the correct research approach:
  Kirillov, A. et al. Segment Anything. ICCV 2023.
  arXiv: https://arxiv.org/abs/2304.02643
"""
import os
import sys
import urllib.request

import numpy as np
import torch

from src.utils.config import SAM_CKPT, SAM_MODEL_TYPE, DEVICE, WEIGHTS_DIR

_SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"


# ---------------------------------------------------------------------------
# Weight download
# ---------------------------------------------------------------------------

def download_sam_weights() -> None:
    """Download SAM ViT-B weights (~375 MB) if not already present."""
    if os.path.isfile(SAM_CKPT):
        print(f"SAM weights already present: {SAM_CKPT}")
        return
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    print(f"Downloading SAM ViT-B weights → {SAM_CKPT} …")
    urllib.request.urlretrieve(_SAM_URL, SAM_CKPT,
                               reporthook=_progress_hook)
    print("\nDownload complete.")


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 // total_size) if total_size > 0 else 0
    print(f"\r  {pct:3d}%  {downloaded // 1_048_576} MB", end="", flush=True)


# ---------------------------------------------------------------------------
# Predictor loader (singleton)
# ---------------------------------------------------------------------------

_sam_predictor = None


def load_sam_predictor(force_reload: bool = False):
    """Return a cached SamPredictor.  Downloads weights if missing."""
    global _sam_predictor
    if _sam_predictor is not None and not force_reload:
        return _sam_predictor

    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError:
        raise ImportError(
            "Install segment-anything:  "
            "pip install git+https://github.com/facebookresearch/segment-anything.git"
        )

    download_sam_weights()
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CKPT)
    sam.to(DEVICE).eval()
    _sam_predictor = SamPredictor(sam)
    print("SAM loaded and ready.")
    return _sam_predictor


# ---------------------------------------------------------------------------
# Point-prompted inference
# ---------------------------------------------------------------------------

def sam_point_prompted(
    img_rgb: np.ndarray,
    gradcam_sal: np.ndarray,
    n_top_points: int = 1,
) -> np.ndarray:
    """Segment the lesion using the GradCAM peak as a SAM prompt point.

    The GradCAM saliency map identifies the discriminative region (WHY the
    classifier fires); SAM then delineates precise boundaries around it.

    Args:
        img_rgb:       H×W×3 uint8 image.
        gradcam_sal:   H×W float32 saliency map from GradCAM.
        n_top_points:  Number of foreground prompt points.  1 is usually enough;
                       more points are useful if the GradCAM map is diffuse.

    Returns:
        Binary mask float32 [0, 1] of shape H×W.
    """
    predictor = load_sam_predictor()
    predictor.set_image(img_rgb)

    h, w = gradcam_sal.shape

    # Collect top-N saliency peaks (separated to avoid clustering)
    input_points = []
    input_labels = []
    used         = np.zeros_like(gradcam_sal, dtype=bool)
    exclusion_r  = max(h, w) // 8     # minimum pixel distance between points

    for _ in range(n_top_points):
        masked = gradcam_sal.copy()
        masked[used] = -1.0
        idx   = int(masked.argmax())
        py, px = divmod(idx, w)

        # Mark surrounding region as used
        yy, xx = np.ogrid[:h, :w]
        used |= ((yy - py) ** 2 + (xx - px) ** 2) <= exclusion_r ** 2

        input_points.append([px, py])
        input_labels.append(1)          # 1 = foreground

    input_points = np.array(input_points)
    input_labels = np.array(input_labels)

    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,           # SAM returns 3 candidate masks
    )
    best_mask = masks[int(scores.argmax())]   # select highest IoU score

    return best_mask.astype(np.float32)


def sam_saliency(
    img_rgb: np.ndarray,
    gradcam_sal: np.ndarray,
) -> np.ndarray:
    """Public API: returns a soft saliency map (same as binary mask for SAM).

    The mask is already binary (0/1) and is returned as-is so that the
    benchmark pipeline treats all method outputs uniformly.
    """
    try:
        mask = sam_point_prompted(img_rgb, gradcam_sal)
        return mask.astype(np.float32)
    except Exception as exc:
        print(f"[WARN] SAM inference failed: {exc}")
        return np.zeros(img_rgb.shape[:2], dtype=np.float32)
