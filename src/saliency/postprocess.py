"""
Saliency map post-processing pipeline.

Three-stage pipeline applied in order:
  1. Dense CRF  — makes saliency contours snap to real image edges
  2. Adaptive threshold  — Otsu on the CRF output
  3. Morphological cleaning  — remove noise blobs, close holes

Dense CRF is standard post-processing in salient object detection (SOD) papers:
  Lafferty, J., McCallum, A., & Pereira, F. (2001). Conditional Random Fields.
  Applied to vision:
  Krahenbuhl, P. & Koltun, V. (2011). Efficient Inference in Fully Connected
  CRFs with Gaussian Edge Potentials. NeurIPS 2011.

pydensecrf (Python wrapper): https://github.com/lucasb-eyer/pydensecrf
"""
import numpy as np
import cv2
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_closing, disk

from src.utils.config import CRF_ITER


# ---------------------------------------------------------------------------
# Dense CRF
# ---------------------------------------------------------------------------

def crf_refine(img_rgb: np.ndarray, sal_map: np.ndarray,
               n_iter: int = CRF_ITER) -> np.ndarray:
    """Refine a soft saliency map with a dense CRF.

    The CRF uses two pairwise potentials:
      • Gaussian (spatial smoothness) — nearby pixels should have similar labels
      • Bilateral (appearance) — nearby pixels with similar colour → same label

    This produces boundaries that respect image edges without requiring a
    separate edge detector.

    Args:
        img_rgb:  H×W×3 uint8 image.
        sal_map:  H×W float32 saliency map in [0, 1].
        n_iter:   Number of CRF mean-field iterations.

    Returns:
        Refined soft saliency map float32 [0, 1].
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError:
        # Graceful degradation: return the unrefined map if pydensecrf is missing
        print("[WARN] pydensecrf not installed; skipping CRF refinement.")
        return sal_map.copy()

    h, w = img_rgb.shape[:2]

    # pydensecrf's addPairwiseBilateral raises "Bad shape for pairwise
    # bilateral" on some non-square resolutions — run CRF on a fixed
    # 512x512 square and resize the result back to the original size.
    img_sq = cv2.resize(img_rgb, (512, 512))
    sal_sq = cv2.resize(sal_map.astype(np.float32), (512, 512))

    # Build 2-class probability map [background, foreground]
    probs = np.stack([1.0 - sal_sq, sal_sq], axis=0).astype(np.float32)
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)

    d = dcrf.DenseCRF2D(512, 512, 2)

    # Unary potential from the saliency network output
    U = unary_from_softmax(probs)
    d.setUnaryEnergy(U)

    # Pairwise Gaussian (spatial smoothness)
    d.addPairwiseGaussian(sxy=3, compat=3)

    # Pairwise Bilateral (appearance-aware)
    rgb_c = np.ascontiguousarray(img_sq.astype(np.uint8))
    d.addPairwiseBilateral(sxy=50, srgb=13, rgbim=rgb_c, compat=10)

    # Mean-field inference
    Q = d.inference(n_iter)
    refined_sq = np.array(Q)[1].reshape(512, 512).astype(np.float32)
    refined = cv2.resize(refined_sq, (w, h))
    return refined


# ---------------------------------------------------------------------------
# Adaptive threshold
# ---------------------------------------------------------------------------

def adaptive_threshold(sal_map: np.ndarray) -> np.ndarray:
    """Binarise the saliency map using Otsu's method on the map itself.

    Using Otsu on the SALIENCY MAP (not the image) avoids the assumption that
    the lesion is the darkest/brightest region — a common failure mode of
    image-domain Otsu thresholding on dermoscopy images.
    """
    try:
        thr = threshold_otsu(sal_map)
    except Exception:
        thr = 0.5
    return (sal_map >= thr).astype(np.uint8)


# ---------------------------------------------------------------------------
# Morphological cleaning
# ---------------------------------------------------------------------------

def morphological_clean(binary_mask: np.ndarray,
                        min_size: int = 500,
                        disk_radius: int = 5) -> np.ndarray:
    """Remove small noise blobs and close holes in the binary mask.

    Args:
        binary_mask:  H×W uint8 or bool mask.
        min_size:     Minimum connected component size in pixels.
        disk_radius:  Structuring element radius for binary closing.

    Returns:
        Cleaned binary mask uint8.
    """
    arr    = binary_mask.astype(bool)
    cleaned = remove_small_objects(arr, min_size=min_size, connectivity=2)
    closed  = binary_closing(cleaned, disk(disk_radius))
    return closed.astype(np.uint8)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def full_postprocess_pipeline(
    img_rgb: np.ndarray,
    sal_map: np.ndarray,
    apply_crf: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """CRF refinement → adaptive Otsu threshold → morphological cleaning.

    Args:
        img_rgb:    H×W×3 uint8 image.
        sal_map:    H×W float32 raw saliency map in [0, 1].
        apply_crf:  Set False to skip CRF (useful for ablation study).

    Returns:
        (refined_soft, binary_clean)
          • refined_soft: float32 [0, 1] after CRF
          • binary_clean: uint8 {0, 1} after thresholding + morphology
    """
    h, w = img_rgb.shape[:2]

    # Standardise to a 512x512 square before running the pipeline — avoids
    # pydensecrf's "Bad shape for pairwise bilateral" error on non-square
    # images — then resize the outputs back to the original resolution.
    img_sq = cv2.resize(img_rgb, (512, 512))
    sal_sq = cv2.resize(sal_map.astype(np.float32), (512, 512))

    refined_sq = crf_refine(img_sq, sal_sq) if apply_crf else sal_sq.copy()
    binary_sq  = adaptive_threshold(refined_sq)
    clean_sq   = morphological_clean(binary_sq)

    refined = cv2.resize(refined_sq, (w, h))
    clean   = cv2.resize(clean_sq, (w, h), interpolation=cv2.INTER_NEAREST)
    return refined.astype(np.float32), clean


def postprocess_no_crf(sal_map: np.ndarray) -> np.ndarray:
    """Threshold + morphological cleaning only (no CRF).  For ablation."""
    binary = adaptive_threshold(sal_map)
    return morphological_clean(binary)
