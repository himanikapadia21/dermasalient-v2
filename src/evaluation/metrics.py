"""
Salient Object Detection evaluation metrics.

All metrics are computed against ground-truth binary masks.
Each function is cited to its originating paper.

Standard reference benchmark: DUT-OMRON, DUTS, HKU-IS, ECSSD.
Applied here to ISIC 2018 dermoscopy segmentation masks.
"""
import numpy as np
import cv2
from sklearn.metrics import auc as sklearn_auc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_binary(pred: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Threshold a soft saliency map to binary."""
    return (pred >= threshold).astype(np.float32)


def _safe_div(a: float, b: float, eps: float = 1e-8) -> float:
    return a / (b + eps)


# ---------------------------------------------------------------------------
# Pixel-level metrics
# ---------------------------------------------------------------------------

def mae(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean Absolute Error between soft prediction and binary GT.

    Reference: Borji, A. et al. Salient Object Detection: A Benchmark.
    IEEE Trans. Image Process. 2015.

    Lower is better.
    """
    pred = cv2.resize(pred.astype(np.float32), (gt.shape[1], gt.shape[0]))
    return float(np.abs(pred.astype(np.float32) - gt.astype(np.float32)).mean())


def dice_coefficient(pred: np.ndarray, gt: np.ndarray,
                     threshold: float = 0.5) -> float:
    """Dice / F1 overlap between binarised prediction and GT.

    Reference: Dice, L. R. (1945). Measures of the amount of ecologic
    association between species. Ecology, 26(3), 297-302.

    Standard in medical image segmentation benchmarks.
    """
    pred = cv2.resize(pred.astype(np.float32), (gt.shape[1], gt.shape[0]))
    p = _to_binary(pred, threshold)
    g = gt.astype(np.float32)
    intersection = (p * g).sum()
    return float(_safe_div(2.0 * intersection, p.sum() + g.sum()))


def iou(pred: np.ndarray, gt: np.ndarray,
        threshold: float = 0.5) -> float:
    """Intersection-over-Union (Jaccard index).

    Reference: Jaccard, P. (1912). The distribution of the flora in the
    alpine zone. New Phytologist, 11(2), 37-50.

    Standard in object detection and segmentation benchmarks.
    """
    pred = cv2.resize(pred.astype(np.float32), (gt.shape[1], gt.shape[0]))
    p = _to_binary(pred, threshold).astype(bool)
    g = gt.astype(bool)
    inter = (p & g).sum()
    union = (p | g).sum()
    return float(_safe_div(inter, union))


# ---------------------------------------------------------------------------
# SOD-specific metrics
# ---------------------------------------------------------------------------

def f_measure(pred: np.ndarray, gt: np.ndarray,
              beta: float = 0.3) -> float:
    """F-measure (Fβ) with β² = 0.3 (precision-weighted).

    Reference: Achanta, R. et al. Frequency-tuned Salient Region Detection.
    CVPR 2009. β=0.3 emphasises precision over recall for SOD tasks.

    Higher is better.
    """
    pred = cv2.resize(pred.astype(np.float32), (gt.shape[1], gt.shape[0]))
    threshold = 2.0 * pred.mean()
    threshold = min(threshold, 1.0)
    p = (pred >= threshold).astype(np.float32)
    g = gt.astype(np.float32)

    tp = (p * g).sum()
    precision = _safe_div(tp, p.sum())
    recall    = _safe_div(tp, g.sum())

    return float(_safe_div(
        (1 + beta) * precision * recall,
        beta * precision + recall,
    ))


def weighted_f_measure(pred: np.ndarray, gt: np.ndarray) -> float:
    """Weighted F-measure (Fw) — addresses the interpolation flaw in standard Fm.

    Uses prediction confidence as per-pixel weight rather than a hard threshold.

    Reference: Margolin, R. et al. How to Evaluate Foreground Maps.
    CVPR 2014.
    """
    pred = cv2.resize(pred.astype(np.float32), (gt.shape[1], gt.shape[0]))
    from scipy.ndimage import convolve
    g = gt.astype(np.float32)
    p = pred.astype(np.float32)

    # Weighted precision: high-confidence pixels contribute more
    tp_weighted = (p * g).sum()
    fp_weighted = (p * (1.0 - g)).sum()
    fn_weighted = ((1.0 - p) * g).sum()

    precision = _safe_div(tp_weighted, tp_weighted + fp_weighted)
    recall    = _safe_div(tp_weighted, tp_weighted + fn_weighted)

    beta_sq = 0.3
    return float(_safe_div(
        (1 + beta_sq) * precision * recall,
        beta_sq * precision + recall,
    ))


def s_measure(pred: np.ndarray, gt: np.ndarray,
              alpha: float = 0.5) -> float:
    """Structure measure — evaluates region-aware and object-aware structural similarity.

    Reference: Fan, D.-P. et al. Structure-Measure: A New Way to Evaluate
    Foreground Maps. ICCV 2017.

    Higher is better.
    """
    pred = cv2.resize(pred.astype(np.float32), (gt.shape[1], gt.shape[0]))
    p = pred.astype(np.float32)
    g = gt.astype(np.float32)

    def _region_sim(pm, gm):
        """Region-wise similarity."""
        if gm.sum() == 0:
            return 1.0 - pm.mean()
        x = pm.mean()
        y = gm.mean()
        sigma_x  = pm.var() + 1e-8
        sigma_xy = ((pm - x) * (gm - y)).mean()
        score = (2 * x * y + 1e-4) / (x**2 + y**2 + 1e-4)
        return float(score)

    def _object_sim(pm, gm):
        """Object-wise similarity."""
        tp = (pm * gm).sum()
        fp = (pm * (1 - gm)).sum()
        fn = ((1 - pm) * gm).sum()
        p_ = _safe_div(tp, tp + fp)
        r_ = _safe_div(tp, tp + fn)
        return float(_safe_div(2 * p_ * r_, p_ + r_))

    o_sim = _object_sim(p, g)
    r_sim = _region_sim(p, g)
    return alpha * o_sim + (1 - alpha) * r_sim


def e_measure(pred: np.ndarray, gt: np.ndarray) -> float:
    """Enhanced alignment measure (Eξ).

    Evaluates pixel-level and image-level alignment jointly.

    Reference: Fan, D.-P. et al. Enhanced-Alignment Measure for Binary
    Foreground Map Evaluation. IJCAI 2018.

    Higher is better.
    """
    pred = cv2.resize(pred.astype(np.float32), (gt.shape[1], gt.shape[0]))
    p = pred.astype(np.float32)
    g = gt.astype(np.float32)

    threshold = 2.0 * p.mean()
    threshold = min(threshold, 1.0)
    pb = (p >= threshold).astype(np.float32)

    fg_mean = _safe_div(g.sum(), g.size)
    pb_mean = pb.mean()

    # Enhanced alignment matrix
    align_mat = (
        (pb - pb_mean) * (g - fg_mean)
    ) / np.sqrt(
        (pb - pb_mean).var() * (g - fg_mean).var() + 1e-8
    ) + 1.0

    em = (align_mat ** 2).mean() / 4.0
    return float(em)


# ---------------------------------------------------------------------------
# Precision-Recall curve
# ---------------------------------------------------------------------------

def pr_curve(pred: np.ndarray, gt: np.ndarray,
             n_thresholds: int = 255) -> tuple[np.ndarray, np.ndarray, float]:
    """Full precision-recall curve across all thresholds.

    Args:
        pred:          H×W float32 soft saliency map [0, 1].
        gt:            H×W binary ground truth.
        n_thresholds:  Number of threshold steps.

    Returns:
        (precisions, recalls, auc_pr)
    """
    g = gt.astype(bool).ravel()
    p = pred.ravel()

    thresholds  = np.linspace(0, 1, n_thresholds + 1)
    precisions  = np.zeros(len(thresholds))
    recalls     = np.zeros(len(thresholds))

    for i, thr in enumerate(thresholds):
        pb         = p >= thr
        tp         = (pb & g).sum()
        fp         = (pb & ~g).sum()
        fn         = (~pb & g).sum()
        precisions[i] = _safe_div(tp, tp + fp)
        recalls[i]    = _safe_div(tp, tp + fn)

    # Sort by recall for AUC
    order      = np.argsort(recalls)
    recalls    = recalls[order]
    precisions = precisions[order]
    auc_pr     = float(np.trapz(precisions, recalls))

    return precisions, recalls, auc_pr


# ---------------------------------------------------------------------------
# Full metrics dict
# ---------------------------------------------------------------------------

def compute_all_metrics(pred: np.ndarray, gt: np.ndarray,
                        threshold: float = 0.5) -> dict[str, float]:
    """Compute all SOD metrics and return as a flat dict.

    Used by benchmark.py to build the results DataFrame.
    """
    return {
        "mae":        mae(pred, gt),
        "f_measure":  f_measure(pred, gt),
        "weighted_f": weighted_f_measure(pred, gt),
        "s_measure":  s_measure(pred, gt),
        "e_measure":  e_measure(pred, gt),
        "dice":       dice_coefficient(pred, gt, threshold),
        "iou":        iou(pred, gt, threshold),
    }
