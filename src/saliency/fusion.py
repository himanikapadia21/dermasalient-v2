"""
Saliency map ensemble fusion.

Combines the outputs of multiple saliency methods via weighted averaging.
Equal weights are the default (no prior on method quality); weights can be
updated after benchmarking to reflect observed performance.

This is the ninth "model" in the benchmark — always included.
"""
import numpy as np
from src.utils.config import ENSEMBLE_WEIGHTS


def normalise(sal: np.ndarray) -> np.ndarray:
    lo, hi = sal.min(), sal.max()
    if hi - lo < 1e-8:
        return np.zeros_like(sal, dtype=np.float32)
    return ((sal - lo) / (hi - lo)).astype(np.float32)


def ensemble_saliency(
    maps: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Weighted average of all available saliency maps.

    Args:
        maps:    Dict mapping method name → H×W float32 saliency map [0, 1].
        weights: Dict mapping method name → scalar weight.
                 Methods missing from weights use weight = 1.0.
                 If None, uses ENSEMBLE_WEIGHTS from config (defaults to equal).

    Returns:
        Normalised fused saliency map float32 [0, 1].
    """
    if weights is None:
        weights = ENSEMBLE_WEIGHTS

    if not maps:
        raise ValueError("No saliency maps provided to ensemble_saliency.")

    # Align shapes — all maps should be the same size but be defensive
    shapes = [m.shape for m in maps.values()]
    if len(set(shapes)) > 1:
        # Upsample all to the largest resolution
        import cv2
        max_h = max(s[0] for s in shapes)
        max_w = max(s[1] for s in shapes)
        maps  = {
            k: cv2.resize(v, (max_w, max_h), interpolation=cv2.INTER_LINEAR)
            for k, v in maps.items()
        }

    accum       = np.zeros_like(next(iter(maps.values())), dtype=np.float64)
    total_weight = 0.0

    for name, sal in maps.items():
        w = weights.get(name, 1.0)
        accum        += w * normalise(sal).astype(np.float64)
        total_weight += w

    fused = (accum / total_weight).astype(np.float32)
    return normalise(fused)


def learned_ensemble(
    maps: dict[str, np.ndarray],
    method_scores: dict[str, float],
) -> np.ndarray:
    """Fusion with weights derived from per-method benchmark performance.

    Args:
        maps:          Dict of method → saliency map.
        method_scores: Dict of method → scalar quality score (e.g., F-measure).
                       Higher score → higher weight.

    Returns:
        Fused saliency map.
    """
    # Softmax-style normalisation to keep weights positive and summing to 1
    import math
    scores = {k: method_scores.get(k, 0.5) for k in maps}
    max_s  = max(scores.values())
    exp_w  = {k: math.exp(v - max_s) for k, v in scores.items()}
    total  = sum(exp_w.values())
    weights = {k: v / total for k, v in exp_w.items()}
    return ensemble_saliency(maps, weights)
