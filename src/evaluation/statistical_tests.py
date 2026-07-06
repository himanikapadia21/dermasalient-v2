"""
Statistical significance testing for saliency benchmark results.

Required before claiming one model is "better" than another in a paper.
Two-sided Wilcoxon signed-rank test is the standard non-parametric test
for paired observations (per-image metric differences).

Reference: Wilcoxon, F. (1945). Individual Comparisons by Ranking Methods.
Biometrics Bulletin, 1(6), 80–83.

Bootstrap confidence intervals follow DiCiccio & Efron (1996).
"""
import itertools

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Wilcoxon test (pairwise)
# ---------------------------------------------------------------------------

def wilcoxon_significance(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Two-sided Wilcoxon signed-rank test between two models.

    Args:
        scores_a: Per-image metric values for model A.
        scores_b: Per-image metric values for model B.
        alpha:    Significance level (default 0.05).

    Returns:
        Dict with statistic, p_value, significant (bool), and effect_size.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Both score arrays must have the same length.")

    # Drop ties (identical scores contribute no information)
    diffs = np.array(scores_a) - np.array(scores_b)
    nonzero = diffs[diffs != 0]

    if len(nonzero) < 10:
        return {
            "statistic":  float("nan"),
            "p_value":    float("nan"),
            "significant": False,
            "effect_size": 0.0,
            "note": f"Only {len(nonzero)} non-tied pairs; test unreliable.",
        }

    stat, p = stats.wilcoxon(scores_a, scores_b, alternative="two-sided")

    # Rank-biserial correlation as effect size (ranges −1 to +1)
    n   = len(nonzero)
    r   = 1.0 - (2.0 * stat) / (n * (n + 1) / 2.0)

    return {
        "statistic":   float(stat),
        "p_value":     float(p),
        "significant": bool(p < alpha),
        "effect_size": float(r),
        "n_pairs":     int(len(scores_a)),
    }


# ---------------------------------------------------------------------------
# All-pairs significance table
# ---------------------------------------------------------------------------

def all_pairs_significance(
    results_df: pd.DataFrame,
    metric: str = "f_measure",
    model_col: str = "model",
    image_col: str = "image_name",
) -> pd.DataFrame:
    """Run Wilcoxon tests for all ordered model pairs on a given metric.

    Args:
        results_df:  Full benchmark DataFrame (one row per image-model combo).
        metric:      Column to test on.
        model_col:   Column name for model identifier.
        image_col:   Column name for per-image identifier.

    Returns:
        DataFrame with columns [model_a, model_b, p_value, significant,
                                 effect_size, a_mean, b_mean, winner].
    """
    models = sorted(results_df[model_col].unique())
    rows   = []

    # Pivot to get per-image scores per model
    pivot = (
        results_df[[image_col, model_col, metric]]
        .pivot_table(index=image_col, columns=model_col, values=metric)
        .dropna()
    )

    for a, b in itertools.combinations(models, 2):
        if a not in pivot.columns or b not in pivot.columns:
            continue
        sa = pivot[a].values
        sb = pivot[b].values
        res = wilcoxon_significance(sa, sb)
        rows.append({
            "model_a":     a,
            "model_b":     b,
            "a_mean":      float(sa.mean()),
            "b_mean":      float(sb.mean()),
            "p_value":     res["p_value"],
            "significant": res["significant"],
            "effect_size": res["effect_size"],
            "winner":      a if sa.mean() > sb.mean() else b,
        })

    return pd.DataFrame(rows).sort_values("p_value")


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def compute_confidence_intervals(
    scores: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict:
    """Bootstrap percentile confidence interval for a metric array.

    Args:
        scores:      1-D array of per-image metric values.
        confidence:  Coverage probability (e.g., 0.95 for 95% CI).
        n_bootstrap: Number of bootstrap resamples.

    Returns:
        Dict with mean, lower, upper, std.
    """
    rng   = np.random.default_rng(seed)
    n     = len(scores)
    boots = rng.choice(scores, size=(n_bootstrap, n), replace=True).mean(axis=1)

    alpha = 1.0 - confidence
    lo    = float(np.percentile(boots, 100 * alpha / 2))
    hi    = float(np.percentile(boots, 100 * (1 - alpha / 2)))

    return {
        "mean":  float(scores.mean()),
        "lower": lo,
        "upper": hi,
        "std":   float(scores.std()),
        "n":     n,
    }


def ci_table(
    results_df: pd.DataFrame,
    metrics: list[str],
    model_col: str = "model",
) -> pd.DataFrame:
    """Bootstrap CIs for every model × metric combination.

    Returns a DataFrame suitable for publication-style tables.
    """
    rows = []
    for model, grp in results_df.groupby(model_col):
        for m in metrics:
            if m not in grp.columns:
                continue
            ci = compute_confidence_intervals(grp[m].dropna().values)
            rows.append({
                "model":  model,
                "metric": m,
                "mean":   ci["mean"],
                "lower":  ci["lower"],
                "upper":  ci["upper"],
                "std":    ci["std"],
                "n":      ci["n"],
            })
    return pd.DataFrame(rows)
