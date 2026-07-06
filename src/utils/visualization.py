"""
Shared visualisation utilities for saliency maps, overlays, and benchmark plots.
"""
import os
import io

import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

from src.utils.config import VIZ_DIR

_CMAP = "jet"


# ---------------------------------------------------------------------------
# Core overlay helpers
# ---------------------------------------------------------------------------

def overlay_saliency(img_rgb: np.ndarray, sal_map: np.ndarray,
                     alpha: float = 0.5) -> np.ndarray:
    """Return BGR image with jet-colormap saliency overlaid."""
    h, w = img_rgb.shape[:2]
    if sal_map.shape != (h, w):
        sal_map = cv2.resize(sal_map, (w, h), interpolation=cv2.INTER_LINEAR)
    sal_u8 = (sal_map * 255).clip(0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(sal_u8, cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def draw_bbox_from_mask(img_rgb: np.ndarray, binary_mask: np.ndarray,
                        colour=(0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Draw the bounding box of the mask on the image."""
    img_copy = img_rgb.copy()
    ys, xs   = np.where(binary_mask > 0)
    if len(xs) == 0:
        return img_copy
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()), int(ys.max())
    cv2.rectangle(img_copy, (x0, y0), (x1, y1), colour, thickness)
    return img_copy


def describe_saliency_region(sal_map: np.ndarray) -> str:
    """Human-readable description of the peak saliency location."""
    h, w   = sal_map.shape
    idx    = sal_map.argmax()
    py, px = divmod(idx, w)
    vert   = "upper" if py < h // 3 else ("lower" if py > 2 * h // 3 else "central")
    horiz  = "left"  if px < w // 3 else ("right" if px > 2 * w // 3 else "centre")
    return f"High saliency in {vert}-{horiz} of image (peak at [{px}, {py}])."


# ---------------------------------------------------------------------------
# Gallery plots
# ---------------------------------------------------------------------------

def saliency_gallery(img_rgb: np.ndarray, saliency_maps: dict,
                     binary_maps: dict | None = None,
                     gt_mask: np.ndarray | None = None,
                     save_path: str | None = None) -> plt.Figure:
    """Matplotlib figure: one column per method showing overlay + binary mask."""
    methods = list(saliency_maps.keys())
    n_cols  = len(methods)
    n_rows  = 3 if binary_maps is not None else 2

    fig, axes = plt.subplots(n_rows, n_cols + 1,
                             figsize=((n_cols + 1) * 3, n_rows * 3))
    fig.suptitle("DermaSalient v2 — Saliency Gallery", fontsize=14)

    # Column 0: original image (and GT mask if available)
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original", fontsize=9)
    if gt_mask is not None:
        axes[1, 0].imshow(gt_mask, cmap="gray")
        axes[1, 0].set_title("GT Mask", fontsize=9)
    else:
        axes[1, 0].axis("off")
    if n_rows == 3:
        axes[2, 0].axis("off")

    for j, name in enumerate(methods, start=1):
        sal = saliency_maps[name]
        ov  = overlay_saliency(img_rgb, sal)
        axes[0, j].imshow(ov)
        axes[0, j].set_title(name, fontsize=8)
        axes[1, j].imshow(sal, cmap=_CMAP, vmin=0, vmax=1)
        axes[1, j].set_title("Soft", fontsize=8)
        if binary_maps is not None and n_rows == 3:
            axes[2, j].imshow(binary_maps.get(name, np.zeros_like(sal)),
                              cmap="gray", vmin=0, vmax=1)
            axes[2, j].set_title("Binary", fontsize=8)

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Benchmark plots (Plotly for interactive use in Gradio)
# ---------------------------------------------------------------------------

def benchmark_bar_chart(df, metric: str = "f_measure") -> go.Figure:
    """Interactive Plotly bar chart — models ranked by metric."""
    summary = (
        df.groupby("model")[metric]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    fig = go.Figure(go.Bar(
        x=summary["model"],
        y=summary["mean"],
        error_y=dict(type="data", array=summary["std"].tolist()),
        marker_color=px.colors.qualitative.Set2[:len(summary)],
    ))
    fig.update_layout(
        title=f"Benchmark Results — {metric}",
        xaxis_title="Model",
        yaxis_title=metric,
        template="plotly_white",
    )
    return fig


def pr_curve_plot(pr_data: dict) -> go.Figure:
    """Plotly PR-curve overlay for all models."""
    fig = go.Figure()
    colours = px.colors.qualitative.Set1
    for i, (name, (precisions, recalls, auc_pr)) in enumerate(pr_data.items()):
        fig.add_trace(go.Scatter(
            x=recalls, y=precisions,
            mode="lines", name=f"{name} (AUC={auc_pr:.3f})",
            line=dict(color=colours[i % len(colours)]),
        ))
    fig.update_layout(
        title="Precision-Recall Curves",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_white",
    )
    return fig


def significance_heatmap(sig_df: pd.DataFrame) -> plt.Figure:
    """Seaborn heatmap of p-values for all model pairs."""
    import pandas as pd
    models  = sorted(set(sig_df["model_a"]) | set(sig_df["model_b"]))
    matrix  = pd.DataFrame(np.ones((len(models), len(models))),
                           index=models, columns=models)
    for _, row in sig_df.iterrows():
        matrix.loc[row["model_a"], row["model_b"]] = row["p_value"]
        matrix.loc[row["model_b"], row["model_a"]] = row["p_value"]
    np.fill_diagonal(matrix.values, 1.0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix.astype(float), annot=True, fmt=".3f", cmap="RdYlGn_r",
                vmin=0, vmax=0.1, ax=ax,
                cbar_kws={"label": "p-value (Wilcoxon)"})
    ax.set_title("Statistical Significance Matrix (p < 0.05 = significant)")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Training curve
# ---------------------------------------------------------------------------

def plot_training_history(history_df) -> plt.Figure:
    """Loss + AUC training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history_df["epoch"], history_df["tr_loss"], label="Train")
    ax1.plot(history_df["epoch"], history_df["va_loss"], label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend()
    ax1.set_title("Loss Curves")

    ax2.plot(history_df["epoch"], history_df["tr_auc"], label="Train")
    ax2.plot(history_df["epoch"], history_df["va_auc"], label="Val")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("AUC-ROC"); ax2.legend()
    ax2.set_title("AUC-ROC Curves")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# PIL ↔ numpy helpers (for Gradio)
# ---------------------------------------------------------------------------

def pil_to_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def rgb_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def fig_to_pil(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf).copy()
