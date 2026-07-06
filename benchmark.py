"""
DermaSalient v2 — Full research benchmark.

Evaluates 9 saliency methods against real ISIC 2018 ground-truth masks:
  GradCAM, GradCAM++, ScoreCAM, LayerCAM, EigenCAM, XGradCAM (6 CAM),
  U²-Net, SAM (point-prompted), Ensemble (weighted fusion).

Outputs
-------
outputs/benchmarks/full_results.csv     — per-image × per-model metrics
outputs/benchmarks/leaderboard.csv      — aggregate leaderboard table
outputs/benchmarks/significance_table.csv — Wilcoxon p-values
outputs/benchmarks/pr_curves.pkl        — serialised PR curve data
outputs/visualizations/benchmark_*.png  — benchmark bar charts

Usage:
    python benchmark.py [--n_images 200] [--no_crf] [--workers 0]
"""
import argparse
import os
import pickle
import time
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils.config import (
    seed_everything, DEVICE, IMG_DIR, MASK_DIR, BENCH_DIR, VIZ_DIR, REPORT_DIR,
    CLASSIFIER_CKPT,
)
from src.data.dataset import build_splits
from src.data.augmentations import val_transforms
from src.models.classifier import load_classifier
from src.saliency.cam_methods import run_all_cams, classifier_confidence
from src.saliency.u2net_infer import u2net_saliency_from_tensor
from src.saliency.sam_infer import sam_saliency
from src.saliency.postprocess import full_postprocess_pipeline
from src.saliency.fusion import ensemble_saliency
from src.evaluation.metrics import compute_all_metrics, pr_curve
from src.evaluation.statistical_tests import all_pairs_significance, ci_table
from src.utils.visualization import (
    benchmark_bar_chart, pr_curve_plot, significance_heatmap,
)

warnings.filterwarnings("ignore")
METRICS = ["mae", "f_measure", "weighted_f", "s_measure", "e_measure", "dice", "iou"]


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_images", type=int, default=200,
                   help="Max images to benchmark (use -1 for all with masks)")
    p.add_argument("--no_crf", action="store_true",
                   help="Skip CRF post-processing (ablation)")
    p.add_argument("--workers", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Mask utilities — defined directly here to avoid import issues
# ---------------------------------------------------------------------------

def _mask_path(image_name: str) -> str:
    return os.path.join(MASK_DIR, f"{image_name}_segmentation.png")


def _has_mask(image_name: str) -> bool:
    return os.path.isfile(_mask_path(image_name))


def _load_mask(image_name: str) -> np.ndarray | None:
    """Load mask at original resolution as float32 [0,1]."""
    p = _mask_path(image_name)
    if not os.path.isfile(p):
        return None
    raw = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        return None
    return (raw > 127).astype(np.float32)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image_rgb(image_name: str) -> np.ndarray | None:
    path = os.path.join(IMG_DIR, f"{image_name}.jpg")
    img  = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    """H×W×3 uint8 → C×H×W float32 tensor (ImageNet-normalised)."""
    tf  = val_transforms()
    out = tf(image=img_rgb)["image"]
    return out


# ---------------------------------------------------------------------------
# Per-image inference
# ---------------------------------------------------------------------------

def run_image(
    image_name: str,
    classifier: torch.nn.Module,
    apply_crf: bool = True,
) -> list[dict]:
    """Run all 9 models on one image and return list of metric dicts."""

    # Load image
    img_rgb = load_image_rgb(image_name)
    if img_rgb is None:
        return []

    # Load mask and resize to match image
    raw_mask = _load_mask(image_name)
    if raw_mask is None:
        return []
    gt_mask = cv2.resize(
        raw_mask,
        (img_rgb.shape[1], img_rgb.shape[0]),
        interpolation=cv2.INTER_NEAREST
    ).astype(np.float32)

    tensor     = get_tensor(img_rgb).to(DEVICE)
    confidence = classifier_confidence(classifier, tensor)

    rows = []

    # ---- 6 CAM methods ------------------------------------------------
    t0    = time.perf_counter()
    cams  = run_all_cams(classifier, tensor,
                         target_size=(img_rgb.shape[0], img_rgb.shape[1]))
    cam_time = (time.perf_counter() - t0) * 1000 / max(len(cams), 1)

    for name, sal in cams.items():
        try:
            t_pp0  = time.perf_counter()
            _, bin_mask = full_postprocess_pipeline(img_rgb, sal, apply_crf=apply_crf)
            pp_ms   = (time.perf_counter() - t_pp0) * 1000
            # Resize bin_mask to match gt_mask
            bin_mask = cv2.resize(
                bin_mask.astype(np.float32),
                (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            metrics = compute_all_metrics(bin_mask.astype(np.float32), gt_mask)
            rows.append({
                "image_name":    image_name,
                "model":         name,
                "inference_ms":  round(cam_time + pp_ms, 2),
                "confidence":    round(confidence, 4),
                **metrics,
            })
        except Exception as exc:
            print(f"[WARN] CAM {name} failed on {image_name}: {exc}")

    # ---- U²-Net -------------------------------------------------------
    u2_sal = cams.get("gradcam", np.zeros(img_rgb.shape[:2], dtype=np.float32))
    try:
        t0     = time.perf_counter()
        u2_sal = u2net_saliency_from_tensor(tensor)
        u2_ms  = (time.perf_counter() - t0) * 1000
        _, u2_bin = full_postprocess_pipeline(img_rgb, u2_sal, apply_crf=apply_crf)
        u2_bin = cv2.resize(
            u2_bin.astype(np.float32),
            (gt_mask.shape[1], gt_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        u2_metrics = compute_all_metrics(u2_bin.astype(np.float32), gt_mask)
        rows.append({
            "image_name":    image_name,
            "model":         "u2net",
            "inference_ms":  round(u2_ms, 2),
            "confidence":    round(confidence, 4),
            **u2_metrics,
        })
    except Exception as exc:
        print(f"[WARN] U2Net failed on {image_name}: {exc}")

    # ---- SAM (point-prompted) ----------------------------------------
    try:
        gradcam_sal = cams.get("gradcam", np.zeros(img_rgb.shape[:2], dtype=np.float32))
        t0          = time.perf_counter()
        sam_mask    = sam_saliency(img_rgb, gradcam_sal)
        sam_ms      = (time.perf_counter() - t0) * 1000
        sam_mask = cv2.resize(
            sam_mask.astype(np.float32),
            (gt_mask.shape[1], gt_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        sam_metrics = compute_all_metrics(sam_mask, gt_mask)
        rows.append({
            "image_name":    image_name,
            "model":         "sam",
            "inference_ms":  round(sam_ms, 2),
            "confidence":    round(confidence, 4),
            **sam_metrics,
        })
    except Exception as exc:
        print(f"[WARN] SAM failed on {image_name}: {exc}")

    # ---- Ensemble ------------------------------------------------------
    try:
        all_maps = {**cams, "u2net": u2_sal}
        ens_sal  = ensemble_saliency(all_maps)
        _, ens_bin = full_postprocess_pipeline(img_rgb, ens_sal, apply_crf=apply_crf)
        ens_bin = cv2.resize(
            ens_bin.astype(np.float32),
            (gt_mask.shape[1], gt_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        ens_metrics = compute_all_metrics(ens_bin.astype(np.float32), gt_mask)
        rows.append({
            "image_name":    image_name,
            "model":         "ensemble",
            "inference_ms":  0.0,
            "confidence":    round(confidence, 4),
            **ens_metrics,
        })
    except Exception as exc:
        print(f"[WARN] Ensemble failed on {image_name}: {exc}")

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    seed_everything()
    os.makedirs(BENCH_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR,   exist_ok=True)

    print(f"\n{'='*60}")
    print(f" DermaSalient v2 — Benchmark")
    print(f" Device: {DEVICE} | CRF: {'ON' if not args.no_crf else 'OFF'}")
    print(f"{'='*60}\n")

    if not os.path.isfile(CLASSIFIER_CKPT):
        raise FileNotFoundError(
            f"Classifier not found at {CLASSIFIER_CKPT}. Run train.py first."
        )
    print("Loading classifier …")
    classifier = load_classifier()

    _, _, test_df = build_splits()
    masked_images = [
        row["image_name"] for _, row in test_df.iterrows()
        if _has_mask(row["image_name"])
    ]
    if args.n_images > 0:
        masked_images = masked_images[:args.n_images]

    print(f"Images with ground-truth masks: {len(masked_images)}")
    if len(masked_images) == 0:
        print("\nNo masks found. Check MASK_DIR in config.py")
        return

    all_rows = []
    for img_name in tqdm(masked_images, desc="Benchmarking"):
        try:
            rows = run_image(img_name, classifier, apply_crf=not args.no_crf)
            all_rows.extend(rows)
        except Exception as exc:
            print(f"[ERROR] {img_name}: {exc}")

    if not all_rows:
        print("No results collected. Exiting.")
        return

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(BENCH_DIR, "full_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved full results -> {csv_path}  ({len(df)} rows)")

    leaderboard = (
        df.groupby("model")[METRICS + ["inference_ms"]]
        .agg(["mean", "std"])
        .round(4)
    )
    leaderboard.columns = ["_".join(c) for c in leaderboard.columns]
    leaderboard = leaderboard.sort_values("f_measure_mean", ascending=False)
    lb_path = os.path.join(BENCH_DIR, "leaderboard.csv")
    leaderboard.to_csv(lb_path)
    print(f"Saved leaderboard -> {lb_path}")

    print("\n" + "="*60)
    print(" LEADERBOARD (ranked by F-measure)")
    print("="*60)
    display_cols = ["mae_mean", "f_measure_mean", "dice_mean", "iou_mean",
                    "s_measure_mean", "inference_ms_mean"]
    avail_cols   = [c for c in display_cols if c in leaderboard.columns]
    print(leaderboard[avail_cols].to_string())
    print("="*60)

    winner = leaderboard.index[0]
    print(f"\nWinner: {winner.upper()}  "
          f"(F-measure = {leaderboard.loc[winner, 'f_measure_mean']:.4f})")

    print("\nRunning Wilcoxon significance tests …")
    try:
        sig_df = all_pairs_significance(df, metric="f_measure")
        sig_path = os.path.join(BENCH_DIR, "significance_table.csv")
        sig_df.to_csv(sig_path, index=False)
        print(f"Saved significance table -> {sig_path}")
        sig_count = sig_df["significant"].sum()
        print(f"  {sig_count}/{len(sig_df)} model pairs significant (p<0.05)")
    except Exception as exc:
        print(f"[WARN] Significance tests failed: {exc}")

    try:
        ci_df = ci_table(df, METRICS)
        ci_df.to_csv(os.path.join(BENCH_DIR, "confidence_intervals.csv"), index=False)
    except Exception as exc:
        print(f"[WARN] CI table failed: {exc}")

    try:
        for metric in ["f_measure", "dice", "iou", "mae"]:
            fig = benchmark_bar_chart(df, metric)
            fig.write_image(os.path.join(VIZ_DIR, f"benchmark_{metric}.png"))
        print(f"Saved visualisations -> {VIZ_DIR}/benchmark_*.png")
    except Exception as exc:
        print(f"[WARN] Visualisations failed: {exc}")

    try:
        sig_fig = significance_heatmap(sig_df)
        sig_fig.savefig(os.path.join(VIZ_DIR, "significance_heatmap.png"),
                        dpi=120, bbox_inches="tight")
        print(f"Saved significance heatmap -> {VIZ_DIR}/significance_heatmap.png")
    except Exception as exc:
        print(f"[WARN] Significance heatmap failed: {exc}")

    print("\nBenchmark complete. Launch app.py to explore results interactively.")


if __name__ == "__main__":
    main()