"""
DermaSalient v2 — EfficientNet-B4 classifier training entry point.

Usage:
    python train.py [--backbone efficientnet_b4] [--epochs 30] [--batch 16]

All paths and hyperparameters default to src/utils/config.py.
"""
import argparse
import os
import subprocess
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.config import (
    seed_everything, BACKBONE, EPOCHS, BATCH_SIZE, DEVICE,
    REPORT_DIR, VIZ_DIR, CLASSIFIER_CKPT,
)
from src.data.dataset import build_dataloaders
from src.models.classifier import build_classifier, train, evaluate, LabelSmoothBCELoss
from src.utils.visualization import plot_training_history
import torch


def parse_args():
    p = argparse.ArgumentParser(description="Train DermaSalient classifier")
    p.add_argument("--backbone", default=BACKBONE)
    p.add_argument("--epochs",   type=int, default=EPOCHS)
    p.add_argument("--batch",    type=int, default=BATCH_SIZE)
    p.add_argument("--workers",  type=int, default=4)
    return p.parse_args()


def save_environment():
    """Save exact package versions for full reproducibility."""
    out = os.path.join(REPORT_DIR, "environment.txt")
    result = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True)
    with open(out, "w") as fh:
        fh.write(result.stdout)
    print(f"Saved environment snapshot → {out}")


def main():
    args = parse_args()
    seed_everything()

    print(f"\n{'='*60}")
    print(f" DermaSalient v2 — Training")
    print(f" Device: {DEVICE} | Backbone: {args.backbone}")
    print(f" Epochs: {args.epochs} | Batch: {args.batch}")
    print(f"{'='*60}\n")

    # Save environment for reproducibility
    save_environment()

    # Data
    print("Building data loaders …")
    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size=args.batch,
        num_workers=args.workers,
    )
    print(f"  Train batches: {len(train_loader)} | "
          f"Val batches: {len(val_loader)} | "
          f"Test batches: {len(test_loader)}")

    # Model
    print(f"\nBuilding {args.backbone} classifier …")
    model = build_classifier(backbone=args.backbone, pretrained=True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # Train
    print("\nStarting training …\n")
    history = train(model, train_loader, val_loader, epochs=args.epochs)

    # Plot training curves
    hist_df = pd.DataFrame(history)
    fig     = plot_training_history(hist_df)
    os.makedirs(VIZ_DIR, exist_ok=True)
    fig.savefig(os.path.join(VIZ_DIR, "training_curves.png"),
                dpi=120, bbox_inches="tight")
    print(f"Saved training curves → {VIZ_DIR}/training_curves.png")

    # Final evaluation on test set
    print("\nEvaluating on test set …")
    model.load_state_dict(torch.load(CLASSIFIER_CKPT, map_location=DEVICE))
    model.to(DEVICE).eval()

    from torch.nn import BCEWithLogitsLoss
    pos_w     = torch.tensor([2000.0 / 584.0], device=DEVICE)
    criterion = LabelSmoothBCELoss(pos_weight=pos_w)
    test_metrics = evaluate(model, test_loader, criterion)
    print(f"\nTest AUC-ROC: {test_metrics['auc']:.4f} | "
          f"Loss: {test_metrics['loss']:.4f}")

    print(f"\nBest model saved → {CLASSIFIER_CKPT}")
    print("Training complete. Run benchmark.py to evaluate saliency methods.")


if __name__ == "__main__":
    main()
