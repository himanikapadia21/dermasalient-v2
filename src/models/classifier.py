"""
EfficientNet-B4 binary classifier for melanoma detection.

Training pipeline features:
  • Mixed-precision (AMP) via torch.cuda.amp
  • Class-weighted BCE loss — pos_weight = N_benign / N_malignant
  • AdamW optimiser (weight decay prevents overfitting; better than plain Adam
    on vision tasks per Loshchilov & Hutter 2019)
  • CosineAnnealingWarmRestarts scheduler with linear warmup
  • Gradient clipping (max_norm=1.0)
  • Label smoothing (ε=0.1)
  • Early stopping on validation AUC-ROC (not accuracy — misleading on
    imbalanced data)
  • Best checkpoint saved to weights/efficientnet_b4_best.pth
  • Training history saved to outputs/reports/training_history.csv
"""
import os
import math
import csv
import time

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score
import timm

from src.utils.config import (
    BACKBONE, LR, WEIGHT_DECAY, EPOCHS, WARMUP_EPOCHS, PATIENCE,
    GRAD_CLIP, LABEL_SMOOTH, T_0, T_MULT, N_BENIGN, N_MALIGNANT,
    DEVICE, AMP, CLASSIFIER_CKPT, REPORT_DIR,
)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_classifier(backbone: str = BACKBONE, pretrained: bool = True) -> nn.Module:
    """Return timm EfficientNet-B4 with a single sigmoid output unit.

    Using timm ensures we get the correct pre-trained weights and consistent
    feature extractor API for the CAM saliency methods.
    """
    model = timm.create_model(backbone, pretrained=pretrained, num_classes=1)
    return model


def get_target_layer(model: nn.Module) -> nn.Module:
    """Return the last convolutional block for CAM target layer.

    EfficientNet-B4 via timm exposes model.blocks[-1] as the deepest
    convolutional stage before global pooling.
    """
    return model.blocks[-1]


# ---------------------------------------------------------------------------
# Label-smoothed BCE wrapper
# ---------------------------------------------------------------------------

class LabelSmoothBCELoss(nn.Module):
    """BCE with logits + label smoothing.

    Smoothing reduces over-confident predictions and improves calibration,
    particularly useful under class imbalance (Müller et al. 2019).
    """
    def __init__(self, epsilon: float = LABEL_SMOOTH, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.epsilon = epsilon
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        smoothed = targets * (1 - self.epsilon) + 0.5 * self.epsilon
        return self.bce(logits, smoothed)


# ---------------------------------------------------------------------------
# Warmup + cosine scheduler
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    """Linear warmup for `warmup_epochs`, then CosineAnnealingWarmRestarts."""

    def __init__(self, optimizer, warmup_epochs: int, T_0: int, T_mult: int,
                 base_lr: float, last_epoch: int = -1):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr       = base_lr
        self.cosine        = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, last_epoch=last_epoch
        )
        self.epoch         = 0

    def step(self) -> None:
        self.epoch += 1
        if self.epoch <= self.warmup_epochs:
            lr = self.base_lr * self.epoch / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        else:
            self.cosine.step(self.epoch - self.warmup_epochs)

    def get_last_lr(self) -> list[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
) -> dict:
    model.train()
    losses, preds, trues = [], [], []

    for batch in loader:
        images = batch["image"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        with autocast(enabled=AMP):
            logits = model(images).squeeze(1)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
        trues.extend(labels.cpu().numpy())

    auc = roc_auc_score(trues, preds) if len(set(trues)) > 1 else 0.5
    return {"loss": float(np.mean(losses)), "auc": auc}


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module) -> dict:
    model.eval()
    losses, preds, trues = [], [], []

    for batch in loader:
        images = batch["image"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)

        with autocast(enabled=AMP):
            logits = model(images).squeeze(1)
            loss   = criterion(logits, labels)

        losses.append(loss.item())
        preds.extend(torch.sigmoid(logits).cpu().numpy())
        trues.extend(labels.cpu().numpy())

    auc = roc_auc_score(trues, preds) if len(set(trues)) > 1 else 0.5
    return {"loss": float(np.mean(losses)), "auc": auc}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = EPOCHS,
) -> list[dict]:
    """Full training loop with early stopping and checkpoint saving.

    Returns list of per-epoch metric dicts.
    """
    pos_weight = torch.tensor([N_BENIGN / N_MALIGNANT], device=DEVICE)
    criterion  = LabelSmoothBCELoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scaler     = GradScaler(enabled=AMP)
    scheduler  = WarmupCosineScheduler(
        optimizer, warmup_epochs=WARMUP_EPOCHS,
        T_0=T_0, T_mult=T_MULT, base_lr=LR,
    )

    model.to(DEVICE)
    best_auc    = 0.0
    no_improve  = 0
    history     = []

    for epoch in range(1, epochs + 1):
        t0      = time.time()
        tr      = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        va      = evaluate(model, val_loader, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        row = {
            "epoch":    epoch,
            "tr_loss":  tr["loss"],   "tr_auc":  tr["auc"],
            "va_loss":  va["loss"],   "va_auc":  va["auc"],
            "lr":       scheduler.get_last_lr()[0],
            "elapsed":  elapsed,
        }
        history.append(row)

        flag = ""
        if va["auc"] > best_auc:
            best_auc   = va["auc"]
            no_improve = 0
            torch.save(model.state_dict(), CLASSIFIER_CKPT)
            flag = "  ← best"
        else:
            no_improve += 1

        print(
            f"Epoch {epoch:03d}/{epochs}  "
            f"tr_loss={tr['loss']:.4f}  tr_auc={tr['auc']:.4f}  "
            f"va_loss={va['loss']:.4f}  va_auc={va['auc']:.4f}  "
            f"lr={row['lr']:.2e}  {elapsed:.0f}s{flag}"
        )

        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no val-AUC improvement for {PATIENCE} epochs).")
            break

    # Save history
    os.makedirs(REPORT_DIR, exist_ok=True)
    csv_path = os.path.join(REPORT_DIR, "training_history.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved training history → {csv_path}")

    return history


def load_classifier(backbone: str = BACKBONE) -> nn.Module:
    """Load the best saved classifier checkpoint."""
    model = build_classifier(backbone, pretrained=False)
    model.load_state_dict(torch.load(CLASSIFIER_CKPT, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model
