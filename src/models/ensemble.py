"""
Ensemble model wrapper — aggregates classifier logits and saliency maps
from multiple backbones or checkpoints.

For DermaSalient v2 the ensemble operates at the saliency level:
individual method maps are fused by src/saliency/fusion.py.
This module wraps a collection of classifiers for ensemble classification
(used when multiple backbone checkpoints are available).
"""
import os
import torch
import torch.nn as nn
import numpy as np

from src.utils.config import DEVICE, WEIGHTS_DIR
from src.models.classifier import build_classifier


class ClassifierEnsemble(nn.Module):
    """Averages sigmoid outputs from multiple classifier checkpoints.

    Useful when training multiple seeds or backbones and wanting a
    committee-of-experts prediction without costly MCDropout.
    """

    def __init__(self, ckpt_paths: list[str], backbone: str = "efficientnet_b4"):
        super().__init__()
        self.models = nn.ModuleList()
        for ckpt in ckpt_paths:
            m = build_classifier(backbone, pretrained=False)
            m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            m.eval()
            self.models.append(m)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns averaged sigmoid probability."""
        probs = [torch.sigmoid(m(x)) for m in self.models]
        return torch.stack(probs, dim=0).mean(0)

    @torch.no_grad()
    def predict_uncertainty(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean_prob, std_prob) — std used as calibrated uncertainty."""
        probs = torch.stack([torch.sigmoid(m(x)) for m in self.models], dim=0)
        return probs.mean(0), probs.std(0)
