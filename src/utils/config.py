"""
Central configuration — every path and hyperparameter in the project comes from here.
Importing this module anywhere gives a single source of truth.
"""
import os
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT   = r"C:\Users\himan\Downloads\darmasalic\siim-isic-melanoma-classification"

# HAM10000 dataset (images + real masks)
HAM_ROOT  = r"C:\Users\himan\.cache\kagglehub\datasets\surajghuwalewala\ham1000-segmentation-and-classification\versions\2"
IMG_DIR   = HAM_ROOT + r"\images"
MASK_DIR  = HAM_ROOT + r"\masks"
TRAIN_CSV = HAM_ROOT + r"\GroundTruth.csv"    # ISIC 2018 segmentation masks

BASE_DIR    = r"C:\Users\himan\Downloads\darmasalic"
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
BENCH_DIR   = os.path.join(OUTPUT_DIR, "benchmarks")
VIZ_DIR     = os.path.join(OUTPUT_DIR, "visualizations")
SAL_DIR     = os.path.join(OUTPUT_DIR, "saliency_maps")
REPORT_DIR  = os.path.join(OUTPUT_DIR, "reports")

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
CLASSIFIER_CKPT = os.path.join(WEIGHTS_DIR, "efficientnet_b4_best.pth")
U2NET_CKPT      = os.path.join(WEIGHTS_DIR, "u2net.pth")
SAM_CKPT        = os.path.join(WEIGHTS_DIR, "sam_vit_b_01ec64.pth")

# ---------------------------------------------------------------------------
# Image / training hyperparameters
# ---------------------------------------------------------------------------
IMG_SIZE    = 512
BATCH_SIZE  = 16
EPOCHS      = 30
LR          = 1e-4
WEIGHT_DECAY = 1e-4
SEED        = 42
WARMUP_EPOCHS = 3
PATIENCE    = 7           # early stopping patience (AUC)
GRAD_CLIP   = 1.0
LABEL_SMOOTH = 0.1
T_0         = 10          # CosineAnnealingWarmRestarts period
T_MULT      = 2

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
N_BENIGN    = 2000
N_MALIGNANT = 584
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15
# TEST_FRAC = 0.15 (implicit)

# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------
BACKBONE    = "efficientnet_b4"

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
AMP         = True        # automatic mixed precision

# ---------------------------------------------------------------------------
# Saliency / post-processing
# ---------------------------------------------------------------------------
CRF_ITER    = 10
TTA_FOLDS   = 8

# SAM variant to use
SAM_MODEL_TYPE = "vit_b"

# Weights for ensemble fusion (learned offline from benchmark; default equal)
ENSEMBLE_WEIGHTS = {
    "gradcam":      1.0,
    "gradcam_pp":   1.0,
    "scorecam":     1.0,
    "layercam":     1.0,
    "eigencam":     1.0,
    "xgradcam":     1.0,
    "u2net":        1.5,   # SOD model typically outperforms CAM
    "sam":          1.5,
}

# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------
def seed_everything(seed: int = SEED) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# Ensure output directories exist at import time
# ---------------------------------------------------------------------------
for _d in [WEIGHTS_DIR, BENCH_DIR, VIZ_DIR, SAL_DIR, REPORT_DIR, MASK_DIR]:
    os.makedirs(_d, exist_ok=True)
