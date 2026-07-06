# DermaSalient v2 — Advanced Medical Saliency Detection System

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-HAM10000-purple.svg)](https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification)

> Benchmarking 9 saliency detection methods on dermoscopy images with a clinical Gradio web app for explainable AI in dermatology.

---

## Overview

Skin cancer is one of the most common cancers worldwide. While AI models can classify skin lesions with high accuracy, clinicians cannot trust decisions they cannot explain. **DermaSalient v2** addresses this by benchmarking 9 saliency detection methods — from gradient-based CAM variants to dedicated salient object detection networks and foundation models — against real clinical lesion masks, then providing an interactive tool for dermatologists to explore model explanations.

### Key Results

| Model | F-measure ↑ | Dice ↑ | IoU ↑ | S-measure ↑ | Speed (ms) |
|---|---|---|---|---|---|
| **U²-Net** | **0.357** | 0.177 | 0.126 | 0.259 | 82 |
| ScoreCAM | 0.346 | 0.047 | 0.034 | 0.058 | 1171 |
| GradCAM | 0.328 | 0.034 | 0.025 | 0.044 | 1172 |
| GradCAM++ | 0.314 | 0.017 | 0.013 | 0.027 | 1172 |
| EigenCAM | 0.312 | 0.000 | 0.000 | 0.008 | 1169 |
| Ensemble | 0.304 | 0.028 | 0.022 | 0.046 | — |
| XGradCAM | 0.296 | 0.013 | 0.010 | 0.021 | 1170 |
| LayerCAM | 0.291 | 0.002 | 0.001 | 0.012 | 1171 |
| SAM | 0.245 | **0.283** | **0.203** | **0.377** | 318 |

> Evaluated on 200 test images from HAM10000 against clinical dermatologist segmentation masks.
> 23/36 model pairs show statistically significant differences (Wilcoxon signed-rank, p<0.05).

**Key finding:** U²-Net achieves the highest F-measure (precision-weighted), while SAM achieves the best Dice and IoU (area overlap). This reveals a precision-coverage tradeoff: U²-Net is more precise about lesion boundaries, SAM provides more complete lesion coverage.

---

## Features

- **9 saliency models** benchmarked head-to-head: GradCAM, GradCAM++, ScoreCAM, LayerCAM, EigenCAM, XGradCAM, U²-Net, SAM (point-prompted), Ensemble
- **Research-grade metrics**: MAE, F-measure (β=0.3), Weighted F-measure, S-measure, E-measure, Dice, IoU
- **Statistical significance**: Wilcoxon signed-rank tests across all model pairs with 95% confidence intervals
- **CRF post-processing**: Bilateral filter edge-aware refinement for cleaner saliency boundaries
- **4-tab Gradio app**: Single image analysis, model comparison, benchmark results, batch processing
- **EfficientNet-B4 classifier**: Fine-tuned on HAM10000 with Test AUC = 0.8838

---

## Project Structure

```
darmasalic/
├── src/
│   ├── data/
│   │   ├── dataset.py          # HAM10000 data pipeline + WeightedRandomSampler
│   │   ├── augmentations.py    # Albumentations training/val/TTA pipelines
│   │   └── masks.py            # Mask loading utilities
│   ├── models/
│   │   └── classifier.py       # EfficientNet-B4 training loop (AMP, AdamW, cosine LR)
│   ├── saliency/
│   │   ├── cam_methods.py      # GradCAM, GradCAM++, ScoreCAM, LayerCAM, EigenCAM, XGradCAM
│   │   ├── u2net_infer.py      # U²-Net salient object detection inference
│   │   ├── sam_infer.py        # SAM point-prompted segmentation
│   │   ├── postprocess.py      # CRF/bilateral filter refinement + morphological cleaning
│   │   └── fusion.py           # Weighted ensemble fusion
│   ├── evaluation/
│   │   ├── metrics.py          # MAE, F-measure, Dice, IoU, S-measure, E-measure, PR curves
│   │   └── statistical_tests.py # Wilcoxon tests, bootstrap confidence intervals
│   └── utils/
│       ├── config.py           # Central config — all paths and hyperparameters
│       └── visualization.py    # Plotly benchmark charts, significance heatmap
├── app/
│   └── app.py                  # 4-tab Gradio clinical interface
├── weights/                    # Model weights (not tracked in git)
├── outputs/
│   ├── benchmarks/             # full_results.csv, leaderboard.csv, significance_table.csv
│   └── visualizations/         # Benchmark bar charts, significance heatmap
├── train.py                    # EfficientNet-B4 training script
├── benchmark.py                # Full 9-model benchmark pipeline
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.10
- CUDA-capable GPU (tested on RTX 5060 Laptop)
- ~5GB disk space for dataset + weights

### Installation

```bash
git clone https://github.com/himanikapadia21/dermasalient-v2.git
cd dermasalient-v2
pip install -r requirements.txt
```

### Dataset

Download HAM10000 via kagglehub:

```python
import kagglehub
path = kagglehub.dataset_download("surajghuwalewala/ham1000-segmentation-and-classification")
print(path)
```

Update `src/utils/config.py` with the downloaded path:

```python
HAM_ROOT = r"path/to/ham1000-segmentation-and-classification"
IMG_DIR  = HAM_ROOT + "/images"
MASK_DIR = HAM_ROOT + "/masks"
TRAIN_CSV = HAM_ROOT + "/GroundTruth.csv"
```

---

## Usage

### Step 1 — Train the Classifier

```bash
python train.py
```

Trains EfficientNet-B4 for 30 epochs with early stopping. Saves best weights to `weights/efficientnet_b4_best.pth`.
Expected: ~6 hours on RTX 5060, final Test AUC ~0.88.

### Step 2 — Run the Benchmark

```bash
python benchmark.py --n_images 200
```

Evaluates all 9 saliency models on the test set. Saves:
- `outputs/benchmarks/full_results.csv` — per-image × per-model metrics (1800 rows)
- `outputs/benchmarks/leaderboard.csv` — aggregate leaderboard
- `outputs/benchmarks/significance_table.csv` — Wilcoxon p-values
- `outputs/visualizations/benchmark_*.png` — bar charts

### Step 3 — Launch the App

```bash
python app/app.py
```

Opens at `http://127.0.0.1:7860`. Use `--share` flag for a public Gradio link.

---

## App Guide

### Tab 1 — Single Image Analysis

Upload a dermoscopy image, select a saliency model, click **Analyse**.

- **Saliency Heatmap**: Red/yellow = model attended here. Blue = ignored.
- **Overlay**: Heatmap blended onto original image for spatial context.
- **Confidence Score**: Mean saliency in detected region, normalized 0–100.
- **Region Text**: Location of peak saliency (e.g. "center-left").
- **Download Overlay**: Save the annotated image for clinical reporting.

### Tab 2 — Model Comparison

Upload one image, all 9 models run simultaneously. Side-by-side grid shows where each model looks — useful for identifying consensus regions across methods.

### Tab 3 — Benchmark Results

Interactive leaderboard loaded from `full_results.csv`. Switch metrics via dropdown. Significance table shows which model differences are statistically real.

### Tab 4 — Batch Processing

Upload a zip of images, download a zip of saliency overlays. For screening large patient cohorts without one-by-one processing.

---

## Methodology

### Classifier Training

| Component | Choice | Rationale |
|---|---|---|
| Backbone | EfficientNet-B4 | Best accuracy/parameter tradeoff for medical imaging |
| Optimizer | AdamW (wd=1e-4) | Better generalization than Adam |
| LR Schedule | CosineAnnealingWarmRestarts | Avoids local minima |
| Loss | BCEWithLogitsLoss + pos_weight | Handles class imbalance (584 mel vs 2000 benign) |
| Precision | FP16 mixed precision | 2x GPU memory efficiency |
| Metric | AUC-ROC | Appropriate for imbalanced medical classification |

### Saliency Post-Processing

```
Raw saliency map
      ↓
Bilateral filter (edge-aware smoothing)
      ↓
Adaptive Otsu threshold
      ↓
Remove small objects (<500px)
      ↓
Binary closing (fill holes)
      ↓
Final clean binary mask
```

### Evaluation Metrics

All metrics computed against HAM10000 clinical dermatologist segmentation masks:

| Metric | Formula | Reference |
|---|---|---|
| MAE | mean\|pred - gt\| | Borji et al., IEEE TIP 2015 |
| F-measure (β=0.3) | (1+β²)·P·R / (β²·P+R) | Achanta et al., CVPR 2009 |
| S-measure | α·S_o + (1-α)·S_r | Fan et al., ICCV 2017 |
| E-measure | Enhanced alignment matrix | Fan et al., IJCAI 2018 |
| Dice | 2·TP / (2·TP+FP+FN) | Standard medical segmentation |
| IoU | TP / (TP+FP+FN) | Standard detection |

---

## Results

### Leaderboard

U²-Net wins on F-measure. SAM wins on Dice and IoU. This reveals a **precision-coverage tradeoff**:

- **U²-Net** is more precise about lesion boundaries (higher F-measure)
- **SAM** provides more complete lesion coverage (higher Dice/IoU)

For clinical applications requiring complete lesion delineation (e.g. surgical planning), SAM is preferred. For applications requiring precise boundary detection (e.g. margin assessment), U²-Net is preferred.

### Statistical Significance

23 of 36 model pairs show statistically significant performance differences (Wilcoxon signed-rank test, p<0.05), confirming that observed ranking differences are not due to random variation.

---

## Requirements

```
torch>=2.1.0
torchvision>=0.16.0
timm>=0.9.12
grad-cam>=1.4.8
albumentations>=1.3.1
segment-anything
opencv-python
opencv-contrib-python
scikit-image
scikit-learn
scipy
pandas
numpy
matplotlib
seaborn
plotly
kaleido
gradio>=4.0.0
tqdm
Pillow
gdown
```

---

## Citation

If you use DermaSalient v2 in your research, please cite:

```bibtex
@software{kapadia2026dermasalient,
  author    = {Kapadia, Himani},
  title     = {DermaSalient v2: Benchmarking Saliency Detection Methods for Dermoscopy Images},
  year      = {2026},
  url       = {https://github.com/himanikapadia21/dermasalient-v2},
  note      = {Master's project, Auckland University of Technology}
}
```

---

## Acknowledgements

- HAM10000 dataset: Tschandl et al., Nature Medicine 2018
- U²-Net: Qin et al., Pattern Recognition 2020
- Segment Anything Model: Kirillov et al., ICCV 2023
- pytorch-grad-cam: Gildenblat et al.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

*For research use only. Not intended for clinical diagnosis.*
