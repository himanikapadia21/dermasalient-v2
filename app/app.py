"""
DermaSalient v2 — Research-Grade Gradio Application.

Four tabs:
  Tab 1 — Single Image Analysis
  Tab 2 — Model Comparison (all 9 methods on one image)
  Tab 3 — Benchmark Results (interactive leaderboard from CSV)
  Tab 4 — Batch Processing (zip upload)

Launch:
    python app/app.py
"""
import os
import sys
import io
import zipfile
import tempfile
import time
import warnings

# Ensure project root on path when invoked from app/ subdirectory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pandas as pd
import torch
import gradio as gr
from PIL import Image
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.config import (
    seed_everything, DEVICE, CLASSIFIER_CKPT, BENCH_DIR,
)
from src.data.augmentations import val_transforms
from src.models.classifier import load_classifier
from src.saliency.cam_methods import run_all_cams, classifier_confidence
from src.saliency.u2net_infer import u2net_saliency_from_tensor
from src.saliency.sam_infer import sam_saliency
from src.saliency.postprocess import full_postprocess_pipeline
from src.saliency.fusion import ensemble_saliency
from src.utils.visualization import (
    overlay_saliency, draw_bbox_from_mask, describe_saliency_region,
    saliency_gallery, benchmark_bar_chart, significance_heatmap,
    pil_to_rgb, rgb_to_pil, fig_to_pil,
)
from src.data.augmentations import tta_predict
from src.evaluation.metrics import compute_all_metrics

warnings.filterwarnings("ignore")
seed_everything()


# ---------------------------------------------------------------------------
# Lazy-loaded models (initialised once on first use)
# ---------------------------------------------------------------------------

_classifier = None


def get_classifier():
    global _classifier
    if _classifier is None:
        if not os.path.isfile(CLASSIFIER_CKPT):
            raise gr.Error("Classifier not found. Run train.py first.")
        _classifier = load_classifier()
    return _classifier


def pil_to_tensor(img_pil: Image.Image) -> tuple[np.ndarray, torch.Tensor]:
    img_rgb = pil_to_rgb(img_pil)
    tf      = val_transforms()
    tensor  = tf(image=img_rgb)["image"].to(DEVICE)
    return img_rgb, tensor


# ---------------------------------------------------------------------------
# Tab 1 — Single Image Analysis
# ---------------------------------------------------------------------------

def analyze_single(
    img_pil: Image.Image,
    method_name: str,
    apply_crf: bool,
    apply_tta: bool,
) -> tuple:
    """Returns: original, saliency_overlay, crf_refined, binary_mask, bbox_img, info_text"""
    if img_pil is None:
        raise gr.Error("Please upload an image.")

    classifier = get_classifier()
    img_rgb, tensor = pil_to_tensor(img_pil)
    h, w = img_rgb.shape[:2]

    t0         = time.perf_counter()
    confidence = classifier_confidence(classifier, tensor)

    # --- Generate saliency map for selected method ---
    if method_name == "ensemble":
        cams   = run_all_cams(classifier, tensor, target_size=(h, w))
        try:
            u2_sal = u2net_saliency_from_tensor(tensor)
        except Exception:
            u2_sal = cams.get("gradcam", np.zeros((h, w), np.float32))
        sal_map = ensemble_saliency({**cams, "u2net": u2_sal})
    elif method_name == "u2net":
        try:
            sal_map = u2net_saliency_from_tensor(tensor)
        except Exception as e:
            raise gr.Error(f"U²-Net failed: {e}")
    elif method_name == "sam":
        cams    = run_all_cams(classifier, tensor, target_size=(h, w))
        gradcam = cams.get("gradcam", np.zeros((h, w), np.float32))
        sal_map = sam_saliency(img_rgb, gradcam)
    else:
        cams    = run_all_cams(classifier, tensor, target_size=(h, w))
        sal_map = cams.get(method_name, np.zeros((h, w), np.float32))

    # Optional TTA
    if apply_tta and method_name not in ("sam",):
        def _infer_fn(t):
            inp = t.unsqueeze(0).to(DEVICE)
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
            with GradCAM(model=classifier,
                         target_layers=[classifier.blocks[-1]]) as cam:
                return cam(inp, [BinaryClassifierOutputTarget(0)])[0]
        sal_map = tta_predict(img_rgb, _infer_fn)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Post-processing
    refined_soft, binary_clean = full_postprocess_pipeline(
        img_rgb, sal_map, apply_crf=apply_crf
    )

    # Outputs
    overlay    = overlay_saliency(img_rgb, sal_map)
    crf_img    = overlay_saliency(img_rgb, refined_soft) if apply_crf else overlay
    bin_img    = (binary_clean * 255).clip(0, 255).astype(np.uint8)
    bin_rgb    = np.stack([bin_img] * 3, axis=-1)
    bbox_img   = draw_bbox_from_mask(img_rgb, binary_clean)
    region_txt = describe_saliency_region(sal_map)

    risk_label = "MALIGNANT" if confidence > 0.5 else "BENIGN"
    info_text  = (
        f"**Prediction:** {risk_label} (confidence {confidence:.1%})\n\n"
        f"**Method:** {method_name} | CRF: {'ON' if apply_crf else 'OFF'} | "
        f"TTA: {'ON' if apply_tta else 'OFF'}\n\n"
        f"**Inference time:** {elapsed_ms:.0f} ms\n\n"
        f"**Region:** {region_txt}\n\n"
        f"**Device:** {DEVICE}"
    )

    return (
        img_pil,
        rgb_to_pil(overlay),
        rgb_to_pil(crf_img),
        Image.fromarray(bin_rgb),
        rgb_to_pil(bbox_img),
        info_text,
    )


# ---------------------------------------------------------------------------
# Tab 2 — Model Comparison
# ---------------------------------------------------------------------------

def compare_all_models(img_pil: Image.Image, apply_crf: bool) -> tuple:
    """Run all 9 models and return a gallery figure + metric bar chart."""
    if img_pil is None:
        raise gr.Error("Please upload an image.")

    classifier  = get_classifier()
    img_rgb, tensor = pil_to_tensor(img_pil)
    h, w = img_rgb.shape[:2]

    cams = run_all_cams(classifier, tensor, target_size=(h, w))

    try:
        u2_sal = u2net_saliency_from_tensor(tensor)
    except Exception:
        u2_sal = cams.get("gradcam", np.zeros((h, w), np.float32))

    gradcam_sal = cams.get("gradcam", np.zeros((h, w), np.float32))
    try:
        sam_mask = sam_saliency(img_rgb, gradcam_sal)
    except Exception:
        sam_mask = np.zeros((h, w), np.float32)

    ens_sal = ensemble_saliency({**cams, "u2net": u2_sal})

    all_sals = {**cams, "u2net": u2_sal, "sam": sam_mask, "ensemble": ens_sal}

    binary_maps = {}
    for name, sal in all_sals.items():
        _, bm = full_postprocess_pipeline(img_rgb, sal, apply_crf=apply_crf)
        binary_maps[name] = bm.astype(np.float32)

    gallery_fig = saliency_gallery(img_rgb, all_sals, binary_maps)
    gallery_pil = fig_to_pil(gallery_fig)
    plt.close(gallery_fig)

    # Metric comparison bar chart (confidence as proxy when no GT mask)
    confidence = classifier_confidence(classifier, tensor)
    bar_fig = go.Figure(go.Bar(
        x=list(all_sals.keys()),
        y=[float(s.mean()) for s in all_sals.values()],
        name="Mean Saliency",
    ))
    bar_fig.update_layout(title=f"Mean saliency per method "
                                f"(classifier confidence: {confidence:.1%})",
                          template="plotly_white")

    return gallery_pil, bar_fig


# ---------------------------------------------------------------------------
# Tab 3 — Benchmark Results
# ---------------------------------------------------------------------------

def load_benchmark_results():
    csv_path = os.path.join(BENCH_DIR, "full_results.csv")
    if not os.path.isfile(csv_path):
        return None, None, None

    df = pd.read_csv(csv_path)
    return df


def show_benchmark(metric: str) -> tuple:
    df = load_benchmark_results()
    if df is None:
        return (
            "No benchmark results found. Run benchmark.py first.",
            go.Figure(),
            go.Figure(),
        )

    bar = benchmark_bar_chart(df, metric)

    # Significance table
    sig_path = os.path.join(BENCH_DIR, "significance_table.csv")
    if os.path.isfile(sig_path):
        sig_df = pd.read_csv(sig_path)
        sig_html = sig_df.to_html(index=False, float_format=lambda x: f"{x:.4f}")
    else:
        sig_html = "<p>Run benchmark.py for significance tests.</p>"

    # Leaderboard summary
    lb_path = os.path.join(BENCH_DIR, "leaderboard.csv")
    if os.path.isfile(lb_path):
        lb = pd.read_csv(lb_path, index_col=0)
        metrics_to_show = [c for c in lb.columns if "mean" in c][:6]
        winner = lb.index[0] if "f_measure_mean" in lb.columns else "N/A"
        summary = (
            f"**Winner:** {winner.upper()}  "
            f"(F-measure = {lb.loc[winner, 'f_measure_mean']:.4f})\n\n"
            f"*Evaluated against ISIC 2018 ground-truth masks.*\n\n"
            f"Wilcoxon p < 0.05 pairs: "
            + (f"{sig_df['significant'].sum()}/{len(sig_df)}" if os.path.isfile(sig_path) else "N/A")
        )
    else:
        summary = "Run benchmark.py to generate results."

    return summary, bar, sig_html


# ---------------------------------------------------------------------------
# Tab 4 — Batch Processing
# ---------------------------------------------------------------------------

def batch_process(
    zip_file_obj,
    method_name: str,
    apply_crf: bool,
) -> str:
    """Process all images in a zip archive; return path to output zip."""
    if zip_file_obj is None:
        raise gr.Error("Please upload a zip file containing images.")

    classifier = get_classifier()
    out_dir    = tempfile.mkdtemp(prefix="dermasalient_batch_")

    with zipfile.ZipFile(zip_file_obj.name, "r") as zf:
        image_names = [n for n in zf.namelist()
                       if n.lower().endswith((".jpg", ".jpeg", ".png"))]

        for img_name in image_names[:500]:     # cap at 500
            try:
                with zf.open(img_name) as f:
                    pil_img = Image.open(f).convert("RGB")
                img_rgb = pil_to_rgb(pil_img)
                tf      = val_transforms()
                tensor  = tf(image=img_rgb)["image"].to(DEVICE)
                h, w    = img_rgb.shape[:2]

                cams    = run_all_cams(classifier, tensor, target_size=(h, w))
                sal     = cams.get(method_name,
                                   ensemble_saliency(cams))
                _, bm   = full_postprocess_pipeline(img_rgb, sal,
                                                    apply_crf=apply_crf)
                overlay = overlay_saliency(img_rgb, sal)

                base = os.path.splitext(os.path.basename(img_name))[0]
                Image.fromarray(overlay).save(
                    os.path.join(out_dir, f"{base}_overlay.png"))
                Image.fromarray((bm * 255).astype(np.uint8)).save(
                    os.path.join(out_dir, f"{base}_mask.png"))
            except Exception as exc:
                print(f"[WARN] Batch {img_name}: {exc}")

    out_zip = os.path.join(out_dir, "dermasalient_batch_results.zip")
    with zipfile.ZipFile(out_zip, "w") as zf:
        for fn in os.listdir(out_dir):
            if fn.endswith(".png"):
                zf.write(os.path.join(out_dir, fn), fn)

    return out_zip


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

METHODS = ["gradcam", "gradcam_pp", "scorecam", "layercam",
           "eigencam", "xgradcam", "u2net", "sam", "ensemble"]

METRICS_LIST = ["f_measure", "dice", "iou", "mae", "s_measure",
                "e_measure", "weighted_f"]


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="DermaSalient v2 — Expert Medical Saliency Analysis",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# DermaSalient v2 — Expert Medical Saliency Analysis\n"
            "Research-grade lesion saliency detection on dermoscopy images.  "
            "9 models · Dense CRF · SAM point-prompted · Statistical benchmarks."
        )

        # ----------------------------------------------------------------
        # Tab 1 — Single Image
        # ----------------------------------------------------------------
        with gr.Tab("Single Image Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    t1_img     = gr.Image(type="pil", label="Upload Image")
                    t1_method  = gr.Dropdown(METHODS, value="ensemble",
                                             label="Saliency Method")
                    t1_crf     = gr.Checkbox(value=True,  label="Apply CRF Post-processing")
                    t1_tta     = gr.Checkbox(value=False, label="Apply TTA (8 augmentations)")
                    t1_run     = gr.Button("Analyse", variant="primary")
                with gr.Column(scale=2):
                    t1_orig    = gr.Image(label="Original")
                    t1_overlay = gr.Image(label="Saliency Overlay")
                    t1_crf_out = gr.Image(label="CRF-Refined")
                    t1_binary  = gr.Image(label="Binary Mask")
                    t1_bbox    = gr.Image(label="Bounding Box")
                    t1_info    = gr.Markdown()

            t1_run.click(
                analyze_single,
                inputs=[t1_img, t1_method, t1_crf, t1_tta],
                outputs=[t1_orig, t1_overlay, t1_crf_out, t1_binary, t1_bbox, t1_info],
            )

        # ----------------------------------------------------------------
        # Tab 2 — Model Comparison
        # ----------------------------------------------------------------
        with gr.Tab("Model Comparison"):
            with gr.Row():
                with gr.Column(scale=1):
                    t2_img  = gr.Image(type="pil", label="Upload Image")
                    t2_crf  = gr.Checkbox(value=True, label="Apply CRF")
                    t2_run  = gr.Button("Compare All 9 Models", variant="primary")
                with gr.Column(scale=2):
                    t2_gallery = gr.Image(label="Gallery (all methods)")
                    t2_bar     = gr.Plot(label="Mean Saliency per Method")

            t2_run.click(
                compare_all_models,
                inputs=[t2_img, t2_crf],
                outputs=[t2_gallery, t2_bar],
            )

        # ----------------------------------------------------------------
        # Tab 3 — Benchmark Results
        # ----------------------------------------------------------------
        with gr.Tab("Benchmark Results"):
            with gr.Row():
                t3_metric = gr.Dropdown(METRICS_LIST, value="f_measure",
                                        label="Metric")
                t3_load   = gr.Button("Load Results", variant="primary")
            with gr.Row():
                t3_summary = gr.Markdown()
            with gr.Row():
                t3_bar     = gr.Plot(label="Method Comparison")
            with gr.Row():
                t3_sig     = gr.HTML(label="Statistical Significance Table")

            t3_load.click(
                show_benchmark,
                inputs=[t3_metric],
                outputs=[t3_summary, t3_bar, t3_sig],
            )

        # ----------------------------------------------------------------
        # Tab 4 — Batch Processing
        # ----------------------------------------------------------------
        with gr.Tab("Batch Processing"):
            with gr.Row():
                with gr.Column():
                    t4_zip    = gr.File(label="Upload ZIP of images", file_types=[".zip"])
                    t4_method = gr.Dropdown(METHODS[:-1], value="gradcam",
                                            label="Method")
                    t4_crf    = gr.Checkbox(value=True, label="Apply CRF")
                    t4_run    = gr.Button("Process Batch", variant="primary")
                with gr.Column():
                    t4_out    = gr.File(label="Download Results ZIP")
                    gr.Markdown(
                        "_Output contains overlay PNG and binary mask PNG "
                        "for each image in the input ZIP._"
                    )

            t4_run.click(
                batch_process,
                inputs=[t4_zip, t4_method, t4_crf],
                outputs=[t4_out],
            )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(share=True, server_port=7860)
