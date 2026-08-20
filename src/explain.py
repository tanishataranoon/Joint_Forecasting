"""
SHAP explainability wrapper (shap.GradientExplainer) for the multi-branch
list-input architecture.

STATUS: this uploaded notebook (thesis__1_.ipynb) ends at BranchFusionBlock
and doesn't yet contain your SHAP cells -- but per your notes, Sections
13-14 of your v6 notebook already have this working, including the
batch-size-1 eval-mode forward pass fix. Port that code in here rather
than rewriting from scratch; this file is just the landing spot so it
lives in src/ instead of a notebook cell, callable from anywhere.

Runs fine on the personal PC post-training -- SHAP on a trained model in
eval mode doesn't need the GPU that training did.

Usage:
    python src/explain.py --checkpoint checkpoints/best_model.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shap
import torch

from src.train import build_model, prepare_data
from src.utils import get_device, load_checkpoint, load_config


def make_gradient_explainer(model, background_batch):
    """
    background_batch: tuple (met, veg, eng) tensors used as the SHAP
    background distribution -- typically a small random sample from the
    training set (e.g. 50-100 sequences), not the full set.

    IMPORTANT (per your existing notes): the model must be in eval() mode
    and support a batch-size-1 forward pass for GradientExplainer and
    single-sample inference to work correctly. Re-verify this whenever
    the architecture changes (e.g. once heads.py is finalized).
    """
    model.eval()
    explainer = shap.GradientExplainer(model, list(background_batch))
    return explainer


def explain_batch(explainer, sample_batch):
    """sample_batch: tuple (met, veg, eng) tensors to explain."""
    shap_values = explainer.shap_values(list(sample_batch))
    return shap_values


def main(checkpoint_path: str, config_path: str = "configs/config.yaml", n_background: int = 50):
    cfg = load_config(config_path)
    device = get_device()

    train_loader, val_loader, test_loader, scalers = prepare_data(cfg)

    model = build_model(cfg).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device.type)
    model.eval()

    # grab a small background sample from train, and a sample to explain from test
    met_bg, veg_bg, eng_bg, _, _ = next(iter(train_loader))
    background = (met_bg[:n_background].to(device), veg_bg[:n_background].to(device), eng_bg[:n_background].to(device))

    met_s, veg_s, eng_s, _, _ = next(iter(test_loader))
    sample = (met_s[:1].to(device), veg_s[:1].to(device), eng_s[:1].to(device))  # batch-size-1 check

    explainer = make_gradient_explainer(model, background)
    shap_values = explain_batch(explainer, sample)

    print("SHAP values computed for 1 sample. Shapes:")
    for i, sv in enumerate(shap_values):
        print(f"  branch {i}: {sv.shape if hasattr(sv, 'shape') else type(sv)}")

    # TODO: port AGRONOMIC_MAP + plotting/reporting logic from your
    # existing Section 13-14 notebook cells here.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.checkpoint, args.config)
