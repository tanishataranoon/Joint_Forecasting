"""
Evaluation metrics: regression (MAE/RMSE/R2, inverse-scaled back to
physical units) and classification (macro-F1, accuracy, per-class
precision/recall) for both drought and heat-stress heads.

Runs on either machine post-training -- only needs the checkpoint +
scalers + a data loader, not the GPU.

Usage:
    python src/evaluate.py --checkpoint checkpoints/best_model.pt --split test
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)

from src.train import build_model, prepare_data
from src.utils import load_checkpoint


@torch.no_grad()
def evaluate(model, loader, scalers, device):
    model.eval()
    all_reg_pred, all_reg_true = [], []
    all_drought_pred, all_drought_true = [], []
    all_heat_pred, all_heat_true = [], []

    for met, veg, eng, y_reg, y_cls in loader:
        met, veg, eng = met.to(device), veg.to(device), eng.to(device)
        out = model(met, veg, eng)

        all_reg_pred.append(out["reg_out"].cpu().numpy())
        all_reg_true.append(y_reg.numpy())
        all_drought_pred.append(out["drought_logits"].argmax(dim=-1).cpu().numpy())
        all_drought_true.append(y_cls[:, 0].numpy())
        all_heat_pred.append(out["heat_logits"].argmax(dim=-1).cpu().numpy())
        all_heat_true.append(y_cls[:, 1].numpy())

    reg_pred = np.concatenate(all_reg_pred)
    reg_true = np.concatenate(all_reg_true)
    # inverse-transform back to physical units (SPI-3 units, deg C Humidex)
    reg_pred_orig = scalers["reg"].inverse_transform(reg_pred)
    reg_true_orig = scalers["reg"].inverse_transform(reg_true)

    drought_pred = np.concatenate(all_drought_pred)
    drought_true = np.concatenate(all_drought_true)
    heat_pred = np.concatenate(all_heat_pred)
    heat_true = np.concatenate(all_heat_true)

    results = {
        "regression": {
            "mae_per_target": mean_absolute_error(reg_true_orig, reg_pred_orig, multioutput="raw_values").tolist(),
            "rmse_per_target": np.sqrt(
                mean_squared_error(reg_true_orig, reg_pred_orig, multioutput="raw_values")
            ).tolist(),
            "r2_per_target": r2_score(reg_true_orig, reg_pred_orig, multioutput="raw_values").tolist(),
        },
        "drought_class": {
            "accuracy": accuracy_score(drought_true, drought_pred),
            "macro_f1": f1_score(drought_true, drought_pred, average="macro"),
            "per_class": precision_recall_fscore_support(drought_true, drought_pred, zero_division=0),
        },
        "heat_class": {
            "accuracy": accuracy_score(heat_true, heat_pred),
            "macro_f1": f1_score(heat_true, heat_pred, average="macro"),
            "per_class": precision_recall_fscore_support(heat_true, heat_pred, zero_division=0),
        },
    }
    return results


def main(checkpoint_path: str, split: str, config_path: str = "configs/config.yaml"):
    from src.utils import get_device, load_config

    cfg = load_config(config_path)
    device = get_device()

    train_loader, val_loader, test_loader, scalers = prepare_data(cfg)
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[split]

    model = build_model(cfg).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model, map_location=device.type)
    # prefer scalers bundled in the checkpoint over freshly re-fit ones,
    # so evaluation always matches exactly what the model was trained on
    scalers = checkpoint.get("scalers", scalers)

    results = evaluate(model, loader, scalers, device)

    print(f"\n=== Evaluation on {split} split ===")
    print("Regression (SPI3, Humidex):")
    print(f"  MAE : {results['regression']['mae_per_target']}")
    print(f"  RMSE: {results['regression']['rmse_per_target']}")
    print(f"  R2  : {results['regression']['r2_per_target']}")
    print(f"\nDrought class -- accuracy: {results['drought_class']['accuracy']:.4f}, "
          f"macro-F1: {results['drought_class']['macro_f1']:.4f}")
    print(f"Heat class    -- accuracy: {results['heat_class']['accuracy']:.4f}, "
          f"macro-F1: {results['heat_class']['macro_f1']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.checkpoint, args.split, args.config)
