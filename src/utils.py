"""
Shared utilities: reproducibility, device selection, config loading,
and checkpoint save/load helpers.

These wrap the pattern discussed for the two-PC workflow: train on the
university GPU, then load the same checkpoint (state_dict, not the full
model object) on a personal PC that may have no GPU at all.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Seed everything for reproducibility across the two machines."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    scalers: dict,
    config: dict,
    path: str,
) -> None:
    """
    Bundles everything needed to resume training OR run inference on a
    different machine: model weights, optimizer state, the fitted scalers
    (met/veg/eng/reg — StandardScaler objects), the config used, and the
    epoch/metrics for bookkeeping.

    Scalers are pickled via torch.save alongside the weights so you never
    have to worry about shipping a separate .pkl file and getting it
    out of sync with the checkpoint.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "scalers": scalers,   # dict of {"met": scaler, "veg": scaler, "eng": scaler, "reg": scaler}
            "config": config,
        },
        path,
    )
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None, map_location="cpu"):
    """
    map_location='cpu' by default so this works unmodified on a personal
    PC with no CUDA device — the university PC can override with 'cuda'.
    """
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Loaded checkpoint from {path} (epoch {checkpoint.get('epoch')})")
    return checkpoint


def save_run_metadata(metadata: dict, path: str) -> None:
    """Small JSON sidecar for anything you want human-readable (not just in the .pt file)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
