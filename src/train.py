"""
Training entrypoint for the compound drought/heat-stress forecasting model.

This script trains the complete multi-branch forecasting architecture.

Architecture:
    Meteorological Branch
            \
    Vegetation Branch ----> Branch Fusion ----> Forecasting Heads
            /                              \
    Engineered Feature Branch               \
                                             ├── Regression Head
                                             │     ├── SPI-3
                                             │     └── Humidex
                                             │
                                             ├── Drought Classification Head
                                             │     └── 4 classes
                                             │
                                             └── Heat Classification Head
                                                   └── 4 classes

Training strategy:
    - Chronological train/validation/test split
    - Scaling fitted on training data only
    - Separate class-weighted loss for drought classification
    - Separate class-weighted loss for heat classification
    - MSE regression loss
    - Early stopping based on validation loss
    - Best model checkpoint saved to checkpoints/

Usage:
    python src/train.py --config configs/config.yaml
"""

import argparse
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Allow imports from project root
# ----------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ----------------------------------------------------------------------
# Libraries
# ----------------------------------------------------------------------

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import DataLoader

# ----------------------------------------------------------------------
# Project modules
# ----------------------------------------------------------------------

from src.dataset import (
    chronological_split,
    create_sequences,
    fit_all_scalers,
    apply_scaler_3d,
    make_loaders,
    subset_arrays,
)

from src.models.branch_encoder import BranchEncoder  # noqa: F401

from src.models.fusion import BranchFusionBlock

from src.models.heads import HazardForecastModel

from src.utils import (
    get_device,
    load_config,
    save_checkpoint,
    save_run_metadata,
    set_seed,
)

# ======================================================================
# MODEL
# ======================================================================


def build_model(cfg: dict) -> HazardForecastModel:
    """
    Build the complete multi-branch forecasting model.
    """

    m = cfg["model"]

    # --------------------------------------------------------------
    # Feature dimensions
    # --------------------------------------------------------------

    met_dim = len(cfg["features"]["meteorological"])

    veg_dim = len(cfg["features"]["vegetation"])

    eng_dim = len(cfg["features"]["engineered"])

    # --------------------------------------------------------------
    # Fusion block
    # --------------------------------------------------------------

    fusion_block = BranchFusionBlock(
        met_dim=met_dim,
        veg_dim=veg_dim,
        eng_dim=eng_dim,
        d_model=m["d_model"],
        d_k=m["d_k"],
        num_heads=m["num_heads"],
        gate_hidden=m["gate_hidden"],
        dropout=m["dropout"],
    )

    # --------------------------------------------------------------
    # Complete forecasting model
    # --------------------------------------------------------------

    model = HazardForecastModel(
        branch_fusion_block=fusion_block,
        d_model=m["d_model"],
        n_reg_targets=len(cfg["features"]["target_regression"]),
        n_classes=4,
        dropout=m["dropout"],
    )

    return model


# ======================================================================
# DATA PREPARATION
# ======================================================================


def prepare_data(cfg: dict):
    """
    Prepare chronological train/validation/test datasets.

    Important:
        Scalers are fitted using TRAINING DATA ONLY.
    """

    # --------------------------------------------------------------
    # Load final processed dataset
    # --------------------------------------------------------------

    final_df = pd.read_csv(
        cfg["data"]["final_dataset"],
        parse_dates=["date"],
    )

    final_df = final_df.sort_values(["district", "date"]).reset_index(drop=True)

    print("\nLoaded final dataset:")
    print(f"  Samples : {len(final_df)}")
    print(f"  Columns : {len(final_df.columns)}")

    # --------------------------------------------------------------
    # Create temporal sequences
    # --------------------------------------------------------------

    print("\nCreating temporal sequences...")

    X_met, X_veg, X_eng, y_reg, y_cls, target_dates, districts = create_sequences(
        final_df,
        met_cols=cfg["features"]["meteorological"],
        veg_cols=cfg["features"]["vegetation"],
        eng_cols=cfg["features"]["engineered"],
        target_reg=cfg["features"]["target_regression"],
        target_cls=cfg["features"]["target_classification"],
        window=cfg["sequence"]["window"],
        horizon=cfg["sequence"]["horizon"],
    )

    print("Sequence shapes:")
    print(f"  Meteorological : {X_met.shape}")
    print(f"  Vegetation    : {X_veg.shape}")
    print(f"  Engineered    : {X_eng.shape}")
    print(f"  Regression    : {y_reg.shape}")
    print(f"  Classification: {y_cls.shape}")

    # --------------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------------

    train_mask, val_mask, test_mask = chronological_split(
        target_dates,
        cfg["split"]["train_years"],
        cfg["split"]["val_years"],
        cfg["split"]["test_years"],
    )

    print("\nSequence split:")
    print(f"  Training   : {train_mask.sum()}")
    print(f"  Validation : {val_mask.sum()}")
    print(f"  Testing    : {test_mask.sum()}")

    # --------------------------------------------------------------
    # Training split
    # --------------------------------------------------------------

    (
        Xm_tr,
        Xv_tr,
        Xe_tr,
        yr_tr,
        yc_tr,
    ) = subset_arrays(
        train_mask,
        X_met,
        X_veg,
        X_eng,
        y_reg,
        y_cls,
    )

    # --------------------------------------------------------------
    # Validation split
    # --------------------------------------------------------------

    (
        Xm_va,
        Xv_va,
        Xe_va,
        yr_va,
        yc_va,
    ) = subset_arrays(
        val_mask,
        X_met,
        X_veg,
        X_eng,
        y_reg,
        y_cls,
    )

    # --------------------------------------------------------------
    # Test split
    # --------------------------------------------------------------

    (
        Xm_te,
        Xv_te,
        Xe_te,
        yr_te,
        yc_te,
    ) = subset_arrays(
        test_mask,
        X_met,
        X_veg,
        X_eng,
        y_reg,
        y_cls,
    )

    # --------------------------------------------------------------
    # Fit scalers on TRAINING DATA ONLY
    # --------------------------------------------------------------

    print("\nFitting scalers on training data only...")

    scalers = fit_all_scalers(
        Xm_tr,
        Xv_tr,
        Xe_tr,
        yr_tr,
    )

    # --------------------------------------------------------------
    # Apply scalers
    # --------------------------------------------------------------

    scaled_splits = {
        "train": (
            apply_scaler_3d(
                Xm_tr,
                scalers["met"],
            ),
            apply_scaler_3d(
                Xv_tr,
                scalers["veg"],
            ),
            apply_scaler_3d(
                Xe_tr,
                scalers["eng"],
            ),
            scalers["reg"].transform(yr_tr).astype("float32"),
            yc_tr,
        ),
        "val": (
            apply_scaler_3d(
                Xm_va,
                scalers["met"],
            ),
            apply_scaler_3d(
                Xv_va,
                scalers["veg"],
            ),
            apply_scaler_3d(
                Xe_va,
                scalers["eng"],
            ),
            scalers["reg"].transform(yr_va).astype("float32"),
            yc_va,
        ),
        "test": (
            apply_scaler_3d(
                Xm_te,
                scalers["met"],
            ),
            apply_scaler_3d(
                Xv_te,
                scalers["veg"],
            ),
            apply_scaler_3d(
                Xe_te,
                scalers["eng"],
            ),
            scalers["reg"].transform(yr_te).astype("float32"),
            yc_te,
        ),
    }

    # --------------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------------

    train_loader, val_loader, test_loader = make_loaders(
        scaled_splits,
        batch_size=cfg["train"]["batch_size"],
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        scalers,
    )


# ======================================================================
# CLASS WEIGHTS
# ======================================================================


def compute_class_weights(
    train_loader,
    device,
):
    """
    Calculate independent class weights for the drought and heat
    classification heads.

    IMPORTANT:
        Only TRAINING labels are used.

    Validation and test labels are never used to calculate weights.

    Returns:
        drought_weights
        heat_weights
    """

    drought_labels = []
    heat_labels = []

    # --------------------------------------------------------------
    # Collect training labels
    # --------------------------------------------------------------

    for (
        _,
        _,
        _,
        _,
        y_cls,
    ) in train_loader:

        drought_labels.extend(y_cls[:, 0].numpy())

        heat_labels.extend(y_cls[:, 1].numpy())

    drought_labels = np.asarray(
        drought_labels,
        dtype=np.int64,
    )

    heat_labels = np.asarray(
        heat_labels,
        dtype=np.int64,
    )

    classes = np.array([0, 1, 2, 3])

    # --------------------------------------------------------------
    # Drought class weights
    # --------------------------------------------------------------

    drought_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=drought_labels,
    )

    # --------------------------------------------------------------
    # Heat class weights
    # --------------------------------------------------------------

    heat_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=heat_labels,
    )

    # --------------------------------------------------------------
    # Convert to PyTorch tensors
    # --------------------------------------------------------------

    drought_weights = torch.tensor(
        drought_weights,
        dtype=torch.float32,
        device=device,
    )

    heat_weights = torch.tensor(
        heat_weights,
        dtype=torch.float32,
        device=device,
    )

    # --------------------------------------------------------------
    # Display weights
    # --------------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("CLASS WEIGHTS")
    print("=" * 60)

    print("Drought class weights:")

    for cls, weight in zip(
        classes,
        drought_weights.cpu().numpy(),
    ):
        print(f"  Class {cls}: {weight:.6f}")

    print("\nHeat class weights:")

    for cls, weight in zip(
        classes,
        heat_weights.cpu().numpy(),
    ):
        print(f"  Class {cls}: {weight:.6f}")

    print("=" * 60)

    return (
        drought_weights,
        heat_weights,
    )


# ======================================================================
# EPOCH
# ======================================================================


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    criterion_reg,
    criterion_drought,
    criterion_heat,
    train: bool = True,
):
    """
    Run one training or validation epoch.
    """

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    # --------------------------------------------------------------
    # Gradient context
    # --------------------------------------------------------------

    context = torch.enable_grad() if train else torch.no_grad()

    with context:

        for (
            met,
            veg,
            eng,
            y_reg,
            y_cls,
        ) in loader:

            # --------------------------------------------------
            # Move data to GPU/CPU
            # --------------------------------------------------

            met = met.to(device)
            veg = veg.to(device)
            eng = eng.to(device)

            y_reg = y_reg.to(device)
            y_cls = y_cls.to(device)

            # --------------------------------------------------
            # Clear gradients
            # --------------------------------------------------

            if train:
                optimizer.zero_grad()

            # --------------------------------------------------
            # Forward pass
            # --------------------------------------------------

            out = model(
                met,
                veg,
                eng,
            )

            # --------------------------------------------------
            # Regression loss
            # --------------------------------------------------

            loss_reg = criterion_reg(
                out["reg_out"],
                y_reg,
            )

            # --------------------------------------------------
            # Drought classification loss
            # --------------------------------------------------

            loss_drought = criterion_drought(
                out["drought_logits"],
                y_cls[:, 0],
            )

            # --------------------------------------------------
            # Heat classification loss
            # --------------------------------------------------

            loss_heat = criterion_heat(
                out["heat_logits"],
                y_cls[:, 1],
            )

            # --------------------------------------------------
            # Total multi-task loss
            # --------------------------------------------------

            loss = loss_reg + loss_drought + loss_heat

            # --------------------------------------------------
            # Backpropagation
            # --------------------------------------------------

            if train:

                loss.backward()

                optimizer.step()

            # --------------------------------------------------
            # Accumulate loss
            # --------------------------------------------------

            total_loss += loss.item() * met.size(0)

    return total_loss / len(loader.dataset)


# ======================================================================
# MAIN
# ======================================================================


def main(config_path: str):

    # --------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------

    cfg = load_config(config_path)

    # --------------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------------

    set_seed(cfg["project"]["seed"])

    # --------------------------------------------------------------
    # Device
    # --------------------------------------------------------------

    device = get_device()

    print("\n")
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    print(f"Device: {device}")

    # --------------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader,
        scalers,
    ) = prepare_data(cfg)

    # --------------------------------------------------------------
    # Build model
    # --------------------------------------------------------------

    print("\nBuilding forecasting model...")

    model = build_model(cfg).to(device)

    print(model)

    # --------------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
    )

    # --------------------------------------------------------------
    # Regression loss
    # --------------------------------------------------------------

    criterion_reg = nn.MSELoss()

    # --------------------------------------------------------------
    # Calculate independent class weights
    # --------------------------------------------------------------

    (
        drought_weights,
        heat_weights,
    ) = compute_class_weights(
        train_loader,
        device,
    )

    # --------------------------------------------------------------
    # Separate classification losses
    # --------------------------------------------------------------

    criterion_drought = nn.CrossEntropyLoss(
        weight=drought_weights,
    )

    criterion_heat = nn.CrossEntropyLoss(
        weight=heat_weights,
    )

    # --------------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------------

    best_val_loss = float("inf")

    patience_counter = 0

    checkpoint_dir = Path(cfg["train"]["checkpoint_dir"])

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Training loop
    # --------------------------------------------------------------

    for epoch in range(
        1,
        cfg["train"]["epochs"] + 1,
    ):

        # ==========================================================
        # TRAIN
        # ==========================================================

        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            criterion_reg=criterion_reg,
            criterion_drought=criterion_drought,
            criterion_heat=criterion_heat,
            train=True,
        )

        # ==========================================================
        # VALIDATION
        # ==========================================================

        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            device=device,
            criterion_reg=criterion_reg,
            criterion_drought=criterion_drought,
            criterion_heat=criterion_heat,
            train=False,
        )

        # ----------------------------------------------------------
        # Print progress
        # ----------------------------------------------------------

        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}"
        )

        # ==========================================================
        # CHECKPOINT
        # ==========================================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            patience_counter = 0

            print("  → Validation loss improved. " "Saving checkpoint.")

            save_checkpoint(
                model,
                optimizer,
                epoch,
                metrics={
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                scalers=scalers,
                config=cfg,
                path=str(checkpoint_dir / "best_model.pt"),
            )

        else:

            patience_counter += 1

            print(
                f"  → No improvement "
                f"({patience_counter}/"
                f"{cfg['train']['early_stopping_patience']})"
            )

            # ------------------------------------------------------
            # Early stopping
            # ------------------------------------------------------

            if patience_counter >= cfg["train"]["early_stopping_patience"]:

                print(
                    f"\nEarly stopping at epoch "
                    f"{epoch} "
                    f"(no validation improvement "
                    f"for "
                    f"{cfg['train']['early_stopping_patience']} "
                    f"epochs)"
                )

                break

    # ==================================================================
    # SAVE RUN METADATA
    # ==================================================================

    save_run_metadata(
        {
            "best_val_loss": best_val_loss,
            "config_path": config_path,
            "device": str(device),
            "class_weights": {
                "drought": drought_weights.detach().cpu().numpy().tolist(),
                "heat": heat_weights.detach().cpu().numpy().tolist(),
            },
        },
        str(checkpoint_dir / "run_metadata.json"),
    )

    # ==================================================================
    # COMPLETE
    # ==================================================================

    print("\n")
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(f"Best validation loss: " f"{best_val_loss:.6f}")

    print(f"Best checkpoint: " f"{checkpoint_dir / 'best_model.pt'}")

    print("=" * 60)


# ======================================================================
# ENTRYPOINT
# ======================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
    )

    args = parser.parse_args()

    main(args.config)
