"""
BiLSTM + Temporal Attention training pipeline.

Complete model:

    Input sequences
        |
        +--> Meteorological BranchEncoder
        |
        +--> Vegetation BranchEncoder
        |
        +--> Engineered BranchEncoder
        |
        v
    Adaptive Cross-Attention + Dynamic Temporal Gating
        |
        v
    BiLSTM + Temporal Attention
        |
        v
    MultiTaskHeads
        |
        +--> SPI3 regression
        +--> Humidex regression
        +--> Drought classification
        +--> Heat classification

Training split:
    2010-2022

Validation:
    2023-2024

Testing:
    2025-2026

The split is chronological and is performed using target dates.
"""

# ======================================================================
# IMPORTS
# ======================================================================

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml


# ======================================================================
# PROJECT ROOT
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ======================================================================
# PROJECT IMPORTS
# ======================================================================

from src.dataset import (
    create_sequences,
    chronological_split,
    subset_arrays,
    fit_all_scalers,
    apply_scaler_3d,
    make_loaders,
)

from src.models.fusion import BranchFusionBlock

from src.models.temporal_backbones.Bilstm_attention.bilstm_attention import (
    BiLSTMAttentionBackbone,
)

from src.models.heads import MultiTaskHeads


# ======================================================================
# REPRODUCIBILITY
# ======================================================================


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducible training.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior where possible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======================================================================
# CONFIGURATION
# ======================================================================


def load_config(config_path: str = "configs/config.yaml"):
    """
    Load YAML configuration relative to the project root.
    """

    config_path = Path(config_path)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found:\n{config_path}"
        )

    print(f"Loading configuration: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


# ======================================================================
# COMPLETE BILSTM MODEL
# ======================================================================


class BiLSTMForecastModel(nn.Module):
    """
    Complete forecasting model for the BiLSTM + Attention experiment.

    Pipeline:

        BranchFusionBlock
            ->
        BiLSTM + Temporal Attention
            ->
        MultiTaskHeads
    """

    def __init__(
        self,
        met_dim: int,
        veg_dim: int,
        eng_dim: int,
        d_model: int,
        d_k: int,
        num_heads: int,
        gate_hidden: int,
        fusion_dropout: float,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        head_dropout: float = 0.30,
    ):
        super().__init__()

        # --------------------------------------------------------------
        # Branch encoders + adaptive fusion
        # --------------------------------------------------------------

        self.fusion = BranchFusionBlock(
            met_dim=met_dim,
            veg_dim=veg_dim,
            eng_dim=eng_dim,
            d_model=d_model,
            d_k=d_k,
            num_heads=num_heads,
            gate_hidden=gate_hidden,
            dropout=fusion_dropout,
        )

        # --------------------------------------------------------------
        # BiLSTM + temporal attention
        # --------------------------------------------------------------

        self.temporal_backbone = BiLSTMAttentionBackbone(
            d_model=d_model,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            dropout=fusion_dropout,
        )
        self.temporal_dropout = nn.Dropout(p=0.3)

        # --------------------------------------------------------------
        # Multi-task heads
        # --------------------------------------------------------------

        self.heads = MultiTaskHeads(
            in_dim=self.temporal_backbone.out_dim,
            n_reg_targets=2,
            n_classes=4,
            dropout=head_dropout,
        )

    def forward(
        self,
        x_met: torch.Tensor,
        x_veg: torch.Tensor,
        x_eng: torch.Tensor,
    ):
        """
        Parameters
        ----------
        x_met:
            (B, T, met_dim)

        x_veg:
            (B, T, veg_dim)

        x_eng:
            (B, T, eng_dim)

        Returns
        -------
        Dictionary containing:

            fused
            cross_attention
            fusion_gates
            pooled
            temporal_attention
            reg_out
            drought_logits
            heat_logits
        """

        # --------------------------------------------------------------
        # 1. Branch encoding + adaptive fusion
        # --------------------------------------------------------------

        fused, cross_attention, fusion_gates = self.fusion(
            x_met,
            x_veg,
            x_eng,
        )

        # --------------------------------------------------------------
        # 2. Temporal backbone
        # --------------------------------------------------------------

        temporal_output = self.temporal_backbone(fused)
        pooled = temporal_output["pooled"]

        pooled = self.temporal_dropout(pooled)

        temporal_attention = temporal_output["temporal_attn"]
        reg_out, drought_logits, heat_logits = self.heads(pooled)

        # --------------------------------------------------------------
        # 3. Multi-task prediction heads
        # --------------------------------------------------------------

        reg_out, drought_logits, heat_logits = self.heads(pooled)

        return {
            "fused": fused,
            "cross_attention": cross_attention,
            "fusion_gates": fusion_gates,
            "pooled": pooled,
            "temporal_attention": temporal_attention,
            "reg_out": reg_out,
            "drought_logits": drought_logits,
            "heat_logits": heat_logits,
        }


# ======================================================================
# MULTI-TASK LOSS
# ======================================================================


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss:

        total =
            regression_weight * regression_loss
            + drought_weight * drought_classification_loss
            + heat_weight * heat_classification_loss

    Regression:
        MSE

    Classification:
        Weighted CrossEntropyLoss
    """

    def __init__(
        self,
        regression_weight: float = 1.0,
        drought_weight: float = 1.0,
        heat_weight: float = 1.0,
        drought_class_weights: torch.Tensor = None,
        heat_class_weights: torch.Tensor = None,
    ):
        super().__init__()

        self.regression_weight = regression_weight
        self.drought_weight = drought_weight
        self.heat_weight = heat_weight

        # --------------------------------------------------------------
        # Regression
        # --------------------------------------------------------------

        self.regression_loss = nn.MSELoss()

        # --------------------------------------------------------------
        # Drought classification
        # --------------------------------------------------------------

        self.drought_loss = nn.CrossEntropyLoss(
            weight=drought_class_weights
        )

        # --------------------------------------------------------------
        # Heat classification
        # --------------------------------------------------------------

        self.heat_loss = nn.CrossEntropyLoss(
            weight=heat_class_weights
        )

    def forward(
        self,
        outputs,
        y_reg,
        y_cls,
    ):
        """
        Parameters
        ----------
        outputs:
            Model output dictionary.

        y_reg:
            (B, 2)

        y_cls:
            (B, 2)

            y_cls[:, 0] = drought class
            y_cls[:, 1] = heat class
        """

        reg_out = outputs["reg_out"]

        drought_logits = outputs["drought_logits"]

        heat_logits = outputs["heat_logits"]

        # --------------------------------------------------------------
        # Regression
        # --------------------------------------------------------------

        reg_loss = self.regression_loss(
            reg_out,
            y_reg,
        )

        # --------------------------------------------------------------
        # Drought classification
        # --------------------------------------------------------------

        drought_target = y_cls[:, 0].long()

        drought_loss = self.drought_loss(
            drought_logits,
            drought_target,
        )

        # --------------------------------------------------------------
        # Heat classification
        # --------------------------------------------------------------

        heat_target = y_cls[:, 1].long()

        heat_loss = self.heat_loss(
            heat_logits,
            heat_target,
        )

        # --------------------------------------------------------------
        # Combined loss
        # --------------------------------------------------------------

        total_loss = (
            self.regression_weight * reg_loss
            + self.drought_weight * drought_loss
            + self.heat_weight * heat_loss
        )

        return {
            "total": total_loss,
            "regression": reg_loss,
            "drought": drought_loss,
            "heat": heat_loss,
        }


# ======================================================================
# CLASS WEIGHTS
# ======================================================================


def compute_class_weights(
    y_cls: np.ndarray,
    n_classes: int = 4,
):
    """
    Compute balanced class weights from TRAINING labels only.

    Weight for class c:

        weight_c = N / (K * N_c)

    where:

        N   = total number of training samples
        K   = number of classes
        N_c = number of training samples belonging to class c

    Parameters
    ----------
    y_cls:
        Classification targets with shape (N, 2).

        Column 0 = drought class
        Column 1 = heat class

    n_classes:
        Number of classes.

    Returns
    -------
    drought_weights:
        Tensor of shape (n_classes,)

    heat_weights:
        Tensor of shape (n_classes,)
    """

    def calculate_single_weights(targets):

        targets = np.asarray(targets).astype(np.int64)

        counts = np.bincount(
            targets,
            minlength=n_classes,
        )

        total = len(targets)

        weights = np.zeros(
            n_classes,
            dtype=np.float32,
        )

        for class_id in range(n_classes):

            if counts[class_id] > 0:

                weights[class_id] = (
                    total
                    / (n_classes * counts[class_id])
                )

            else:

                weights[class_id] = 0.0

        return weights, counts

    # --------------------------------------------------------------
    # Drought
    # --------------------------------------------------------------

    drought_weights, drought_counts = calculate_single_weights(
        y_cls[:, 0]
    )

    # --------------------------------------------------------------
    # Heat
    # --------------------------------------------------------------

    heat_weights, heat_counts = calculate_single_weights(
        y_cls[:, 1]
    )

    # --------------------------------------------------------------
    # Print
    # --------------------------------------------------------------

    print()
    print("=" * 78)
    print("TRAINING-ONLY CLASS WEIGHTS")
    print("=" * 78)

    print()
    print("DROUGHT")
    print("-" * 78)

    for class_id in range(n_classes):

        print(
            f"  Class {class_id}: "
            f"count={drought_counts[class_id]:,} | "
            f"weight={drought_weights[class_id]:.6f}"
        )

    print()
    print("HEAT")
    print("-" * 78)

    for class_id in range(n_classes):

        print(
            f"  Class {class_id}: "
            f"count={heat_counts[class_id]:,} | "
            f"weight={heat_weights[class_id]:.6f}"
        )

    print("=" * 78)

    return (
        torch.tensor(
            drought_weights,
            dtype=torch.float32,
        ),
        torch.tensor(
            heat_weights,
            dtype=torch.float32,
        ),
    )


# ======================================================================
# ONE TRAINING EPOCH
# ======================================================================


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    total_loss = 0.0
    total_reg = 0.0
    total_drought = 0.0
    total_heat = 0.0

    total_samples = 0

    for (
        x_met,
        x_veg,
        x_eng,
        y_reg,
        y_cls,
    ) in loader:

        # --------------------------------------------------------------
        # Move to device
        # --------------------------------------------------------------

        x_met = x_met.to(device)
        x_veg = x_veg.to(device)
        x_eng = x_eng.to(device)

        y_reg = y_reg.to(device)
        y_cls = y_cls.to(device)

        # --------------------------------------------------------------
        # Clear gradients
        # --------------------------------------------------------------

        optimizer.zero_grad(set_to_none=True)

        # --------------------------------------------------------------
        # Forward
        # --------------------------------------------------------------

        outputs = model(
            x_met,
            x_veg,
            x_eng,
        )

        # --------------------------------------------------------------
        # Loss
        # --------------------------------------------------------------

        losses = criterion(
            outputs,
            y_reg,
            y_cls,
        )

        # --------------------------------------------------------------
        # Backward
        # --------------------------------------------------------------

        losses["total"].backward()

        # --------------------------------------------------------------
        # Gradient clipping
        # --------------------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        # --------------------------------------------------------------
        # Update
        # --------------------------------------------------------------

        optimizer.step()

        # --------------------------------------------------------------
        # Accumulate
        # --------------------------------------------------------------

        batch_size = x_met.size(0)

        total_loss += (
            losses["total"].item() * batch_size
        )

        total_reg += (
            losses["regression"].item() * batch_size
        )

        total_drought += (
            losses["drought"].item() * batch_size
        )

        total_heat += (
            losses["heat"].item() * batch_size
        )

        total_samples += batch_size

    return {
        "total": total_loss / total_samples,
        "regression": total_reg / total_samples,
        "drought": total_drought / total_samples,
        "heat": total_heat / total_samples,
    }


# ======================================================================
# VALIDATION
# ======================================================================


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0
    total_reg = 0.0
    total_drought = 0.0
    total_heat = 0.0

    total_samples = 0

    for (
        x_met,
        x_veg,
        x_eng,
        y_reg,
        y_cls,
    ) in loader:

        # --------------------------------------------------------------
        # Move to device
        # --------------------------------------------------------------

        x_met = x_met.to(device)
        x_veg = x_veg.to(device)
        x_eng = x_eng.to(device)

        y_reg = y_reg.to(device)
        y_cls = y_cls.to(device)

        # --------------------------------------------------------------
        # Forward
        # --------------------------------------------------------------

        outputs = model(
            x_met,
            x_veg,
            x_eng,
        )

        # --------------------------------------------------------------
        # Loss
        # --------------------------------------------------------------

        losses = criterion(
            outputs,
            y_reg,
            y_cls,
        )

        # --------------------------------------------------------------
        # Accumulate
        # --------------------------------------------------------------

        batch_size = x_met.size(0)

        total_loss += (
            losses["total"].item() * batch_size
        )

        total_reg += (
            losses["regression"].item() * batch_size
        )

        total_drought += (
            losses["drought"].item() * batch_size
        )

        total_heat += (
            losses["heat"].item() * batch_size
        )

        total_samples += batch_size

    return {
        "total": total_loss / total_samples,
        "regression": total_reg / total_samples,
        "drought": total_drought / total_samples,
        "heat": total_heat / total_samples,
    }


# ======================================================================
# SAVE CHECKPOINT
# ======================================================================


def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_loss,
    val_loss,
    path,
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )


# ======================================================================
# MAIN
# ======================================================================


def main():

    print("=" * 78)
    print("BiLSTM + TEMPORAL ATTENTION TRAINING")
    print("=" * 78)

    # ==================================================================
    # CONFIGURATION
    # ==================================================================

    config = load_config(
        "configs/config.yaml"
    )

    seed = config["project"]["seed"]

    set_seed(seed)

    # ==================================================================
    # DEVICE
    # ==================================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # ==================================================================
    # CONFIGURATION VALUES
    # ==================================================================

    met_cols = config["features"]["meteorological"]

    veg_cols = config["features"]["vegetation"]

    eng_cols = config["features"]["engineered"]

    target_reg = config["features"]["target_regression"]

    target_cls = config["features"]["target_classification"]

    window = config["sequence"]["window"]

    horizon = config["sequence"]["horizon"]

    d_model = config["model"]["d_model"]

    d_k = config["model"]["d_k"]

    num_heads = config["model"]["num_heads"]

    gate_hidden = config["model"]["gate_hidden"]

    dropout = config["model"]["dropout"]

    batch_size = config["train"]["batch_size"]

    epochs = config["train"]["epochs"]

    learning_rate = config["train"]["lr"]

    patience = config["train"]["early_stopping_patience"]

    checkpoint_dir = Path(
        config["train"]["checkpoint_dir"]
    )

    # ==================================================================
    # CONFIGURATION SUMMARY
    # ==================================================================

    print()
    print("Configuration")
    print("-" * 78)

    print(
        f"Meteorological features : {len(met_cols)}"
    )

    print(
        f"Vegetation features     : {len(veg_cols)}"
    )

    print(
        f"Engineered features     : {len(eng_cols)}"
    )

    print(
        f"Window                  : {window}"
    )

    print(
        f"Horizon                 : {horizon} days"
    )

    print(
        f"D_MODEL                 : {d_model}"
    )

    print(
        f"D_K                     : {d_k}"
    )

    print(
        f"Attention heads         : {num_heads}"
    )

    print(
        f"Gate hidden             : {gate_hidden}"
    )

    print(
        f"Batch size              : {batch_size}"
    )

    print(
        f"Maximum epochs          : {epochs}"
    )

    print(
        f"Learning rate           : {learning_rate}"
    )

    print(
        f"Early stopping patience : {patience}"
    )

    print("-" * 78)

    # ==================================================================
    # LOAD DATASET
    # ==================================================================

    dataset_path = Path(
        config["data"]["final_dataset"]
    )

    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    if not dataset_path.exists():

        raise FileNotFoundError(
            f"Final dataset not found:\n{dataset_path}"
        )

    print()
    print(
        f"Loading dataset: {dataset_path}"
    )

    df = pd.read_csv(
        dataset_path
    )

    # ==================================================================
    # DATE
    # ==================================================================

    if "date" not in df.columns:

        raise KeyError(
            "Dataset does not contain required 'date' column."
        )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # ==================================================================
    # REQUIRED COLUMNS
    # ==================================================================

    required_columns = (
        met_cols
        + veg_cols
        + eng_cols
        + target_reg
        + target_cls
        + [
            "district",
            "date",
        ]
    )

    missing_columns = [
        c
        for c in required_columns
        if c not in df.columns
    ]

    if missing_columns:

        raise KeyError(
            "The following required columns are missing:\n"
            + "\n".join(missing_columns)
        )

    # ==================================================================
    # CREATE SLIDING WINDOWS
    # ==================================================================

    print()
    print(
        "Creating sliding-window sequences..."
    )

    (
        X_met,
        X_veg,
        X_eng,
        y_reg,
        y_cls,
        target_dates,
        districts,
    ) = create_sequences(
        df=df,
        met_cols=met_cols,
        veg_cols=veg_cols,
        eng_cols=eng_cols,
        target_reg=target_reg,
        target_cls=target_cls,
        window=window,
        horizon=horizon,
    )

    print(
        f"Meteorological sequences : {X_met.shape}"
    )

    print(
        f"Vegetation sequences     : {X_veg.shape}"
    )

    print(
        f"Engineered sequences     : {X_eng.shape}"
    )

    print(
        f"Regression targets       : {y_reg.shape}"
    )

    print(
        f"Classification targets   : {y_cls.shape}"
    )

    # ==================================================================
    # CHRONOLOGICAL SPLIT
    # ==================================================================

    train_years = config["split"]["train_years"]

    val_years = config["split"]["val_years"]

    test_years = config["split"]["test_years"]

    (
        train_mask,
        val_mask,
        test_mask,
    ) = chronological_split(
        target_dates=target_dates,
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
    )

    print()
    print("Chronological split")
    print("-" * 78)

    print(
        f"Train {train_years}: "
        f"{train_mask.sum():,} samples"
    )

    print(
        f"Validation {val_years}: "
        f"{val_mask.sum():,} samples"
    )

    print(
        f"Test {test_years}: "
        f"{test_mask.sum():,} samples"
    )

    # ==================================================================
    # EXTRACT SPLITS
    # ==================================================================

    train_arrays = subset_arrays(
        train_mask,
        X_met,
        X_veg,
        X_eng,
        y_reg,
        y_cls,
    )

    val_arrays = subset_arrays(
        val_mask,
        X_met,
        X_veg,
        X_eng,
        y_reg,
        y_cls,
    )

    test_arrays = subset_arrays(
        test_mask,
        X_met,
        X_veg,
        X_eng,
        y_reg,
        y_cls,
    )

    (
        Xm_tr,
        Xv_tr,
        Xe_tr,
        yr_tr,
        yc_tr,
    ) = train_arrays

    (
        Xm_val,
        Xv_val,
        Xe_val,
        yr_val,
        yc_val,
    ) = val_arrays

    (
        Xm_test,
        Xv_test,
        Xe_test,
        yr_test,
        yc_test,
    ) = test_arrays

    # ==================================================================
    # CLASS WEIGHTS
    # ==================================================================

    drought_class_weights, heat_class_weights = (
        compute_class_weights(
            yc_tr,
            n_classes=4,
        )
    )

    # ==================================================================
    # FIT SCALERS ON TRAINING DATA ONLY
    # ==================================================================

    print()
    print(
        "Fitting training-only scalers..."
    )

    scalers = fit_all_scalers(
        Xm_tr,
        Xv_tr,
        Xe_tr,
        yr_tr,
        scaler_dir=PROJECT_ROOT / "data/processed/scalers",
    )

    print(
        "Scalers saved to: "
        f"{PROJECT_ROOT / 'data/processed/scalers'}"
    )

    # ==================================================================
    # APPLY INPUT SCALERS
    # ==================================================================

    Xm_tr = apply_scaler_3d(
        Xm_tr,
        scalers["met"],
    )

    Xv_tr = apply_scaler_3d(
        Xv_tr,
        scalers["veg"],
    )

    Xe_tr = apply_scaler_3d(
        Xe_tr,
        scalers["eng"],
    )

    Xm_val = apply_scaler_3d(
        Xm_val,
        scalers["met"],
    )

    Xv_val = apply_scaler_3d(
        Xv_val,
        scalers["veg"],
    )

    Xe_val = apply_scaler_3d(
        Xe_val,
        scalers["eng"],
    )

    Xm_test = apply_scaler_3d(
        Xm_test,
        scalers["met"],
    )

    Xv_test = apply_scaler_3d(
        Xv_test,
        scalers["veg"],
    )

    Xe_test = apply_scaler_3d(
        Xe_test,
        scalers["eng"],
    )

    # ==================================================================
    # SCALE REGRESSION TARGETS
    # ==================================================================

    yr_tr = scalers["reg"].transform(
        yr_tr
    ).astype(np.float32)

    yr_val = scalers["reg"].transform(
        yr_val
    ).astype(np.float32)

    yr_test = scalers["reg"].transform(
        yr_test
    ).astype(np.float32)

    # ==================================================================
    # BUILD LOADERS
    # ==================================================================

    scaled_splits = {
        "train": (
            Xm_tr,
            Xv_tr,
            Xe_tr,
            yr_tr,
            yc_tr,
        ),
        "val": (
            Xm_val,
            Xv_val,
            Xe_val,
            yr_val,
            yc_val,
        ),
        "test": (
            Xm_test,
            Xv_test,
            Xe_test,
            yr_test,
            yc_test,
        ),
    }

    (
        train_loader,
        val_loader,
        test_loader,
    ) = make_loaders(
        scaled_splits,
        batch_size=batch_size,
    )

    # ==================================================================
    # BUILD MODEL
    # ==================================================================

    print()
    print(
        "Building BiLSTM + Attention model..."
    )

    model = BiLSTMForecastModel(
        met_dim=len(met_cols),
        veg_dim=len(veg_cols),
        eng_dim=len(eng_cols),
        d_model=d_model,
        d_k=d_k,
        num_heads=num_heads,
        gate_hidden=gate_hidden,
        fusion_dropout=0.20,
        lstm_hidden=64,
        lstm_layers=1,
        head_dropout=0.30,
    ).to(device)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{total_parameters:,}"
    )

    print(
        "DEBUG 1: Model construction complete",
        flush=True,
    )

    # ==================================================================
    # LOSS
    # ==================================================================

    loss_weights = config["model"]["loss_weights"]

    print(
        f"DEBUG 2: Loss weights loaded: "
        f"{loss_weights}",
        flush=True,
    )

    # IMPORTANT:
    # The dynamically calculated training-only class weights
    # are passed here.
    criterion = MultiTaskLoss(
        regression_weight=loss_weights["regression"],
        drought_weight=loss_weights["drought"],
        heat_weight=loss_weights["heat"],
        drought_class_weights=drought_class_weights.to(device),
        heat_class_weights=heat_class_weights.to(device),
    ).to(device)

    print(
        "DEBUG 3: Criterion created",
        flush=True,
    )

    # ==================================================================
    # OPTIMIZER
    # ==================================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    print(
        "DEBUG 4: Optimizer created",
        flush=True,
    )

    # ==================================================================
    # LEARNING-RATE SCHEDULER
    # ==================================================================

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    print(
        "DEBUG 5: Scheduler created",
        flush=True,
    )

    # ==================================================================
    # TRAINING HISTORY
    # ==================================================================

    history = {
        "train_total": [],
        "train_regression": [],
        "train_drought": [],
        "train_heat": [],
        "val_total": [],
        "val_regression": [],
        "val_drought": [],
        "val_heat": [],
        "learning_rate": [],
    }

    # ==================================================================
    # EARLY STOPPING
    # ==================================================================

    best_val_loss = float("inf")

    best_epoch = 0

    epochs_without_improvement = 0

    checkpoint_dir = (
        PROJECT_ROOT / checkpoint_dir
        if not checkpoint_dir.is_absolute()
        else checkpoint_dir
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_checkpoint = (
        checkpoint_dir
        / "bilstm_attention_best.pt"
    )

    # ==================================================================
    # START TRAINING
    # ==================================================================

    print()
    print("=" * 78)
    print("STARTING TRAINING")
    print("=" * 78)

    print(
        "DEBUG 6: About to enter training loop",
        flush=True,
    )

    for epoch in range(
        1,
        epochs + 1,
    ):

        # --------------------------------------------------------------
        # Training
        # --------------------------------------------------------------

        train_losses = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        val_losses = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        # --------------------------------------------------------------
        # Scheduler
        # --------------------------------------------------------------

        scheduler.step(
            val_losses["total"]
        )

        current_lr = optimizer.param_groups[0]["lr"]

        # --------------------------------------------------------------
        # Store history
        # --------------------------------------------------------------

        history["train_total"].append(
            train_losses["total"]
        )

        history["train_regression"].append(
            train_losses["regression"]
        )

        history["train_drought"].append(
            train_losses["drought"]
        )

        history["train_heat"].append(
            train_losses["heat"]
        )

        history["val_total"].append(
            val_losses["total"]
        )

        history["val_regression"].append(
            val_losses["regression"]
        )

        history["val_drought"].append(
            val_losses["drought"]
        )

        history["val_heat"].append(
            val_losses["heat"]
        )

        history["learning_rate"].append(
            current_lr
        )

        # --------------------------------------------------------------
        # Print epoch
        # --------------------------------------------------------------

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"Train Loss: "
            f"{train_losses['total']:.5f} | "
            f"Val Loss: "
            f"{val_losses['total']:.5f} | "
            f"Reg: "
            f"{val_losses['regression']:.5f} | "
            f"Drought: "
            f"{val_losses['drought']:.5f} | "
            f"Heat: "
            f"{val_losses['heat']:.5f} | "
            f"LR: "
            f"{current_lr:.2e}",
            flush=True,
        )

        # --------------------------------------------------------------
        # Save best model
        # --------------------------------------------------------------

        if val_losses["total"] < best_val_loss:

            best_val_loss = val_losses["total"]

            best_epoch = epoch

            epochs_without_improvement = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_losses["total"],
                val_loss=val_losses["total"],
                path=best_checkpoint,
            )

            print(
                f"  ✓ New best model saved "
                f"(epoch {epoch})",
                flush=True,
            )

        else:

            epochs_without_improvement += 1

        # --------------------------------------------------------------
        # Early stopping
        # --------------------------------------------------------------

        if (
            epochs_without_improvement
            >= patience
        ):

            print()

            print(
                f"Early stopping triggered "
                f"after {epoch} epochs."
            )

            break

    # ==================================================================
    # SAVE TRAINING HISTORY
    # ==================================================================

    history_path = (
        checkpoint_dir
        / "bilstm_attention_history.csv"
    )

    history_df = pd.DataFrame(
        history
    )

    history_df.insert(
        0,
        "epoch",
        np.arange(
            1,
            len(history_df) + 1,
        ),
    )

    history_df.to_csv(
        history_path,
        index=False,
    )

    # ==================================================================
    # LOAD BEST MODEL
    # ==================================================================

    print()
    print(
        "Loading best checkpoint..."
    )

    if not best_checkpoint.exists():

        raise FileNotFoundError(
            "Best checkpoint was not created. "
            "Training may have failed before the "
            "first validation epoch."
        )

    checkpoint = torch.load(
        best_checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================

    print()
    print("=" * 78)
    print("BILSTM TRAINING COMPLETE")
    print("=" * 78)

    print(
        f"Best epoch     : {best_epoch}"
    )

    print(
        f"Best val loss  : {best_val_loss:.6f}"
    )

    print(
        f"Checkpoint     : {best_checkpoint}"
    )

    print(
        f"History        : {history_path}"
    )

    print("=" * 78)

    print()

    print(
        "Next step: run evaluate_bilstm.py "
        "on the held-out 2025-2026 test set."
    )


# ======================================================================
# ENTRY POINT
# ======================================================================


if __name__ == "__main__":
    main()