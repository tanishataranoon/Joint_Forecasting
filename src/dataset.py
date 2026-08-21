"""
Sliding-window sequence construction, chronological train/val/test split,
train-fit-only scaling, and the PyTorch Dataset/DataLoader wrapper.

The data pipeline is designed to prevent temporal leakage:

    Raw dataset
        |
        v
    Per-district sliding windows
        |
        v
    Chronological split by target date
        |
        +--> Train
        +--> Validation
        +--> Test
        |
        v
    Fit scalers ONLY on training data
        |
        v
    Apply the same fitted scalers to
    train / validation / test data

Classification labels are NOT scaled.
Regression targets are standardized using a scaler
fitted only on the training regression targets.
"""

import numpy as np
import pandas as pd
import torch
import joblib

from pathlib import Path

from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader, Dataset


# ======================================================================
# SEQUENCE CREATION
# ======================================================================

def create_sequences(
    df: pd.DataFrame,
    met_cols,
    veg_cols,
    eng_cols,
    target_reg,
    target_cls,
    window: int = 30,
    horizon: int = 1,
):
    """
    Create chronological sliding-window sequences independently
    for each district.

    A sequence contains:

        past `window` days
            |
            v
        future target at `horizon`

    Therefore, the model uses historical observations to predict
    a future target.

    Windows are constructed independently for every district so
    that a sequence can never cross from one district into another.

    Parameters
    ----------
    df:
        Final processed dataframe.

    met_cols:
        Meteorological feature columns.

    veg_cols:
        Vegetation feature columns.

    eng_cols:
        Engineered feature columns.

    target_reg:
        Regression target columns.

        Expected:
            SPI3
            Humidex

    target_cls:
        Classification target columns.

        Expected:
            drought_target_7d
            heat_target_7d

    window:
        Number of historical timesteps.

    horizon:
        Forecast horizon in days.

    Returns
    -------
    X_met:
        (N, window, met_features)

    X_veg:
        (N, window, veg_features)

    X_eng:
        (N, window, engineered_features)

    y_reg:
        (N, 2)

    y_cls:
        (N, 2)

    target_dates:
        Target date corresponding to each sequence.

    districts:
        District corresponding to each sequence.
    """

    X_met = []
    X_veg = []
    X_eng = []

    y_reg = []
    y_cls = []

    target_dates = []
    districts = []

    # ------------------------------------------------------------------
    # Process each district independently
    # ------------------------------------------------------------------

    for district, g in df.groupby("district"):

        g = (
            g.sort_values("date")
            .reset_index(drop=True)
        )

        met = g[met_cols].values
        veg = g[veg_cols].values
        eng = g[eng_cols].values

        reg = g[target_reg].values
        cls = g[target_cls].values

        dates = g["date"].values

        n = len(g)

        last_start = (
            n
            - window
            - horizon
            + 1
        )

        # --------------------------------------------------------------
        # Sliding window
        # --------------------------------------------------------------

        for t in range(last_start):

            end = t + window

            target_idx = (
                end
                + horizon
                - 1
            )

            X_met.append(
                met[t:end]
            )

            X_veg.append(
                veg[t:end]
            )

            X_eng.append(
                eng[t:end]
            )

            y_reg.append(
                reg[target_idx]
            )

            y_cls.append(
                cls[target_idx]
            )

            target_dates.append(
                dates[target_idx]
            )

            districts.append(
                district
            )

    return (
        np.array(
            X_met,
            dtype=np.float32,
        ),
        np.array(
            X_veg,
            dtype=np.float32,
        ),
        np.array(
            X_eng,
            dtype=np.float32,
        ),
        np.array(
            y_reg,
            dtype=np.float32,
        ),
        np.array(
            y_cls,
            dtype=np.int64,
        ),
        np.array(
            target_dates
        ),
        np.array(
            districts
        ),
    )


# ======================================================================
# CHRONOLOGICAL SPLIT
# ======================================================================

def chronological_split(
    target_dates,
    train_years,
    val_years,
    test_years,
):
    """
    Perform a chronological train/validation/test split using
    the target date of each sequence.

    Example:

        Train      : 2010-2022
        Validation : 2023-2024
        Test       : 2025-2026

    The split is performed using target years rather than randomly
    shuffling samples.

    This preserves the temporal forecasting setting:

        Past
         |
         v
        Train
         |
         v
        Validation
         |
         v
        Test
    """

    target_years = (
        pd.to_datetime(
            target_dates
        ).year
    )

    train_mask = (
        (target_years >= train_years[0])
        &
        (target_years <= train_years[1])
    )

    val_mask = (
        (target_years >= val_years[0])
        &
        (target_years <= val_years[1])
    )

    test_mask = (
        (target_years >= test_years[0])
        &
        (target_years <= test_years[1])
    )

    return (
        train_mask,
        val_mask,
        test_mask,
    )


# ======================================================================
# SUBSET ARRAYS
# ======================================================================

def subset_arrays(
    mask,
    X_met,
    X_veg,
    X_eng,
    y_reg,
    y_cls,
):
    """
    Extract one chronological split using a boolean mask.
    """

    return (
        X_met[mask],
        X_veg[mask],
        X_eng[mask],
        y_reg[mask],
        y_cls[mask],
    )


# ======================================================================
# 3D FEATURE SCALER
# ======================================================================

def fit_scaler_3d(
    X: np.ndarray,
) -> StandardScaler:
    """
    Fit a StandardScaler on a 3D sequence array.

    Input shape:

        (samples, timesteps, features)

    The scaler is fitted independently for each feature using
    all training samples and timesteps.

    IMPORTANT
    ---------
    This function must only receive TRAINING data.
    """

    n, t, f = X.shape

    scaler = StandardScaler()

    scaler.fit(
        X.reshape(-1, f)
    )

    return scaler


# ======================================================================
# APPLY 3D FEATURE SCALER
# ======================================================================

def apply_scaler_3d(
    X: np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    """
    Apply an already-fitted StandardScaler to a 3D sequence array.
    """

    n, t, f = X.shape

    X_scaled = (
        scaler
        .transform(
            X.reshape(-1, f)
        )
        .reshape(
            n,
            t,
            f,
        )
    )

    return X_scaled.astype(
        np.float32
    )


# ======================================================================
# FIT ALL SCALERS
# ======================================================================

def fit_all_scalers(
    Xm_tr: np.ndarray,
    Xv_tr: np.ndarray,
    Xe_tr: np.ndarray,
    yr_tr: np.ndarray,
    scaler_dir: str = "data/processed/scalers",
):
    """
    Fit all normalization scalers using TRAINING DATA ONLY.

    Feature scalers:
        met -> meteorological features
        veg -> vegetation features
        eng -> engineered features

    Target scaler:
        reg -> SPI3 + Humidex

    The fitted scalers are saved to disk and can later be reused
    during evaluation and inference.

    Classification targets are NOT scaled.
    """

    scaler_dir = Path(
        scaler_dir
    )

    scaler_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Fit feature scalers
    # ------------------------------------------------------------------

    met_scaler = fit_scaler_3d(
        Xm_tr
    )

    veg_scaler = fit_scaler_3d(
        Xv_tr
    )

    eng_scaler = fit_scaler_3d(
        Xe_tr
    )

    # ------------------------------------------------------------------
    # Fit regression target scaler
    # ------------------------------------------------------------------

    reg_scaler = StandardScaler()

    reg_scaler.fit(
        yr_tr
    )

    # ------------------------------------------------------------------
    # Save scalers
    # ------------------------------------------------------------------

    joblib.dump(
        met_scaler,
        scaler_dir / "met_scaler.pkl",
    )

    joblib.dump(
        veg_scaler,
        scaler_dir / "veg_scaler.pkl",
    )

    joblib.dump(
        eng_scaler,
        scaler_dir / "eng_scaler.pkl",
    )

    joblib.dump(
        reg_scaler,
        scaler_dir / "reg_scaler.pkl",
    )

    print(
        f"Scalers saved to: {scaler_dir}"
    )

    return {
        "met": met_scaler,
        "veg": veg_scaler,
        "eng": eng_scaler,
        "reg": reg_scaler,
    }


# ======================================================================
# LOAD ALL SCALERS
# ======================================================================

def load_all_scalers(
    scaler_dir: str = "data/processed/scalers",
):
    """
    Load previously fitted scalers from disk.

    These scalers should have been fitted using training data only.
    """

    scaler_dir = Path(
        scaler_dir
    )

    scalers = {
        "met": joblib.load(
            scaler_dir
            / "met_scaler.pkl"
        ),

        "veg": joblib.load(
            scaler_dir
            / "veg_scaler.pkl"
        ),

        "eng": joblib.load(
            scaler_dir
            / "eng_scaler.pkl"
        ),

        "reg": joblib.load(
            scaler_dir
            / "reg_scaler.pkl"
        ),
    }

    return scalers


# ======================================================================
# PYTORCH DATASET
# ======================================================================

class HazardDataset(Dataset):
    """
    PyTorch Dataset for the multi-branch hazard forecasting model.
    """

    def __init__(
        self,
        X_met,
        X_veg,
        X_eng,
        y_reg,
        y_cls,
    ):

        self.X_met = torch.tensor(
            X_met,
            dtype=torch.float32,
        )

        self.X_veg = torch.tensor(
            X_veg,
            dtype=torch.float32,
        )

        self.X_eng = torch.tensor(
            X_eng,
            dtype=torch.float32,
        )

        self.y_reg = torch.tensor(
            y_reg,
            dtype=torch.float32,
        )

        self.y_cls = torch.tensor(
            y_cls,
            dtype=torch.long,
        )

    def __len__(self):
        return len(
            self.X_met
        )

    def __getitem__(self, idx):

        return (
            self.X_met[idx],
            self.X_veg[idx],
            self.X_eng[idx],
            self.y_reg[idx],
            self.y_cls[idx],
        )


# ======================================================================
# DATALOADERS
# ======================================================================

def make_loaders(
    scaled_splits: dict,
    batch_size: int = 64,
):
    """
    Create PyTorch DataLoaders.

    scaled_splits must contain:

        train
        val
        test

    Each split must contain:

        X_met
        X_veg
        X_eng
        y_reg
        y_cls
    """

    train_loader = DataLoader(
        HazardDataset(
            *scaled_splits["train"]
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        HazardDataset(
            *scaled_splits["val"]
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        HazardDataset(
            *scaled_splits["test"]
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )