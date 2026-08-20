"""
Sliding-window sequence construction, chronological train/val/test split,
train-fit-only scaling, and the PyTorch Dataset/DataLoader wrapper.

Extracted from notebook cells 62-84.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


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
    Built per district so a window never mixes two districts' timelines.
    The label for each window is the hazard state `horizon` days after the
    window ends -- the model only ever sees the past `window` days to
    predict the future, which is the only leakage constraint that matters.
    """
    X_met, X_veg, X_eng = [], [], []
    y_reg, y_cls, target_dates, districts = [], [], [], []

    for district, g in df.groupby("district"):
        g = g.sort_values("date").reset_index(drop=True)
        met = g[met_cols].values
        veg = g[veg_cols].values
        eng = g[eng_cols].values
        reg = g[target_reg].values
        cls = g[target_cls].values
        dates = g["date"].values

        n = len(g)
        last_start = n - window - horizon + 1
        for t in range(last_start):
            end = t + window
            target_idx = end + horizon - 1

            X_met.append(met[t:end])
            X_veg.append(veg[t:end])
            X_eng.append(eng[t:end])
            y_reg.append(reg[target_idx])
            y_cls.append(cls[target_idx])
            target_dates.append(dates[target_idx])
            districts.append(district)

    return (
        np.array(X_met, dtype=np.float32),
        np.array(X_veg, dtype=np.float32),
        np.array(X_eng, dtype=np.float32),
        np.array(y_reg, dtype=np.float32),
        np.array(y_cls, dtype=np.int64),
        np.array(target_dates),
        np.array(districts),
    )


def chronological_split(target_dates, train_years, val_years, test_years):
    """
    train_years/val_years/test_years are [start, end] inclusive pairs,
    e.g. [2010, 2022]. Split by the target date's year (not a random
    shuffle) so the model is always evaluated on data strictly after
    what it trained on.
    """
    target_years = pd.to_datetime(target_dates).year
    train_mask = (target_years >= train_years[0]) & (target_years <= train_years[1])
    val_mask = (target_years >= val_years[0]) & (target_years <= val_years[1])
    test_mask = (target_years >= test_years[0]) & (target_years <= test_years[1])
    return train_mask, val_mask, test_mask


def subset_arrays(mask, X_met, X_veg, X_eng, y_reg, y_cls):
    return X_met[mask], X_veg[mask], X_eng[mask], y_reg[mask], y_cls[mask]


from pathlib import Path
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


def fit_scaler_3d(X: np.ndarray) -> StandardScaler:
    """
    Fit a StandardScaler on a 3D sequence array.

    X shape:
        (samples, timesteps, features)

    The scaler is fitted across all training samples and timesteps,
    separately for each feature.
    """
    n, t, f = X.shape

    scaler = StandardScaler()
    scaler.fit(X.reshape(-1, f))

    return scaler


def apply_scaler_3d(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """
    Apply an already-fitted scaler to a 3D sequence array.
    """
    n, t, f = X.shape

    X_scaled = scaler.transform(X.reshape(-1, f)).reshape(n, t, f)

    return X_scaled.astype(np.float32)


def fit_all_scalers(
    Xm_tr: np.ndarray,
    Xv_tr: np.ndarray,
    Xe_tr: np.ndarray,
    yr_tr: np.ndarray,
    scaler_dir: str = "data/processed/scalers",
):
    """
    Fit all scalers using TRAINING data only.

    Scalers:
        met -> meteorological features
        veg -> vegetation features
        eng -> engineered features
        reg -> regression targets

    The fitted scalers are saved to disk so the exact same
    normalization can be reused during evaluation/inference.
    """

    scaler_dir = Path(scaler_dir)
    scaler_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Fit ONLY on training data
    # ---------------------------------------------------------

    met_scaler = fit_scaler_3d(Xm_tr)
    veg_scaler = fit_scaler_3d(Xv_tr)
    eng_scaler = fit_scaler_3d(Xe_tr)

    reg_scaler = StandardScaler()
    reg_scaler.fit(yr_tr)

    # ---------------------------------------------------------
    # Save fitted scalers
    # ---------------------------------------------------------

    joblib.dump(met_scaler, scaler_dir / "met_scaler.pkl")

    joblib.dump(veg_scaler, scaler_dir / "veg_scaler.pkl")

    joblib.dump(eng_scaler, scaler_dir / "eng_scaler.pkl")

    joblib.dump(reg_scaler, scaler_dir / "reg_scaler.pkl")

    print(f"Scalers saved to: {scaler_dir}")

    return {
        "met": met_scaler,
        "veg": veg_scaler,
        "eng": eng_scaler,
        "reg": reg_scaler,
    }


def load_all_scalers(scaler_dir: str = "data/processed/scalers"):
    """
    Load previously fitted scalers from disk.
    """

    scaler_dir = Path(scaler_dir)

    scalers = {
        "met": joblib.load(scaler_dir / "met_scaler.pkl"),
        "veg": joblib.load(scaler_dir / "veg_scaler.pkl"),
        "eng": joblib.load(scaler_dir / "eng_scaler.pkl"),
        "reg": joblib.load(scaler_dir / "reg_scaler.pkl"),
    }

    return scalers


class HazardDataset(Dataset):
    def __init__(self, X_met, X_veg, X_eng, y_reg, y_cls):
        self.X_met = torch.tensor(X_met)
        self.X_veg = torch.tensor(X_veg)
        self.X_eng = torch.tensor(X_eng)
        self.y_reg = torch.tensor(y_reg)
        self.y_cls = torch.tensor(y_cls)

    def __len__(self):
        return len(self.X_met)

    def __getitem__(self, idx):
        return (
            self.X_met[idx],
            self.X_veg[idx],
            self.X_eng[idx],
            self.y_reg[idx],
            self.y_cls[idx],
        )


def make_loaders(scaled_splits: dict, batch_size: int = 64):
    """
    scaled_splits: dict with keys 'train', 'val', 'test', each a tuple
    (X_met, X_veg, X_eng, y_reg, y_cls) already scaled.
    """
    train_loader = DataLoader(
        HazardDataset(*scaled_splits["train"]), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        HazardDataset(*scaled_splits["val"]), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        HazardDataset(*scaled_splits["test"]), batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader, test_loader
