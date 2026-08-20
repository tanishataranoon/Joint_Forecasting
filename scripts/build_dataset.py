"""
Build the forecasting dataset.

Two modes are supported:

1. FULL PIPELINE
   NASA POWER + MODIS -> alignment -> QC -> feature engineering
   -> future targets -> chronological split

   python scripts/build_dataset.py --config configs/config.yaml

2. FROM CLEANED DATA
   Existing cleaned_data.csv -> feature engineering -> future targets
   -> chronological split

   python scripts/build_dataset.py --config configs/config.yaml --from-cleaned

The --from-cleaned mode is intended for development and debugging.
It avoids downloading NASA POWER / MODIS and avoids running QC again.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import ee
from dotenv import load_dotenv

# ----------------------------------------------------------------------
# Project path
# ----------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

load_dotenv()

from src.data.fetch import (
    fetch_all_modis,
    fetch_all_nasa_power,
    align_all_districts,
)

from src.data.qc import run_full_qc_pipeline

from src.data.features import (
    engineer_features,
    add_severity_labels,
)

from src.utils import load_config

# ======================================================================
# CHRONOLOGICAL SPLIT
# ======================================================================


def chronological_split(df: pd.DataFrame, cfg: dict):
    """
    Split data chronologically according to TARGET DATE.

    Train:
        2010-2022

    Validation:
        2023-2024

    Test:
        2025-2026

    The split is performed using the date associated with the input
    sample / target prediction instance.

    No random shuffling is performed.
    """

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    split_cfg = cfg["split"]

    train_start, train_end = split_cfg["train_years"]
    val_start, val_end = split_cfg["val_years"]
    test_start, test_end = split_cfg["test_years"]

    train_mask = (df["date"].dt.year >= train_start) & (df["date"].dt.year <= train_end)

    val_mask = (df["date"].dt.year >= val_start) & (df["date"].dt.year <= val_end)

    test_mask = (df["date"].dt.year >= test_start) & (df["date"].dt.year <= test_end)

    train_df = df.loc[train_mask].copy()
    val_df = df.loc[val_mask].copy()
    test_df = df.loc[test_mask].copy()

    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------

    if len(train_df) == 0:
        raise ValueError("Training split is empty.")

    if len(val_df) == 0:
        raise ValueError("Validation split is empty.")

    if len(test_df) == 0:
        raise ValueError("Test split is empty.")

    # ------------------------------------------------------------------
    # Print split information
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("CHRONOLOGICAL DATA SPLIT")
    print("=" * 70)

    print(f"Training   : {train_start}-{train_end} " f"| samples = {len(train_df)}")

    print(f"Validation : {val_start}-{val_end} " f"| samples = {len(val_df)}")

    print(f"Testing    : {test_start}-{test_end} " f"| samples = {len(test_df)}")

    print("\nDate ranges:")

    print(
        f"Train      : "
        f"{train_df['date'].min().date()} -> "
        f"{train_df['date'].max().date()}"
    )

    print(
        f"Validation : "
        f"{val_df['date'].min().date()} -> "
        f"{val_df['date'].max().date()}"
    )

    print(
        f"Test       : "
        f"{test_df['date'].min().date()} -> "
        f"{test_df['date'].max().date()}"
    )

    print("=" * 70)

    return train_df, val_df, test_df


# ======================================================================
# SAVE SPLITS
# ======================================================================


def save_splits(train_df, val_df, test_df, processed_dir):
    """
    Save chronological train/validation/test datasets.
    """

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "validation.csv"
    test_path = processed_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nSaved split datasets:")

    print(f"Train      : {train_path}")
    print(f"Validation : {val_path}")
    print(f"Test       : {test_path}")


# ======================================================================
# FEATURE + TARGET PROCESSING
# ======================================================================


def process_cleaned_data(clean_df: pd.DataFrame, cfg: dict):
    """
    Process an already-cleaned dataset.

    This is the part that can be rerun without internet access.
    """

    print("\n" + "=" * 70)
    print("PROCESSING EXISTING CLEANED DATA")
    print("=" * 70)

    print("\nInput columns:")
    print(clean_df.columns.tolist())

    print("\nInput shape:")
    print(clean_df.shape)

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    print(
        "\nEngineering features " "(monthly SPI-3, VPD, Humidex, temporal features)..."
    )

    clean_df = engineer_features(clean_df)

    # ------------------------------------------------------------------
    # Keep only model-relevant columns
    # ------------------------------------------------------------------

    final_cols = (
        ["date", "district"]
        + cfg["features"]["meteorological"]
        + cfg["features"]["vegetation"]
        + cfg["features"]["engineered"]
        + [
            "doy_sin",
            "doy_cos",
        ]
    )

    # Remove duplicates while preserving order
    final_cols = list(dict.fromkeys(final_cols))

    missing_cols = [col for col in final_cols if col not in clean_df.columns]

    if missing_cols:
        raise ValueError(
            "Missing columns after feature engineering: " + ", ".join(missing_cols)
        )

    final_df = clean_df[final_cols].copy()

    # ------------------------------------------------------------------
    # Remove rows where SPI3 is unavailable.
    #
    # Because conventional monthly SPI-3 requires previous months,
    # the beginning of the time series naturally has missing SPI3.
    # ------------------------------------------------------------------

    before_spi = len(final_df)

    final_df = final_df.dropna(subset=["SPI3"]).reset_index(drop=True)

    print(f"\nRemoved {before_spi - len(final_df)} rows " "without available SPI-3.")

    # ------------------------------------------------------------------
    # 7-day future targets
    # ------------------------------------------------------------------

    horizon = cfg["sequence"].get("horizon", 7)

    print(f"\nConstructing future targets: " f"{horizon}-day ahead...")

    final_df = add_severity_labels(
        final_df,
        horizon_days=horizon,
    )

    # ------------------------------------------------------------------
    # IMPORTANT:
    #
    # After creating future targets, the target belongs to t+horizon.
    #
    # We retain the current date as the prediction/reference date.
    # The model therefore learns:
    #
    #     X(t) -> Y(t+7)
    #
    # without exposing future observations as input features.
    # ------------------------------------------------------------------

    print(f"\nFinal processed dataset shape: " f"{final_df.shape}")

    return final_df


# ======================================================================
# FULL PIPELINE
# ======================================================================


def run_full_pipeline(cfg):
    """
    Run the complete data acquisition + processing pipeline.
    """

    districts = cfg["study_area"]["districts"]

    start_date = cfg["study_area"]["start_date"]
    end_date = cfg["study_area"]["end_date"]

    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Google Earth Engine
    # ------------------------------------------------------------------

    print("\nInitializing Google Earth Engine...")

    project_id = os.getenv("GEE_PROJECT_ID") or cfg["project"].get("gee_project_id")

    if not project_id:
        project_id = input("Enter your Google Earth Engine project ID: ")

    ee.Authenticate()
    ee.Initialize(project=project_id)

    gaul = ee.FeatureCollection("FAO/GAUL/2015/level2")

    bd_districts = gaul.filter(ee.Filter.eq("ADM0_NAME", "Bangladesh"))

    # ------------------------------------------------------------------
    # NASA POWER
    # ------------------------------------------------------------------

    print("\nFetching NASA POWER data...")

    power_df = fetch_all_nasa_power(
        districts,
        start_date,
        end_date,
    )

    power_df.to_csv(
        cfg["data"]["nasa_power_raw"],
        index=False,
    )

    # ------------------------------------------------------------------
    # MODIS
    # ------------------------------------------------------------------

    print("\nFetching MODIS data...")

    modis_df = fetch_all_modis(
        districts,
        bd_districts,
        start_date,
        end_date,
    )

    modis_df.to_csv(
        cfg["data"]["modis_raw"],
        index=False,
    )

    # ------------------------------------------------------------------
    # Temporal alignment
    # ------------------------------------------------------------------

    print("\nAligning to daily resolution...")

    full_df = align_all_districts(
        power_df,
        modis_df,
        districts,
    )

    full_df.to_csv(
        cfg["data"]["merged_raw"],
        index=False,
    )

    print(f"Saved merged data: " f"{cfg['data']['merged_raw']}")

    # ------------------------------------------------------------------
    # Quality control
    # ------------------------------------------------------------------

    print("\nRunning quality control...")

    clean_df = run_full_qc_pipeline(full_df)

    clean_df.to_csv(
        cfg["data"]["cleaned_data"],
        index=False,
    )

    print(f"Saved cleaned data: " f"{cfg['data']['cleaned_data']}")

    return clean_df


# ======================================================================
# MAIN
# ======================================================================


def main(config_path: str, from_cleaned: bool = False):

    cfg = load_config(config_path)

    processed_dir = Path(cfg["data"]["processed_dir"])

    # ==================================================================
    # MODE 1: FROM EXISTING CLEANED DATA
    # ==================================================================

    if from_cleaned:

        cleaned_path = Path(cfg["data"]["cleaned_data"])

        if not cleaned_path.exists():
            raise FileNotFoundError(
                f"\nCleaned dataset not found:\n"
                f"{cleaned_path}\n\n"
                f"Run the full pipeline once first."
            )

        print("\n" + "=" * 70)
        print("FROM-CLEANED MODE")
        print("=" * 70)

        print(f"Reading existing cleaned data:\n" f"{cleaned_path}")

        clean_df = pd.read_csv(
            cleaned_path,
            parse_dates=["date"],
        )

    # ==================================================================
    # MODE 2: FULL PIPELINE
    # ==================================================================

    else:

        clean_df = run_full_pipeline(cfg)

    # ==================================================================
    # FEATURE ENGINEERING + TARGET CREATION
    # ==================================================================

    final_df = process_cleaned_data(
        clean_df,
        cfg,
    )

    # ------------------------------------------------------------------
    # Save final dataset
    # ------------------------------------------------------------------

    final_path = Path(cfg["data"]["final_dataset"])

    final_df.to_csv(
        final_path,
        index=False,
    )

    print(f"\nFinal dataset written to:\n" f"{final_path}")

    print(f"Final shape: " f"{final_df.shape}")

    # ==================================================================
    # CHRONOLOGICAL SPLIT
    # ==================================================================

    train_df, val_df, test_df = chronological_split(
        final_df,
        cfg,
    )

    save_splits(
        train_df,
        val_df,
        test_df,
        processed_dir,
    )

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================

    print("\n" + "=" * 70)
    print("DATA PIPELINE COMPLETE")
    print("=" * 70)

    print(f"Full dataset : {len(final_df):,} samples")

    print(f"Train        : {len(train_df):,} samples")

    print(f"Validation   : {len(val_df):,} samples")

    print(f"Test         : {len(test_df):,} samples")

    print("\nTarget columns:")

    target_cols = [col for col in final_df.columns if "target" in col]

    for col in target_cols:
        print(f"  - {col}")

    print("=" * 70)


# ======================================================================
# CLI
# ======================================================================


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Build drought/heat forecasting dataset."
    )

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to configuration YAML file.",
    )

    parser.add_argument(
        "--from-cleaned",
        action="store_true",
        help=(
            "Skip NASA POWER, MODIS, GEE, alignment, and QC. "
            "Start directly from data/processed/cleaned_data.csv."
        ),
    )

    args = parser.parse_args()

    main(
        args.config,
        from_cleaned=args.from_cleaned,
    )
