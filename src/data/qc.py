"""
Quality control: physical-limit checks, IQR outlier flagging, and
gap-aware interpolation (short/medium/long gap handling by block size).

Extracted from notebook cells 31-35.
"""

import numpy as np
import pandas as pd

PHYSICAL_LIMITS = {
    "T2M": (-5, 50),  # deg C, loose bounds for Bangladesh's climate
    "T2M_MAX": (-5, 55),
    "T2M_MIN": (-10, 45),
    "RH2M": (0, 100),
    "PRECTOTCORR": (0, 500),  # mm/day, generous ceiling for monsoon extremes
    "WS2M": (0, 50),  # m/s
    "ALLSKY_SFC_SW_DWN": (0, 40),  # MJ/m^2/day
}


QC_COLS = [
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "PRECTOTCORR",
    "ALLSKY_SFC_SW_DWN",
    "WS2M",
]


def apply_physical_limits(
    df: pd.DataFrame, limits: dict = PHYSICAL_LIMITS
) -> pd.DataFrame:

    df = df.copy()

    for col, (lo, hi) in limits.items():

        if col not in df.columns:
            continue

        mask = (df[col] < lo) | (df[col] > hi)

        n_removed = int(mask.sum())

        if n_removed > 0:
            print(f"  {col}: removing " f"{n_removed} physically implausible values")

        df.loc[mask, col] = np.nan

    return df


def apply_iqr_outliers(df: pd.DataFrame, cols: list, k: float = 3.0) -> pd.DataFrame:

    df = df.copy()

    for col in cols:

        if col not in df.columns:
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lo = q1 - k * iqr
        hi = q3 + k * iqr

        mask = (df[col] < lo) | (df[col] > hi)

        n_flagged = int(mask.sum())

        if n_flagged > 0:
            print(
                f"  {col}: flagging "
                f"{n_flagged} statistical outliers "
                f"(IQR, k={k})"
            )

        df.loc[mask, col] = np.nan

    return df


def run_qc_per_district(
    full_df: pd.DataFrame, qc_cols: list = QC_COLS, k: float = 3.0
) -> pd.DataFrame:

    qc_results = []

    for name, group in full_df.groupby("district"):

        print(f"Quality control for {name}:")

        g = apply_physical_limits(group)

        g = apply_iqr_outliers(g, qc_cols, k=k)

        qc_results.append(g)

    return pd.concat(qc_results, ignore_index=True)


def fill_gaps_gap_aware(
    df: pd.DataFrame,
    numeric_cols,
    short_max: int = 3,
    medium_max: int = 15,
    very_long_min: int = 90,
) -> pd.DataFrame:
    """
    Short gaps (<= short_max days):
        Linear interpolation.

    Medium gaps:
        Spline interpolation, falling back to linear interpolation
        if spline interpolation is unavailable.

    Long gaps (< very_long_min):
        Day-of-year seasonal climatology.

    Gaps >= very_long_min:
        Left as NaN and dropped downstream.
    """

    df = df.copy()

    doy = df.index.dayofyear

    for col in numeric_cols:

        series = df[col]

        is_na = series.isna()

        if not is_na.any():
            continue

        block_id = (is_na != is_na.shift()).cumsum()

        block_sizes = is_na.groupby(block_id).transform("sum")

        short_mask = is_na & (block_sizes <= short_max)

        medium_mask = is_na & (block_sizes > short_max) & (block_sizes <= medium_max)

        long_mask = is_na & (block_sizes > medium_max) & (block_sizes < very_long_min)

        # --------------------------------------------------
        # Short gaps: linear interpolation
        # --------------------------------------------------

        lin = series.interpolate(method="linear", limit=short_max, limit_area="inside")

        series = series.where(~short_mask, lin)

        # --------------------------------------------------
        # Medium gaps: spline interpolation
        # --------------------------------------------------

        try:

            spl = series.interpolate(
                method="spline", order=2, limit=medium_max, limit_area="inside"
            )

            series = series.where(~medium_mask, spl)

        except Exception:

            lin2 = series.interpolate(
                method="linear", limit=medium_max, limit_area="inside"
            )

            series = series.where(~medium_mask, lin2)

        # --------------------------------------------------
        # Long gaps: seasonal climatology
        # --------------------------------------------------

        if long_mask.any():

            climatology = series.groupby(doy).transform("mean")

            series = series.where(~long_mask, climatology)

        df[col] = series

    return df


def clean_district(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean one district's time series.

    The district identifier is handled explicitly by
    run_full_qc_pipeline(), so this function only performs
    date sorting and gap filling.
    """

    df = df.copy()

    # Make sure date is datetime
    df["date"] = pd.to_datetime(df["date"])

    # Sort chronologically
    df = df.set_index("date").sort_index()

    # Only numeric columns are interpolated.
    # Metadata such as district is not modified.
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    df = fill_gaps_gap_aware(df, numeric_cols)

    return df.reset_index()


def run_full_qc_pipeline(
    full_df: pd.DataFrame, qc_cols: list = QC_COLS
) -> pd.DataFrame:

    # --------------------------------------------------
    # Step 1: Per-district QC
    # --------------------------------------------------

    qc_df = run_qc_per_district(full_df, qc_cols)

    # --------------------------------------------------
    # Step 2: Report NaNs introduced by QC
    # --------------------------------------------------

    total_nans = qc_df[qc_cols].isna().sum().sum()

    print("\nTotal NaNs introduced by QC step:", total_nans)

    # --------------------------------------------------
    # Step 3: Clean each district independently
    #
    # IMPORTANT:
    # We explicitly capture the district name from
    # groupby() and restore it after cleaning.
    # This avoids pandas GroupBy.apply() dropping
    # the grouping column.
    # --------------------------------------------------

    clean_parts = []

    for district, group in qc_df.groupby("district", sort=True):

        cleaned = clean_district(group)

        # Explicitly restore district metadata
        cleaned["district"] = district

        clean_parts.append(cleaned)

    # --------------------------------------------------
    # Step 4: Combine all districts
    # --------------------------------------------------

    clean_df = pd.concat(clean_parts, ignore_index=True)

    # --------------------------------------------------
    # Step 5: Sort by district and date
    # --------------------------------------------------

    clean_df["date"] = pd.to_datetime(clean_df["date"])

    clean_df = clean_df.sort_values(["district", "date"]).reset_index(drop=True)

    # --------------------------------------------------
    # Step 6: Remove rows belonging to very-long gaps
    # --------------------------------------------------

    n_before = len(clean_df)

    clean_df = clean_df.dropna(subset=qc_cols)

    n_after = len(clean_df)

    print(
        f"Dropped {n_before - n_after} rows "
        f"from gaps too long to safely fill "
        f"(>= 90 days)"
    )

    # --------------------------------------------------
    # Step 7: Final validation
    # --------------------------------------------------

    if "district" not in clean_df.columns:
        raise RuntimeError(
            "QC pipeline failed: 'district' column " "was lost during cleaning."
        )

    return clean_df
