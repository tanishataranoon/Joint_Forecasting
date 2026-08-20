"""
Feature engineering and future target construction.

Features:
- Conventional monthly SPI-3
- VPD
- Humidex
- Cyclic temporal features

Targets:
- 7-day-ahead drought severity
- 7-day-ahead heat-stress severity

Forecasting rule:
Only information available at time t is used as an input feature.
Future values at t+7 are used only for target construction.
"""

import numpy as np
import pandas as pd
from scipy.stats import gamma, norm

# ======================================================================
# SPI-3
# ======================================================================


def compute_monthly_precipitation(
    group: pd.DataFrame, precip_col: str = "PRECTOTCORR"
) -> pd.DataFrame:

    g = group.copy()
    g["date"] = pd.to_datetime(g["date"])
    g = g.sort_values("date")

    monthly = (
        g.set_index("date")[precip_col]
        .resample("MS")
        .sum(min_count=1)
        .to_frame("monthly_precip")
    )

    return monthly


def compute_spi3_monthly(
    group: pd.DataFrame, precip_col: str = "PRECTOTCORR"
) -> pd.DataFrame:
    """
    Conventional monthly SPI-3.

    1. Aggregate daily precipitation to monthly totals.
    2. Calculate 3-month accumulated precipitation.
    3. Fit gamma separately for each calendar month.
    4. Convert cumulative probability to standard normal SPI.
    """

    monthly = compute_monthly_precipitation(group, precip_col=precip_col)

    monthly["precip_3month"] = (
        monthly["monthly_precip"].rolling(window=3, min_periods=3).sum()
    )

    monthly["SPI3"] = np.nan

    for month in range(1, 13):

        mask = monthly.index.month == month

        values = monthly.loc[mask, "precip_3month"].dropna()

        if len(values) < 10:
            continue

        zero_probability = (values <= 0).mean()

        positive_values = values[values > 0]

        if len(positive_values) < 5:
            continue

        try:

            shape, loc, scale = gamma.fit(positive_values, floc=0)

        except Exception:
            continue

        month_values = monthly.loc[mask, "precip_3month"]

        valid_mask = month_values.notna()

        valid_values = month_values[valid_mask]

        if valid_values.empty:
            continue

        gamma_cdf = gamma.cdf(valid_values, shape, loc=loc, scale=scale)

        probabilities = zero_probability + (1.0 - zero_probability) * gamma_cdf

        probabilities = np.clip(probabilities, 0.0001, 0.9999)

        monthly.loc[valid_values.index, "SPI3"] = norm.ppf(probabilities)

    return monthly[["SPI3"]]


def add_spi3(clean_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add leakage-safe monthly SPI-3 to daily observations.

    For a day in month M, only the SPI-3 calculated from the
    most recently completed month M-1 is used.

    Example:

        July 15 → June SPI-3
        August 15 → July SPI-3
    """

    df = clean_df.copy()

    df["date"] = pd.to_datetime(df["date"])

    district_frames = []

    for district, group in df.groupby("district"):

        group = group.sort_values("date").copy()

        monthly_spi = compute_spi3_monthly(group)

        monthly_spi = monthly_spi.reset_index()

        monthly_spi["month_start"] = monthly_spi["date"]

        monthly_spi = monthly_spi[["month_start", "SPI3"]]

        # Shift because the current month's complete
        # precipitation is not available at the beginning
        # of that month.
        monthly_spi["SPI3_available"] = monthly_spi["SPI3"].shift(1)

        group["month_start"] = group["date"].dt.to_period("M").dt.to_timestamp()

        group = group.merge(
            monthly_spi[["month_start", "SPI3_available"]], on="month_start", how="left"
        )

        group["SPI3"] = group["SPI3_available"]

        group.drop(columns=["month_start", "SPI3_available"], inplace=True)

        district_frames.append(group)

    return (
        pd.concat(district_frames)
        .sort_values(["district", "date"])
        .reset_index(drop=True)
    )


# ======================================================================
# DROUGHT CLASSIFICATION
# ======================================================================


def classify_spi(spi):

    if pd.isna(spi):
        return np.nan

    if spi >= 2.0:
        return "Extremely Wet"

    elif spi >= 1.5:
        return "Very Wet"

    elif spi >= 1.0:
        return "Moderately Wet"

    elif spi > -1.0:
        return "Near Normal"

    elif spi > -1.5:
        return "Moderately Dry"

    elif spi > -2.0:
        return "Severely Dry"

    else:
        return "Extremely Dry"


def spi3_to_class(spi):

    if pd.isna(spi):
        return np.nan

    if spi > -1.0:
        return 0

    elif spi > -1.5:
        return 1

    elif spi > -2.0:
        return 2

    else:
        return 3


# ======================================================================
# VPD
# ======================================================================


def compute_vpd(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:

    svp = 0.611 * np.exp((17.27 * temp_c) / (temp_c + 237.3))

    avp = svp * (rh_pct / 100.0)

    return svp - avp


# ======================================================================
# HUMIDEX
# ======================================================================


def compute_dewpoint(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:

    a = 17.27
    b = 237.7

    rh_safe = rh_pct.clip(lower=1e-6)

    alpha = (a * temp_c) / (b + temp_c) + np.log(rh_safe / 100.0)

    return b * alpha / (a - alpha)


def compute_humidex(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:

    dewpoint_c = compute_dewpoint(temp_c, rh_pct)

    dewpoint_k = dewpoint_c + 273.15

    e = 6.11 * np.exp(5417.7530 * (1 / 273.16 - 1 / dewpoint_k))

    return temp_c + 0.5555 * (e - 10)


def classify_humidex(h):

    if pd.isna(h):
        return np.nan

    if h < 20:
        return "Cool"

    elif h < 30:
        return "Little Discomfort"

    elif h < 40:
        return "Some Discomfort"

    elif h <= 45:
        return "Great Discomfort"

    else:
        return "Dangerous"


def humidex_to_class(hmdx):

    if pd.isna(hmdx):
        return np.nan

    if hmdx < 30:
        return 0

    elif hmdx < 40:
        return 1

    elif hmdx <= 45:
        return 2

    else:
        return 3


# ======================================================================
# TEMPORAL FEATURES
# ======================================================================


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    day_of_year = df["date"].dt.dayofyear

    period = 365.25

    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / period)

    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / period)

    return df


# ======================================================================
# FEATURE ENGINEERING
# ======================================================================


def engineer_features(clean_df: pd.DataFrame) -> pd.DataFrame:

    clean_df = add_spi3(clean_df)

    clean_df["VPD"] = compute_vpd(clean_df["T2M"], clean_df["RH2M"])

    clean_df["Humidex"] = compute_humidex(clean_df["T2M"], clean_df["RH2M"])

    clean_df = add_temporal_features(clean_df)

    return clean_df


# ======================================================================
# 7-DAY FUTURE TARGETS
# ======================================================================


def add_future_targets(df: pd.DataFrame, horizon_days: int = 7) -> pd.DataFrame:
    """
    Construct 7-day-ahead targets.

    Input at time t:

        X(t)

    Target:

        drought_class(t+7)
        heat_class(t+7)

    No future feature values are copied into X(t).
    """

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(["district", "date"]).reset_index(drop=True)

    # Current severity derived from current observations.
    df["_drought_current"] = df["SPI3"].apply(spi3_to_class)

    df["_heat_current"] = df["Humidex"].apply(humidex_to_class)

    # Shift within each district.
    df["drought_target_7d"] = df.groupby("district")["_drought_current"].shift(
        -horizon_days
    )

    df["heat_target_7d"] = df.groupby("district")["_heat_current"].shift(-horizon_days)

    # Future target must exist.
    df = df.dropna(
        subset=["SPI3", "Humidex", "drought_target_7d", "heat_target_7d"]
    ).reset_index(drop=True)

    df["drought_target_7d"] = df["drought_target_7d"].astype(int)

    df["heat_target_7d"] = df["heat_target_7d"].astype(int)

    df.drop(columns=["_drought_current", "_heat_current"], inplace=True)

    return df


# ======================================================================
# BACKWARD COMPATIBILITY
# ======================================================================


def add_severity_labels(final_df: pd.DataFrame, horizon_days: int = 7) -> pd.DataFrame:

    return add_future_targets(final_df, horizon_days=horizon_days)
