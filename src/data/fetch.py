"""
Data acquisition: NASA POWER (meteorological) and MODIS (NDVI/LST via
Google Earth Engine), plus daily temporal alignment between the two.

Extracted from notebook cells 10-27. Only runs on whichever machine has
GEE authenticated and internet access — the university PC almost
certainly doesn't need to run this at all, since final_dataset.csv is
already produced and just needs to be copied over.
"""
import time
from concurrent.futures import ThreadPoolExecutor

import ee
import numpy as np
import pandas as pd
import requests


# ----------------------------------------------------------------------
# Study area
# ----------------------------------------------------------------------
def get_district_geometry(bd_districts: "ee.FeatureCollection", district_name: str):
    feat = bd_districts.filter(ee.Filter.eq("ADM2_NAME", district_name)).first()
    return feat.geometry()


def generate_district_points(lat: float, lon: float, offset: float = 0.15) -> dict:
    return {
        "center": (lat, lon),
        "north": (lat + offset, lon),
        "south": (lat - offset, lon),
        "east": (lat, lon + offset),
        "west": (lat, lon - offset),
    }


# ----------------------------------------------------------------------
# NASA POWER
# ----------------------------------------------------------------------
def fetch_nasa_power(lat: float, lon: float, start_date: str, end_date: str, max_retries: int = 3) -> pd.DataFrame:
    """
    Pull daily meteorological data from the NASA POWER API for a single point, with retry.
    Parameters: T2M (mean temp), T2M_MAX/MIN, RH2M (rel. humidity),
    PRECTOTCORR (bias-corrected precipitation), ALLSKY_SFC_SW_DWN (solar radiation), WS2M (wind speed).
    """
    params = "T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN,WS2M"
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters={params}&community=AG"
        f"&longitude={lon}&latitude={lat}"
        f"&start={start_date.replace('-', '')}&end={end_date.replace('-', '')}"
        "&format=JSON"
    )

    last_err = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=90)
            response.raise_for_status()
            data = response.json()["properties"]["parameter"]
            df = pd.DataFrame(data)
            df.index = pd.to_datetime(df.index, format="%Y%m%d")
            df.index.name = "date"
            df = df.replace(-999, np.nan)  # NASA POWER's missing-value fill code
            return df
        except requests.exceptions.RequestException as e:
            last_err = e
            wait = 2 ** attempt
            print(f"    retry {attempt + 1}/{max_retries} after error ({e}); waiting {wait}s")
            time.sleep(wait)
    raise last_err


def fetch_nasa_power_district_avg(lat: float, lon: float, start_date: str, end_date: str, offset: float = 0.15) -> pd.DataFrame:
    points = generate_district_points(lat, lon, offset)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            label: executor.submit(fetch_nasa_power, plat, plon, start_date, end_date)
            for label, (plat, plon) in points.items()
        }
        dfs = [f.result() for f in futures.values()]

    avg_df = pd.concat(dfs).groupby(level=0).mean(numeric_only=True)
    return avg_df


def fetch_all_nasa_power(districts: dict, start_date: str, end_date: str) -> pd.DataFrame:
    power_data = {}
    for name, coords in districts.items():
        print(f"Fetching NASA POWER data for {name} (5-point average)...")
        df = fetch_nasa_power_district_avg(coords["lat"], coords["lon"], start_date, end_date)
        df["district"] = name
        power_data[name] = df
    return pd.concat(power_data.values()).reset_index()


# ----------------------------------------------------------------------
# MODIS (NDVI + LST) via Google Earth Engine
# ----------------------------------------------------------------------
def mask_ndvi_quality(image):
    """MOD13Q1 SummaryQA: 0=good, 1=marginal, 2=snow/ice, 3=cloudy. Keep 0-1 only."""
    qa = image.select("SummaryQA")
    good_quality = qa.lte(1)
    return image.updateMask(good_quality)


def get_modis_ndvi(geometry, start_date: str, end_date: str) -> pd.DataFrame:
    collection = (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .filterDate(start_date, end_date)
        .filterBounds(geometry)
        .map(mask_ndvi_quality)
        .select("NDVI")
    )

    def reduce_image(img):
        mean_val = img.reduceRegion(ee.Reducer.mean(), geometry, 250).get("NDVI")
        return ee.Feature(None, {"date": img.date().format("YYYY-MM-dd"), "NDVI": mean_val})

    features = collection.map(reduce_image).getInfo()["features"]
    records = [(f["properties"]["date"], f["properties"].get("NDVI", None)) for f in features]

    df = pd.DataFrame(records, columns=["date", "NDVI"])
    df = df.dropna(subset=["NDVI"])
    df["date"] = pd.to_datetime(df["date"])
    df["NDVI"] = df["NDVI"] * 0.0001
    return df.set_index("date")


def mask_lst_quality(image):
    """MOD11A2 QC_Day: bits 0-1 = mandatory QA flag, 00 = good data quality. Keep only good."""
    qc = image.select("QC_Day")
    quality_bits = qc.bitwiseAnd(3)
    good_quality = quality_bits.eq(0)
    return image.updateMask(good_quality)


def get_modis_lst(geometry, start_date: str, end_date: str) -> pd.DataFrame:
    collection = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterDate(start_date, end_date)
        .filterBounds(geometry)
        .map(mask_lst_quality)
        .select("LST_Day_1km")
    )

    def reduce_image(img):
        mean_val = img.reduceRegion(ee.Reducer.mean(), geometry, 1000).get("LST_Day_1km")
        return ee.Feature(None, {"date": img.date().format("YYYY-MM-dd"), "LST": mean_val})

    features = collection.map(reduce_image).getInfo()["features"]
    records = [(f["properties"]["date"], f["properties"].get("LST", None)) for f in features]

    df = pd.DataFrame(records, columns=["date", "LST"])
    df = df.dropna(subset=["LST"])
    df["date"] = pd.to_datetime(df["date"])
    df["LST"] = df["LST"] * 0.02 - 273.15
    return df.set_index("date")


def fetch_all_modis(districts: dict, bd_districts, start_date: str, end_date: str) -> pd.DataFrame:
    modis_data = {}
    for name in districts:
        print(f"Fetching MODIS data for {name} (QA-masked)...")
        geom = get_district_geometry(bd_districts, name)
        ndvi = get_modis_ndvi(geom, start_date, end_date)
        lst = get_modis_lst(geom, start_date, end_date)
        merged = ndvi.join(lst, how="outer")
        merged["district"] = name
        modis_data[name] = merged
    return pd.concat(modis_data.values()).reset_index()


# ----------------------------------------------------------------------
# Temporal alignment
# ----------------------------------------------------------------------
def align_daily(power_d: pd.DataFrame, modis_d: pd.DataFrame) -> pd.DataFrame:
    """
    NASA POWER is daily; MODIS is an 8/16-day composite.
    Reindex MODIS onto the daily axis and interpolate NDVI/LST
    so every day receives an aligned vegetation value.
    """

    power_d = power_d.set_index("date").sort_index()
    modis_d = modis_d.set_index("date").sort_index()

    full_index = pd.date_range(power_d.index.min(), power_d.index.max(), freq="D")

    # Only interpolate the numerical MODIS variables.
    # 'district' is categorical and must never be interpolated.
    modis_numeric = modis_d[["NDVI", "LST"]].apply(pd.to_numeric, errors="coerce")

    modis_daily = (
        modis_numeric.reindex(full_index).interpolate(method="time").ffill().bfill()
    )

    merged = power_d.join(modis_daily[["NDVI", "LST"]], how="left")

    return merged.reset_index().rename(columns={"index": "date"})


def align_all_districts(power_df: pd.DataFrame, modis_df: pd.DataFrame, districts: dict) -> pd.DataFrame:
    merged_data = []
    for name in districts:
        p = power_df[power_df["district"] == name].copy()
        m = modis_df[modis_df["district"] == name].copy()
        merged = align_daily(p, m)
        merged["district"] = name
        merged_data.append(merged)
    return pd.concat(merged_data).reset_index(drop=True)
