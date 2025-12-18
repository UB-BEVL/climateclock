from __future__ import annotations

from datetime import tzinfo
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import requests

# Storage ---------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SENSOR_STORE = DATA_DIR / "sensors.parquet"

SENSOR_COLUMNS = [
    "timestamp",
    "sensor_id",
    "temperature",
    "relative_humidity",
    "ghi",
    "wind_speed",
    "wind_dir",
    "lat",
    "lon",
    "source",
]

COLUMN_ALIASES = {
    "timestamp": ["timestamp", "time", "datetime", "ts"],
    "sensor_id": ["sensor_id", "id", "device", "device_id"],
    "temperature": ["temperature", "temp", "temp_c", "air_temperature"],
    "relative_humidity": ["relative_humidity", "rh", "humidity"],
    "ghi": ["ghi", "solar", "global_horizontal_irradiance"],
    "wind_speed": ["wind_speed", "windspeed", "wind"],
    "wind_dir": ["wind_dir", "winddir", "wdir", "wd", "wind_direction"],
    "lat": ["lat", "latitude"],
    "lon": ["lon", "longitude"],
    "source": ["source", "provider"],
}

FLOAT_COLUMNS = ["temperature", "relative_humidity", "ghi", "wind_speed", "wind_dir", "lat", "lon"]


def _calc_abs_humidity(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    """Compute absolute humidity (g/m³) from temperature (°C) and RH (%)."""
    temp_c = pd.to_numeric(temp_c, errors="coerce")
    rh_pct = pd.to_numeric(rh_pct, errors="coerce")
    es = 610.94 * np.exp(17.625 * temp_c / (temp_c + 243.04))
    vap = es * (rh_pct / 100.0)
    return 216.7 * vap / (temp_c + 273.15)


def _require_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    for target, aliases in COLUMN_ALIASES.items():
        if target in renamed.columns:
            continue
        for alias in aliases:
            if alias in renamed.columns:
                renamed = renamed.rename(columns={alias: target})
                break
    for col in SENSOR_COLUMNS:
        if col not in renamed.columns:
            renamed[col] = np.nan
    return renamed[SENSOR_COLUMNS]


def ingest_csv(uploaded_file) -> pd.DataFrame:
    """Parse a CSV upload into the canonical schema.

    Parameters
    ----------
    uploaded_file: IO
        File-like object returned by Streamlit's uploader.

    Returns
    -------
    pd.DataFrame
        Schema-aligned sensor data indexed by timestamp.
    """
    df = pd.read_csv(uploaded_file)
    normalized = _require_columns(df)
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True, errors="coerce")
    normalized["sensor_id"] = normalized["sensor_id"].fillna("sensor_csv")
    normalized["source"] = normalized["source"].fillna("csv_upload")
    for col in FLOAT_COLUMNS:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
    normalized = normalized.dropna(subset=["timestamp"])
    return normalized


def fetch_live_api(url: str) -> pd.DataFrame:
    """Fetch JSON data from a REST endpoint and coerce into schema.

    Parameters
    ----------
    url : str
        REST endpoint returning JSON rows or an object with a ``data`` field.

    Returns
    -------
    pd.DataFrame
        Sensor dataframe aligned to :data:`SENSOR_COLUMNS`.
    """
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    if isinstance(payload, dict):
        data = payload.get("data") or payload.get("records") or payload
        df = pd.json_normalize(data)
    else:
        df = pd.json_normalize(payload)
    normalized = _require_columns(df)
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True, errors="coerce")
    normalized["sensor_id"] = normalized["sensor_id"].fillna("sensor_api")
    normalized["source"] = normalized["source"].fillna("api")
    for col in FLOAT_COLUMNS:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
    normalized = normalized.dropna(subset=["timestamp"])
    return normalized


def normalize_sensor_df(df: pd.DataFrame, tz: Optional[Union[str, tzinfo]]) -> pd.DataFrame:
    """Ensure dataframe matches schema and is localized to the provided timezone."""
    normalized = _require_columns(df)
    ts = pd.to_datetime(normalized["timestamp"], utc=True, errors="coerce")
    if tz:
        ts = ts.dt.tz_convert(tz)
    normalized["timestamp"] = ts
    normalized = normalized.dropna(subset=["timestamp"])
    normalized["sensor_id"] = normalized["sensor_id"].fillna("sensor")
    normalized["source"] = normalized["source"].fillna("unknown")
    for col in FLOAT_COLUMNS:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
    return normalized


def load_sensor_data() -> pd.DataFrame:
    """Load stored sensor data from the local parquet store."""
    if SENSOR_STORE.exists():
        return pd.read_parquet(SENSOR_STORE)
    return pd.DataFrame(columns=SENSOR_COLUMNS)


def append_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    """Append new readings to the local store and return the combined dataframe."""
    existing = load_sensor_data()
    combined = pd.concat([existing, df], ignore_index=True)
    combined = (
        combined.dropna(subset=["timestamp"])
        .drop_duplicates(subset=["timestamp", "sensor_id"], keep="last")
        .sort_values("timestamp")
    )
    combined.to_parquet(SENSOR_STORE, index=False)
    return combined


def _annotate_climatology(df: pd.DataFrame) -> pd.DataFrame:
    """Add helper columns for climatology joins (doy, hour, month)."""
    out = df.copy()
    out["doy"] = out["timestamp"].dt.dayofyear
    out["hour"] = out["timestamp"].dt.hour
    out["month"] = out["timestamp"].dt.month
    return out


def build_sensor_climatology(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sensor data by day-of-year/hour means (temp, humidity, solar, wind)."""
    if df.empty:
        return pd.DataFrame(columns=[
            "doy", "hour", "sensor_temp", "sensor_rh", "sensor_ghi",
            "sensor_abs_hum", "sensor_windspd", "sensor_winddir",
        ])
    annotated = _annotate_climatology(df)

    if {"temperature", "relative_humidity"}.issubset(annotated.columns):
        annotated["abs_hum"] = _calc_abs_humidity(annotated["temperature"], annotated["relative_humidity"])
    else:
        annotated["abs_hum"] = np.nan

    for col in ["ghi", "wind_speed", "wind_dir"]:
        if col not in annotated.columns:
            annotated[col] = np.nan

    def _circular_mean(deg: pd.Series) -> float:
        vals = pd.to_numeric(deg, errors="coerce").dropna()
        if vals.empty:
            return np.nan
        rad = np.deg2rad(vals)
        mean_angle = np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())
        return (np.degrees(mean_angle) + 360.0) % 360.0

    grouped = (
        annotated.groupby(["doy", "hour"], as_index=False)
        .agg({
            "temperature": "mean",
            "relative_humidity": "mean",
            "ghi": "mean",
            "abs_hum": "mean",
            "wind_speed": "mean",
            "wind_dir": _circular_mean,
        })
        .rename(
            columns={
                "temperature": "sensor_temp",
                "relative_humidity": "sensor_rh",
                "ghi": "sensor_ghi",
                "abs_hum": "sensor_abs_hum",
                "wind_speed": "sensor_windspd",
                "wind_dir": "sensor_winddir",
            }
        )
    )
    return grouped


def build_epw_climatology(cdf: pd.DataFrame) -> pd.DataFrame:
    """Aggregate EPW climate dataframe into typical doy/hour values."""
    if cdf.empty:
        return pd.DataFrame(columns=[
            "doy", "hour", "epw_temp", "epw_rh", "epw_ghi",
            "epw_abs_hum", "epw_windspd", "epw_winddir",
        ])
    df = cdf.copy()
    df["doy"] = df.index.dayofyear
    df["hour"] = df.index.hour

    if {"drybulb", "relhum"}.issubset(df.columns):
        df["abs_hum"] = _calc_abs_humidity(df["drybulb"], df["relhum"])
    else:
        df["abs_hum"] = np.nan

    for col in ["glohorrad", "windspd", "winddir"]:
        if col not in df.columns:
            df[col] = np.nan

    def _circular_mean(deg: pd.Series) -> float:
        vals = pd.to_numeric(deg, errors="coerce").dropna()
        if vals.empty:
            return np.nan
        rad = np.deg2rad(vals)
        mean_angle = np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())
        return (np.degrees(mean_angle) + 360.0) % 360.0

    grouped = (
        df.groupby(["doy", "hour"], as_index=False)
        .agg({
            "drybulb": "mean",
            "relhum": "mean",
            "glohorrad": "mean",
            "abs_hum": "mean",
            "windspd": "mean",
            "winddir": _circular_mean,
        })
        .rename(
            columns={
                "drybulb": "epw_temp",
                "relhum": "epw_rh",
                "glohorrad": "epw_ghi",
                "abs_hum": "epw_abs_hum",
                "windspd": "epw_windspd",
                "winddir": "epw_winddir",
            }
        )
    )
    return grouped


def compare_epw_vs_sensor(epw_clim: pd.DataFrame, sensor_clim: pd.DataFrame) -> pd.DataFrame:
    """Merge climatologies and compute bias columns."""
    if epw_clim.empty:
        return sensor_clim.assign(
            epw_temp=np.nan,
            epw_rh=np.nan,
            epw_ghi=np.nan,
            epw_abs_hum=np.nan,
            epw_windspd=np.nan,
            epw_winddir=np.nan,
            temp_bias=np.nan,
            rh_bias=np.nan,
            ghi_bias=np.nan,
            abs_hum_bias=np.nan,
            windspd_bias=np.nan,
            winddir_bias=np.nan,
        )
    if sensor_clim.empty:
        return epw_clim.assign(
            sensor_temp=np.nan,
            sensor_rh=np.nan,
            sensor_ghi=np.nan,
            sensor_abs_hum=np.nan,
            sensor_windspd=np.nan,
            sensor_winddir=np.nan,
            temp_bias=np.nan,
            rh_bias=np.nan,
            ghi_bias=np.nan,
            abs_hum_bias=np.nan,
            windspd_bias=np.nan,
            winddir_bias=np.nan,
        )
    merged = pd.merge(epw_clim, sensor_clim, on=["doy", "hour"], how="outer")
    merged["temp_bias"] = merged["sensor_temp"] - merged["epw_temp"]
    merged["rh_bias"] = merged["sensor_rh"] - merged["epw_rh"]
    merged["ghi_bias"] = merged["sensor_ghi"] - merged["epw_ghi"]
    merged["abs_hum_bias"] = merged["sensor_abs_hum"] - merged["epw_abs_hum"]
    merged["windspd_bias"] = merged["sensor_windspd"] - merged["epw_windspd"]

    def _wind_dir_diff(row):
        a, b = row.get("sensor_winddir"), row.get("epw_winddir")
        if pd.isna(a) or pd.isna(b):
            return np.nan
        diff = (a - b + 540) % 360 - 180
        return diff

    merged["winddir_bias"] = merged.apply(_wind_dir_diff, axis=1)
    return merged
#sensor