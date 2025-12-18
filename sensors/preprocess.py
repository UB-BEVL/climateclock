"""Sensor preprocessing utilities (pure, no Streamlit).

Pipeline helpers: timezone normalization, sampling step inference,
resampling, denoising, sigma clipping, coverage diagnostics, time keys.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def normalize_timezone(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Ensure datetime index is tz-aware in the given tz.

    If the index is naive, localize. If already aware, convert.
    """
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    if out.index.tz is None:
        out.index = out.index.tz_localize(tz)
    else:
        out.index = out.index.tz_convert(tz)
    return out


def infer_sampling_step_seconds(df: pd.DataFrame) -> Optional[int]:
    """Infer median sampling step in seconds from datetime index."""
    if not isinstance(df.index, pd.DatetimeIndex) or len(df.index) < 2:
        return None
    diffs = df.index.to_series().diff().dropna().dt.total_seconds()
    if diffs.empty:
        return None
    return int(float(diffs.median()))


def resample_df(df: pd.DataFrame, freq: str, agg: str = "mean") -> pd.DataFrame:
    """Resample numeric columns to freq using provided aggregation."""
    if df.empty:
        return df.copy()
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return df.copy()
    resampler = df[numeric_cols].resample(freq)
    if agg == "median":
        out = resampler.median()
    else:
        out = resampler.mean()
    return out


def denoise_df(df: pd.DataFrame, method: str, window: str) -> pd.DataFrame:
    """Apply rolling median/mean to numeric columns."""
    if method == "none" or df.empty:
        return df.copy()
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return df.copy()
    roll = df[numeric_cols].rolling(window=window, min_periods=1, center=True)
    if method == "rolling_median":
        smoothed = roll.median()
    else:
        smoothed = roll.mean()
    out = df.copy()
    out[numeric_cols] = smoothed
    return out


def _mad(series: pd.Series) -> float:
    med = series.median()
    return float(np.median(np.abs(series - med)))


def sigma_clip_df(df: pd.DataFrame, cols: Iterable[str], sigma: float = 4.0) -> pd.DataFrame:
    """Clip outliers in specified columns using median±sigma*MAD."""
    if df.empty:
        return df.copy()
    out = df.copy()
    for col in cols:
        if col not in out or not pd.api.types.is_numeric_dtype(out[col]):
            continue
        series = out[col].copy()
        med = series.median()
        mad = _mad(series)
        if mad == 0 or math.isnan(mad):
            continue
        upper = med + sigma * mad
        lower = med - sigma * mad
        out[col] = series.clip(lower=lower, upper=upper)
    return out


def coverage_diagnostics(df: pd.DataFrame) -> dict:
    """Return basic coverage stats for a datetime-indexed dataframe."""
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return {
            "start": None,
            "end": None,
            "duration_days": None,
            "n_points": 0,
            "step_seconds": None,
            "missing_pct": {},
            "largest_gap_hours": None,
            "gap_count": 0,
        }
    idx = df.index
    start, end = idx.min(), idx.max()
    step = infer_sampling_step_seconds(df)
    diffs = idx.to_series().diff().dropna().dt.total_seconds()
    gap_thresh = (step or diffs.median() or 0) * 3 if not diffs.empty else None
    gap_count = int((diffs > gap_thresh).sum()) if gap_thresh is not None else 0
    largest_gap_hours = float(diffs.max() / 3600.0) if not diffs.empty else None
    missing_pct = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            missing_pct[col] = float(df[col].isna().mean())
    return {
        "start": start,
        "end": end,
        "duration_days": float((end - start).total_seconds() / 86400.0) if start and end else None,
        "n_points": int(len(df)),
        "step_seconds": step,
        "missing_pct": missing_pct,
        "largest_gap_hours": largest_gap_hours,
        "gap_count": gap_count,
    }


def add_time_keys(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df.copy()
    out = df.copy()
    out["month"] = out.index.month
    out["doy"] = out.index.dayofyear
    out["hour"] = out.index.hour
    return out
#sensorhelp