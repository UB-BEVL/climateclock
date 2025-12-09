from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------
# Comfort metric primitives
# ---------------------------

def compute_di(df: pd.DataFrame, temp_col: str = "drybulb", rh_col: str = "relhum") -> pd.Series:
    """Compute Thom's Discomfort Index (DI).

    Formula (Thom, 1959):
        DI = T - 0.55 * (1 - RH/100) * (T - 14.5)
    where T is dry-bulb temperature in °C and RH is relative humidity in %.
    """
    if temp_col not in df or rh_col not in df:
        raise KeyError(f"Missing required columns for DI: {temp_col}, {rh_col}")
    T = df[temp_col].astype(float)
    RH = df[rh_col].astype(float).clip(0, 100)
    di = T - 0.55 * (1 - RH / 100.0) * (T - 14.5)
    return pd.Series(di.to_numpy(), index=df.index, name="DI")


def compute_utci_approx(
    df: pd.DataFrame,
    temp_col: str = "drybulb",
    rh_col: str = "relhum",
    wind_col: str = "windspd",
) -> pd.Series:
    """Approximate UTCI from basic meteorological variables.

    This uses a reduced form of the official UTCI polynomial (Broede et al. 2012) that
    captures first-order effects of air temperature (Ta), vapor pressure (vp), and wind speed (ws):

        UTCI ≈ Ta + 0.607562 + 0.022771 * Ta + 0.000806 * Ta**2 + 0.002 * vp
                - 0.065 * ws + 0.001 * Ta * ws - 0.015 * Ta * vp - 0.00025 * vp * ws

    where vapor pressure vp (hPa) = RH/100 * 6.105 * exp(17.27 * Ta / (237.7 + Ta)).

    The coefficients above are a simplified fit intended for rapid assessments; they do not
    replace the full UTCI implementation but provide comparable trends for design studies.
    """
    missing = [c for c in (temp_col, rh_col, wind_col) if c not in df.columns]
    if missing:
        raise KeyError(f"Cannot compute UTCI approximation, missing columns: {missing}")

    Ta = df[temp_col].astype(float)
    RH = df[rh_col].astype(float).clip(0, 100)
    ws = df[wind_col].astype(float).fillna(1.5).clip(lower=0.1)

    vp = (RH / 100.0) * 6.105 * np.exp((17.27 * Ta) / (237.7 + Ta))
    utci = (
        Ta
        + 0.607562
        + 0.022771 * Ta
        + 0.000806 * (Ta**2)
        + 0.002 * vp
        - 0.065 * ws
        + 0.001 * Ta * ws
        - 0.015 * Ta * vp / 100.0
        - 0.00025 * vp * ws
    )
    return pd.Series(utci.to_numpy(), index=df.index, name="UTCI")


def compute_heat_index(
    df: pd.DataFrame,
    temp_col: str = "drybulb",
    rh_col: str = "relhum",
) -> pd.Series:
    """Compute NOAA heat index (°C) from temperature (°C) and RH (%)."""
    missing = [c for c in (temp_col, rh_col) if c not in df.columns]
    if missing:
        raise KeyError(f"Cannot compute Heat Index, missing columns: {missing}")

    T = df[temp_col].astype(float)
    RH = df[rh_col].astype(float).clip(0, 100)
    Tf = T * 9.0 / 5.0 + 32.0

    hi_f = (
        -42.379
        + 2.04901523 * Tf
        + 10.14333127 * RH
        - 0.22475541 * Tf * RH
        - 0.00683783 * (Tf ** 2)
        - 0.05481717 * (RH ** 2)
        + 0.00122874 * (Tf ** 2) * RH
        + 0.00085282 * Tf * (RH ** 2)
        - 0.00000199 * (Tf ** 2) * (RH ** 2)
    )

    adj_low_rh = ((13 - RH) / 4.0) * np.sqrt((17 - np.abs(Tf - 95)) / 17)
    adj_high_rh = ((RH - 85) / 10.0) * ((87 - Tf) / 5.0)

    cond_low = (RH < 13) & (Tf >= 80) & (Tf <= 112)
    cond_high = (RH > 85) & (Tf >= 80) & (Tf <= 87)
    hi_f = np.where(cond_low, hi_f - adj_low_rh, hi_f)
    hi_f = np.where(cond_high, hi_f + adj_high_rh, hi_f)
    hi_f = np.where(Tf < 80, Tf, hi_f)

    hi_c = (hi_f - 32.0) * 5.0 / 9.0
    return pd.Series(hi_c, index=df.index, name="heat_index")


def compute_humidex(
    df: pd.DataFrame,
    temp_col: str = "drybulb",
    rh_col: str = "relhum",
) -> pd.Series:
    """Compute Canadian humidex value."""
    missing = [c for c in (temp_col, rh_col) if c not in df.columns]
    if missing:
        raise KeyError(f"Cannot compute Humidex, missing columns: {missing}")

    T = df[temp_col].astype(float)
    RH = df[rh_col].astype(float).clip(1, 100)
    a, b = 17.625, 243.04
    gamma = np.log(RH / 100.0) + (a * T) / (b + T)
    dew_point = (b * gamma) / (a - gamma)
    e = 6.11 * np.exp(5417.7530 * ((1 / 273.16) - 1 / (273.15 + dew_point)))
    humidex = T + 0.5555 * (e - 10.0)
    return pd.Series(humidex, index=df.index, name="humidex")


def build_adaptive_band(
    temp_series: pd.Series,
    acceptability: float = 0.8,
    alpha: float = 0.2,
) -> pd.DataFrame:
    """Return adaptive comfort low/high (ASHRAE 55 style) aligned to the source index."""
    if temp_series is None or temp_series.empty:
        return pd.DataFrame(columns=["low", "high"])
    daily = temp_series.resample("1D").mean().dropna()
    if daily.empty:
        return pd.DataFrame(columns=["low", "high"], index=temp_series.index)
    trm = daily.ewm(alpha=alpha, adjust=False).mean()
    t_comf = (0.31 * trm + 17.8).clip(-30, 60)
    if acceptability >= 0.9:
        span = 2.5
    else:
        span = 3.5
    low = (t_comf - span).reindex(temp_series.index, method="ffill")
    high = (t_comf + span).reindex(temp_series.index, method="ffill")
    return pd.DataFrame({"low": low, "high": high})


# ---------------------------
# Classification helpers
# ---------------------------

_DI_BANDS = [
    (-np.inf, 21.0, "Comfortable"),
    (21.0, 24.0, "Slight Discomfort"),
    (24.0, 27.0, "Discomfort"),
    (27.0, 29.0, "Strong Discomfort"),
    (29.0, np.inf, "Medical Emergency"),
]


def classify_di(di: pd.Series) -> pd.Series:
    categories = pd.Series(index=di.index, dtype="object")
    for lower, upper, label in _DI_BANDS:
        mask = (di > lower) & (di <= upper)
        categories.loc[mask] = label
    return categories.rename("DI Category")


_UTCI_BANDS = [
    (-np.inf, -27, "Extreme cold stress"),
    (-27, -13, "Very strong cold stress"),
    (-13, 0, "Strong cold stress"),
    (0, 9, "Moderate cold stress"),
    (9, 26, "No thermal stress"),
    (26, 32, "Moderate heat stress"),
    (32, 38, "Strong heat stress"),
    (38, 46, "Very strong heat stress"),
    (46, np.inf, "Extreme heat stress"),
]


def classify_utci(utci: pd.Series) -> pd.Series:
    categories = pd.Series(index=utci.index, dtype="object")
    for lower, upper, label in _UTCI_BANDS:
        mask = (utci > lower) & (utci <= upper)
        categories.loc[mask] = label
    return categories.rename("UTCI Category")


# ---------------------------
# Aggregations
# ---------------------------


def summarize_comfort(
    df: pd.DataFrame,
    di: Optional[pd.Series] = None,
    utci: Optional[pd.Series] = None,
    freq: str = "A",
    comfort_band: Optional[Tuple[float, float]] = (18.0, 26.0),
    adaptive_band: Optional[pd.DataFrame] = None,
    overheating_thresholds: Sequence[float] = (26.0, 28.0, 30.0),
    cold_thresholds: Optional[Sequence[float]] = None,
    percentiles: Optional[Sequence[float]] = (0.9, 0.95),
    occupancy_mask: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Aggregate comfort diagnostics over the requested period."""
    if df.empty:
        return pd.DataFrame()

    data = pd.DataFrame(index=df.index)
    data["temp"] = df.get("drybulb")
    data["relhum"] = df.get("relhum")

    if di is not None:
        data["DI"] = di
        data["DI Category"] = classify_di(di)
    if utci is not None:
        data["UTCI"] = utci
        data["UTCI Category"] = classify_utci(utci)

    if occupancy_mask is not None:
        occ = occupancy_mask.reindex(data.index).fillna(False)
        data = data.loc[occ]

    agg = {}

    if adaptive_band is not None and {"low", "high"}.issubset(adaptive_band.columns):
        low = adaptive_band["low"].reindex(data.index).interpolate("time").ffill().bfill()
        high = adaptive_band["high"].reindex(data.index).interpolate("time").ffill().bfill()
        comfort_mask = (data["temp"] >= low) & (data["temp"] <= high)
    else:
        lo_hi = comfort_band if comfort_band is not None else (18.0, 26.0)
        comfort_mask = data["temp"].between(lo_hi[0], lo_hi[1])

    agg["hours_total"] = (data["temp"].notna()).astype(int).resample(freq).sum()
    agg["hours_in_comfort_band"] = comfort_mask.astype(int).resample(freq).sum()
    agg["fraction_in_comfort_band"] = agg["hours_in_comfort_band"] / agg["hours_total"].replace(0, np.nan)

    for thresh in sorted(set(overheating_thresholds or [])):
        label = f"overheating_hours_{int(thresh)}C"
        mask = data["temp"] > thresh
        agg[label] = mask.astype(int).resample(freq).sum()

    if cold_thresholds:
        for thresh in sorted(set(cold_thresholds)):
            label = f"cold_hours_below_{int(thresh)}C"
            mask = data["temp"] < thresh
            agg[label] = mask.astype(int).resample(freq).sum()

    if "DI" in data.columns:
        discomfort_mask = data["DI"] >= 24
        agg["hours_di_discomfort"] = discomfort_mask.astype(int).resample(freq).sum()

    if "UTCI Category" in data.columns:
        utci_cat = data["UTCI Category"].copy()
        heat_mask = utci_cat.str.contains("heat", case=False, na=False)
        cold_mask = utci_cat.str.contains("cold", case=False, na=False)
        agg["hours_utci_heat_stress"] = heat_mask.astype(int).resample(freq).sum()
        agg["hours_utci_cold_stress"] = cold_mask.astype(int).resample(freq).sum()

    if percentiles:
        for p in percentiles:
            label = f"temp_p{int(p * 100)}"
            agg[label] = data["temp"].resample(freq).quantile(p)
            if "DI" in data.columns:
                agg[f"di_p{int(p * 100)}"] = data["DI"].resample(freq).quantile(p)
            if "UTCI" in data.columns:
                agg[f"utci_p{int(p * 100)}"] = data["UTCI"].resample(freq).quantile(p)

    result = pd.concat(agg, axis=1)
    return result


# ---------------------------
# Degree metrics
# ---------------------------


def compute_degree_metrics(
    df: pd.DataFrame,
    temp_col: str = "drybulb",
    base_heat: float = 18.0,
    base_cool: float = 26.0,
    freq: str = "D",
) -> pd.DataFrame:
    if temp_col not in df:
        raise KeyError(f"Missing temperature column: {temp_col}")
    temp = df[temp_col].astype(float)
    hdd_hourly = (base_heat - temp).clip(lower=0)
    cdd_hourly = (temp - base_cool).clip(lower=0)

    hourly = pd.DataFrame({
        "HDD_hour": hdd_hourly,
        "CDD_hour": cdd_hourly,
    })
    deg = hourly.resample(freq).sum()
    deg["HDD_day"] = deg["HDD_hour"] / 24.0
    deg["CDD_day"] = deg["CDD_hour"] / 24.0
    return deg


def summarize_loads(deg_df: pd.DataFrame, freq: str = "A") -> pd.DataFrame:
    if deg_df.empty:
        return pd.DataFrame()
    resampled = deg_df.resample(freq).sum()
    resampled = resampled.rename(columns={
        "HDD_hour": "heating_degree_hours",
        "CDD_hour": "cooling_degree_hours",
        "HDD_day": "heating_degree_days",
        "CDD_day": "cooling_degree_days",
    })
    return resampled


# ---------------------------
# Scenario comparison
# ---------------------------


def _scenario_metrics(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    di = None
    try:
        if {"drybulb", "relhum"}.issubset(df.columns):
            di = compute_di(df)
    except Exception:
        di = None
    utci = None
    try:
        utci = compute_utci_approx(df)
    except Exception:
        utci = None
    comfort = summarize_comfort(df, di, utci, freq="A")
    deg = compute_degree_metrics(df)
    loads = summarize_loads(deg)

    parts = []
    if not comfort.empty:
        last = comfort.iloc[-1]
        parts.append(last[[
            "fraction_in_comfort_band",
            "overheating_hours_26C",
            "overheating_hours_28C",
            "overheating_hours_30C",
            "hours_utci_heat_stress",
            "hours_utci_cold_stress",
            "hours_di_discomfort",
        ]].dropna())
    if not loads.empty:
        last_load = loads.iloc[-1]
        parts.append(last_load[[
            "heating_degree_hours",
            "cooling_degree_hours",
            "heating_degree_days",
            "cooling_degree_days",
        ]])

    mean_temp = pd.Series({"mean_temp": float(df["drybulb"].mean())}) if "drybulb" in df else pd.Series({"mean_temp": np.nan})
    parts.append(mean_temp)

    if parts:
        return pd.concat(parts)
    return pd.Series(dtype=float)


def compare_comfort_across_scenarios(scenarios: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = {}
    for name, df in scenarios.items():
        try:
            rows[name] = _scenario_metrics(df)
        except Exception as exc:
            rows[name] = pd.Series({"error": str(exc)})
    return pd.DataFrame(rows).T
