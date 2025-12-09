from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import live_sensors as ls


EPW_COLUMNS = [
    "year","month","day","hour","minute","datasource",
    "drybulb","dewpoint","relhum","atmos_pressure",
    "exthorrad","extdirrad","horirsky",
    "glohorrad","dirnorrad","difhorrad",
    "glohorillum","dirnorillum","difhorillum","zenlum",
    "winddir","windspd","totskycvr","opaqskycvr",
    "visibility","ceiling_hgt","presweathobs","presweathcodes",
    "precip_wtr","aerosol_opt_depth","snowdepth","days_last_snow",
    "albedo","liq_precip_depth","liq_precip_rate"
]

SCENARIO_MAP = {
    "SSP1-2.6": "ssp126",
    "SSP2-4.5": "ssp245",
    "SSP3-7.0": "ssp370",
    "SSP5-8.5": "ssp585",
}

SCENARIO_DESCRIPTIONS = {
    "SSP1-2.6": "Low emissions / sustainability",
    "SSP2-4.5": "Middle of the road",
    "SSP3-7.0": "Regional rivalry (high)",
    "SSP5-8.5": "Fossil-fueled development (very high)",
}

TARGET_YEARS = (2050, 2080)

REFERENCE_DELTA_CANDIDATES = (
    Path(__file__).resolve().parents[1] / "data/cmip6_monthly_deltas.csv",
    Path(__file__).resolve().parents[1] / "weather_cache/cmip6_monthly_deltas.csv",
)


def _lat_band(latitude: Optional[float]) -> Optional[str]:
    if latitude is None or np.isnan(latitude):
        return None
    abs_lat = abs(float(latitude))
    if abs_lat < 23.5:
        return "tropics"
    if abs_lat < 50:
        return "midlat"
    return "highlat"


def _load_reference_deltas(band: Optional[str]) -> Optional[pd.DataFrame]:
    for candidate in REFERENCE_DELTA_CANDIDATES:
        if candidate.exists():
            try:
                table = pd.read_csv(candidate)
            except Exception:
                continue
            required = {"scenario", "year", "month", "delta_temp"}
            if not required.issubset(table.columns):
                continue
            if "lat_band" in table.columns and band:
                subset = table.loc[table["lat_band"].astype(str).str.lower() == band.lower()]
                if not subset.empty:
                    return subset
            return table
    return None


def _synthetic_cmip6_deltas() -> pd.DataFrame:
    months = np.arange(1, 13)
    records = []
    scenario_scale = {
        "ssp126": 0.8,
        "ssp245": 1.0,
        "ssp370": 1.2,
        "ssp585": 1.5,
    }
    year_scale = {2050: 1.0, 2080: 1.4}
    for scenario, scen_mult in scenario_scale.items():
        for year, year_mult in year_scale.items():
            for month in months:
                seasonal = np.sin(((month - 1) / 12.0) * 2 * np.pi) * 0.5 + 1.0
                base = scen_mult * year_mult * seasonal
                records.append({
                    "scenario": scenario,
                    "year": year,
                    "month": month,
                    "delta_temp": round(base * 1.2, 3),
                    "delta_rh": round(-base * 2.0, 3),
                    "delta_wind": round(base * 0.2, 3),
                    "delta_ghi": round(base * 8.0, 3),
                })
    return pd.DataFrame(records)


def load_cmip6_deltas(latitude: Optional[float] = None) -> pd.DataFrame:
    """Return CMIP6 deltas, preferring real tables if supplied on disk."""
    band = _lat_band(latitude)
    reference = _load_reference_deltas(band)
    if reference is not None and not reference.empty:
        return reference
    return _synthetic_cmip6_deltas()


def compute_sensor_bias(
    sensor_clim: pd.DataFrame,
    epw_clim: pd.DataFrame,
) -> Tuple[pd.DataFrame, float]:
    if sensor_clim.empty or epw_clim.empty:
        return pd.DataFrame(columns=["doy", "hour", "temp_bias", "rh_bias"]), 0.0
    merged = pd.merge(epw_clim, sensor_clim, on=["doy", "hour"], how="outer")
    merged["temp_bias"] = merged["sensor_temp"] - merged["epw_temp"]
    merged["rh_bias"] = merged["sensor_rh"] - merged["epw_rh"]
    coverage = float(merged["temp_bias"].notna().sum()) / max(len(merged), 1)
    subset = merged[["doy", "hour", "temp_bias", "rh_bias"]]
    return subset, coverage


def sensor_informed_baseline(epw_df: pd.DataFrame, bias_table: pd.DataFrame) -> pd.DataFrame:
    if bias_table is None or bias_table.empty:
        return epw_df.copy()
    df = epw_df.copy()
    index = pd.MultiIndex.from_arrays(
        [df.index.dayofyear, df.index.hour], names=["doy", "hour"]
    )
    bias_lookup = bias_table.set_index(["doy", "hour"])
    temp_bias = bias_lookup.get("temp_bias")
    rh_bias = bias_lookup.get("rh_bias")

    temp_adj = bias_lookup.reindex(index)["temp_bias"].fillna(0.0) if temp_bias is not None else 0.0
    df["drybulb"] = df["drybulb"] + temp_adj.values if isinstance(temp_adj, pd.Series) else df["drybulb"]

    if "relhum" in df.columns and rh_bias is not None:
        rh_adj = bias_lookup.reindex(index)["rh_bias"].fillna(0.0)
        df["relhum"] = (df["relhum"] + rh_adj.values).clip(0, 100)

    return df


def _dewpoint_from_temp_rh(temp_c: pd.Series, rh: pd.Series) -> pd.Series:
    a, b = 17.625, 243.04
    rh = rh.clip(lower=0.1, upper=100.0)
    gamma = np.log(rh / 100.0) + (a * temp_c) / (b + temp_c)
    return (b * gamma) / (a - gamma)


def apply_cmip6_deltas(
    epw_df: pd.DataFrame,
    cmip6_table: pd.DataFrame,
    scenario: str,
    year: int,
    temp_only: bool = False,
) -> pd.DataFrame:
    df = epw_df.copy()
    deltas = cmip6_table[(cmip6_table["scenario"] == scenario) & (cmip6_table["year"] == year)]
    if deltas.empty:
        return df
    delta_lookup = deltas.set_index("month")
    months = df.index.month

    temp_delta = pd.Series(months, index=df.index).map(delta_lookup["delta_temp"]).fillna(0.0)
    df["drybulb"] = df["drybulb"] + temp_delta.values

    if not temp_only and "relhum" in df.columns and "delta_rh" in delta_lookup:
        rh_delta = pd.Series(months, index=df.index).map(delta_lookup["delta_rh"]).fillna(0.0)
        df["relhum"] = (df["relhum"] + rh_delta.values).clip(0, 100)

    if not temp_only and "glohorrad" in df.columns and "delta_ghi" in delta_lookup:
        ghi_delta = pd.Series(months, index=df.index).map(delta_lookup["delta_ghi"]).fillna(0.0)
        df["glohorrad"] = (df["glohorrad"] + ghi_delta.values).clip(lower=0)

    if not temp_only and "windspd" in df.columns and "delta_wind" in delta_lookup:
        wind_delta = pd.Series(months, index=df.index).map(delta_lookup["delta_wind"]).fillna(0.0)
        df["windspd"] = (df["windspd"] + wind_delta.values).clip(lower=0)

    if {"drybulb", "relhum"}.issubset(df.columns):
        df["dewpoint"] = _dewpoint_from_temp_rh(df["drybulb"], df["relhum"])

    return df


def _ensure_epw_columns(df: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in EPW_COLUMNS:
        if col not in out.columns:
            if col in template.columns:
                out[col] = template[col]
            else:
                out[col] = 0
    return out[EPW_COLUMNS]


def _compose_epw_text(header: dict, df: pd.DataFrame) -> bytes:
    df = _ensure_epw_columns(df, df)
    def _format_cell(value):
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{value:.6f}".rstrip("0").rstrip(".")
        return str(value)

    def _fmt(value, precision=6):
        if value is None:
            return ""
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return ""
            return f"{value:.{precision}f}".rstrip("0").rstrip(".")
        return str(value)

    loc = header.get("location", {})
    loc_line = ",".join([
        "LOCATION",
        loc.get("city", ""),
        loc.get("state_province", ""),
        loc.get("country", ""),
        loc.get("source", ""),
        loc.get("wmo", ""),
        _fmt(loc.get("latitude"), 4),
        _fmt(loc.get("longitude"), 4),
        _fmt(loc.get("timezone"), 2),
        _fmt(loc.get("elevation_m"), 2),
    ])
    header_lines = [
        loc_line,
        header.get("design_conditions", ""),
        header.get("typical_extreme_periods", ""),
        header.get("ground_temps", ""),
        header.get("holidays_dst", ""),
        header.get("data_periods", ""),
        header.get("comments1", ""),
        header.get("comments2", ""),
    ]
    row_strings = []
    for _, row in df[EPW_COLUMNS].iterrows():
        row_strings.append(
            ",".join(_format_cell(row[col]) for col in EPW_COLUMNS)
        )
    data_csv = "\n".join(row_strings) + "\n"
    text = "\n".join(header_lines) + "\n" + data_csv
    return text.encode("latin-1", errors="replace")


def build_future_epw(
    epw_df: pd.DataFrame,
    header: dict,
    scenario_label: str,
    year: int,
    cmip6_table: pd.DataFrame,
    bias_table: Optional[pd.DataFrame] = None,
    use_sensor_baseline: bool = True,
    temp_only: bool = False,
) -> Dict[str, object]:
    scenario = SCENARIO_MAP.get(scenario_label, scenario_label)
    base_cols = epw_df.columns
    df = sensor_informed_baseline(epw_df, bias_table) if use_sensor_baseline else epw_df.copy()
    df = df.reindex(columns=base_cols)
    future_df = apply_cmip6_deltas(df, cmip6_table, scenario, year, temp_only=temp_only)
    future_df = future_df.reindex(columns=base_cols)
    epw_bytes = _compose_epw_text(header, future_df)
    return {"df": future_df, "bytes": epw_bytes}


def generate_download_payloads(
    epw_df: pd.DataFrame,
    header: dict,
    scenario_label: str,
    cmip6_table: pd.DataFrame,
    bias_table: Optional[pd.DataFrame],
    use_sensor_baseline: bool,
    temp_only: bool,
) -> Dict[int, Dict[str, object]]:
    outputs: Dict[int, Dict[str, object]] = {}
    loc = header.get("location", {}) if header else {}
    slug = loc.get("city") or loc.get("state_province") or "site"
    safe_slug = ''.join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in slug).strip("_") or "site"
    for year in TARGET_YEARS:
        bundle = build_future_epw(
            epw_df,
            header,
            scenario_label,
            year,
            cmip6_table,
            bias_table=bias_table,
            use_sensor_baseline=use_sensor_baseline,
            temp_only=temp_only,
        )
        scenario = SCENARIO_MAP.get(scenario_label, scenario_label)
        file_name = f"{safe_slug}_future_{year}_{scenario}.epw"
        outputs[year] = {
            "file_name": file_name,
            "bytes": bundle["bytes"],
            "df": bundle["df"],
        }
    return outputs
