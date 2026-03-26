from __future__ import annotations

import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go

import live_sensors as ls


def fetch_openmeteo_10day_forecast(lat: float, lon: float) -> pd.DataFrame:
    """Fetch 10-day hourly deterministic forecast from Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,surface_pressure,wind_speed_10m,wind_direction_10m,shortwave_radiation,direct_normal_irradiance,diffuse_radiation",
        "wind_speed_unit": "ms",
        "forecast_days": 10,
        "timezone": "UTC"
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()["hourly"]

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["time"]),
        "temp_forecast": data["temperature_2m"],
        "rh_forecast": data["relative_humidity_2m"],
        "dew_forecast": data["dew_point_2m"],
        "pressure_forecast": data["surface_pressure"],  # hPa
        "windspd_forecast": data["wind_speed_10m"],    # m/s
        "winddir_forecast": data["wind_direction_10m"],
        "ghi_forecast": data["shortwave_radiation"],
        "dni_forecast": data["direct_normal_irradiance"],
        "dhi_forecast": data["diffuse_radiation"],
    })
    
    # Convert pressure from hPa to Pa
    df["pressure_forecast"] = df["pressure_forecast"] * 100.0
    return df


def build_forecast_epw_dataframe(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Map Open-Meteo columns into the standard 35 EPW columns format for EPW generation."""
    epw_cols = [
        "year", "month", "day", "hour", "minute", "datasource", "drybulb", "dewpoint",
        "relhum", "atmos_pressure", "exthorrad", "extdirrad", "horirsky", "glohorrad",
        "dirnorrad", "difhorrad", "glohorillum", "dirnorillum", "difhorillum", "zenlum",
        "winddir", "windspd", "totskycvr", "opaqskycvr", "visibility", "ceiling_hgt",
        "presweathobs", "presweathcodes", "precip_wtr", "aerosol_opt_depth", "snowdepth",
        "days_last_snow", "albedo", "liq_precip_depth", "liq_precip_rate"
    ]
    
    out = pd.DataFrame(index=forecast_df.index, columns=epw_cols)
    out["year"] = forecast_df["timestamp"].dt.year
    out["month"] = forecast_df["timestamp"].dt.month
    out["day"] = forecast_df["timestamp"].dt.day
    out["hour"] = forecast_df["timestamp"].dt.hour + 1  # EPW convention is 1-24
    out["minute"] = 60
    out["datasource"] = "OpenMeteo-NWP"
    
    out["drybulb"] = forecast_df["temp_forecast"]
    out["dewpoint"] = forecast_df["dew_forecast"]
    out["relhum"] = forecast_df["rh_forecast"]
    out["atmos_pressure"] = forecast_df["pressure_forecast"]
    out["glohorrad"] = forecast_df["ghi_forecast"]
    out["dirnorrad"] = forecast_df["dni_forecast"]
    out["difhorrad"] = forecast_df["dhi_forecast"]
    out["windspd"] = forecast_df["windspd_forecast"]
    out["winddir"] = forecast_df["winddir_forecast"]
    
    # Fill missing EPW fields with standard missing values
    out["exthorrad"] = 9999
    out["extdirrad"] = 9999
    out["horirsky"] = 9999
    out["glohorillum"] = 999999
    out["dirnorillum"] = 999999
    out["difhorillum"] = 999999
    out["zenlum"] = 9999
    out["totskycvr"] = 99
    out["opaqskycvr"] = 99
    out["visibility"] = 9999
    out["ceiling_hgt"] = 99999
    out["presweathobs"] = 9
    out["presweathcodes"] = 999999999
    out["precip_wtr"] = 999
    out["aerosol_opt_depth"] = 0.999
    out["snowdepth"] = 999
    out["days_last_snow"] = 99
    out["albedo"] = 999
    out["liq_precip_depth"] = 999
    out["liq_precip_rate"] = 999
    
    # Adjust hour 24 convention
    mask_24 = out["hour"] == 25
    out.loc[mask_24, "hour"] = 1
    
    return out


def load_epw_climatology(cdf: pd.DataFrame) -> pd.DataFrame:
    """Extract baseline EPW temperatures to compute forecast bias by day-of-year and hour."""
    if cdf is None or cdf.empty or "drybulb" not in cdf:
        return pd.DataFrame(columns=["doy", "hour", "epw_temp"])
        
    df = cdf.copy()
    if pd.api.types.is_datetime64_any_dtype(df.index):
        df["doy"] = df.index.dayofyear
        df["hour"] = df.index.hour
    else:
        df["doy"] = df.get("day_of_year", df.index // 24 + 1)
        df["hour"] = df.get("hour", df.index % 24)
        
    df["epw_temp"] = df["drybulb"]
    return df[["doy", "hour", "epw_temp"]].drop_duplicates(["doy", "hour"])


def compare_forecast_to_epw(df_forecast: pd.DataFrame, df_epw_clim: pd.DataFrame) -> pd.DataFrame:
    if df_forecast.empty or df_epw_clim.empty:
        return pd.DataFrame()
    merged = df_forecast.copy()
    merged["doy"] = pd.to_datetime(merged["timestamp"]).dt.dayofyear
    merged["hour"] = pd.to_datetime(merged["timestamp"]).dt.hour
    merged = merged.merge(df_epw_clim, on=["doy", "hour"], how="left")
    merged["epw_typical_temp"] = merged["epw_temp"]
    merged["epw_temp_bias_forecast"] = merged["temp_forecast"] - merged["epw_typical_temp"]
    return merged


def plot_forecast(
    df_forecast: pd.DataFrame,
    recent_history: Optional[pd.Series] = None,
) -> go.Figure:
    fig = go.Figure()
    if df_forecast.empty:
        fig.update_layout(title="No forecast available")
        return fig
    if recent_history is not None:
        history = recent_history.dropna().sort_index()
        if not history.empty:
            fig.add_trace(go.Scatter(
                x=history.index,
                y=history.values,
                mode="lines",
                name="Observed temperature",
                line=dict(color="#d1d5db", width=1.5, dash="dot"),
                hovertemplate="Observed %{x|%b %d %H:%M}<br>%{y:.1f} °C<extra></extra>",
                legendgroup="history",
            ))
    fig.add_trace(go.Scatter(
        x=df_forecast["timestamp"],
        y=df_forecast["temp_forecast"],
        mode="lines",
        name="10-Day Deterministic Forecast",
        line=dict(color="#60a5fa", width=2)
    ))
    fig.update_layout(
        height=360,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        yaxis_title="°C",
        template="plotly_dark"
    )
    return fig


def summarize_peak_event(df_forecast: pd.DataFrame) -> Optional[dict]:
    """Return the timestamp and bounds for the warmest predicted hour."""
    if df_forecast.empty or "temp_forecast" not in df_forecast:
        return None
    valid = df_forecast.dropna(subset=["temp_forecast"])
    if valid.empty:
        return None
    idx = valid["temp_forecast"].idxmax()
    row = valid.loc[idx]
    timestamp = pd.to_datetime(row.get("timestamp"))
    return {
        "timestamp": timestamp,
        "temp": float(row.get("temp_forecast", np.nan)),
    }


def plot_bias(df_bias: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df_bias.empty:
        fig.update_layout(title="No bias data available", template="plotly_dark")
        return fig
    fig.add_trace(go.Bar(
        x=df_bias["timestamp"],
        y=df_bias["epw_temp_bias_forecast"],
        marker_color="#f97316",
        name="Forecast - EPW"
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis_title="ΔT (°C)",
        template="plotly_dark"
    )
    return fig


def plot_overheating(df_forecast: pd.DataFrame, threshold: float = 30.0) -> go.Figure:
    fig = go.Figure()
    if df_forecast.empty:
        fig.update_layout(title="No overheating data", template="plotly_dark")
        return fig
    mask = df_forecast["temp_forecast"] >= threshold
    if not mask.any():
        fig.update_layout(title="No overheating risk detected", template="plotly_dark")
        return fig
    flagged = df_forecast.loc[mask]
    fig.add_trace(go.Scatter(
        x=flagged["timestamp"],
        y=flagged["temp_forecast"],
        mode="markers",
        marker=dict(size=10, color="#ef4444"),
        name=">= Threshold"
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title="°C",
        template="plotly_dark"
    )
    return fig
