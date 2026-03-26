from __future__ import annotations

import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go

import live_sensors as ls


def get_weather_emoji(wmo_code: int) -> str:
    """Map WMO weather codes to universally understood emojis."""
    if wmo_code == 0:
        return "☀️"
    elif wmo_code in [1, 2, 3]:
        return "⛅"
    elif wmo_code in [45, 48]:
        return "🌫️"
    elif wmo_code in [51, 53, 55, 56, 57]:
        return "🌧️"
    elif wmo_code in [61, 63, 65, 66, 67]:
        return "🌧️"
    elif wmo_code in [71, 73, 75, 77]:
        return "❄️"
    elif wmo_code in [80, 81, 82]:
        return "🌦️"
    elif wmo_code in [85, 86]:
        return "🌨️"
    elif wmo_code in [95, 96, 99]:
        return "⛈️"
    else:
        return "🌡️"


def fetch_openmeteo_10day_forecast(lat: float, lon: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch 10-day hourly deterministic forecast & daily aggregates from Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,surface_pressure,wind_speed_10m,wind_direction_10m,shortwave_radiation,direct_normal_irradiance,diffuse_radiation",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
        "wind_speed_unit": "ms",
        "forecast_days": 10,
        "timezone": "UTC"
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    
    data = payload["hourly"]
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
    
    daily_data = payload["daily"]
    daily_df = pd.DataFrame({
        "date": pd.to_datetime(daily_data["time"]),
        "weather_code": daily_data["weather_code"],
        "temp_max": daily_data["temperature_2m_max"],
        "temp_min": daily_data["temperature_2m_min"],
        "precip_sum": daily_data["precipitation_sum"],
        "precip_prob": daily_data["precipitation_probability_max"]
    })
    
    daily_df["emoji"] = daily_df["weather_code"].apply(get_weather_emoji)
    
    return df, daily_df


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


def plot_bias(df: pd.DataFrame) -> go.Figure:
    """Plot difference between forecast and EPW."""
    fig = go.Figure()
    
    if df.empty or "epw_temp_bias_forecast" not in df.columns:
        return fig

    # Split into positive (hotter forecast) and negative (colder forecast) bias
    pos_mask = df["epw_temp_bias_forecast"] > 0
    neg_mask = df["epw_temp_bias_forecast"] <= 0

    # Hotter than EPW
    fig.add_trace(go.Bar(
        x=df.index[pos_mask],
        y=df.loc[pos_mask, "epw_temp_bias_forecast"],
        marker_color='#ef4444',  # Red
        name="Forecast Hotter (+)"
    ))

    # Colder than EPW
    fig.add_trace(go.Bar(
        x=df.index[neg_mask],
        y=df.loc[neg_mask, "epw_temp_bias_forecast"],
        marker_color='#3b82f6',  # Blue
        name="Forecast Colder (-)"
    ))

    fig.update_layout(
        title="Hourly Temperature Bias (Next 10 Days)",
        yaxis_title="Delta (°C)",
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=False, linecolor="rgba(255,255,255,0.2)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
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


def calculate_degree_days(daily_df: pd.DataFrame, base_temp: float = 18.0) -> tuple[float, float]:
    """Calculate Heating Degree Days (HDD) and Cooling Degree Days (CDD)."""
    if daily_df.empty:
        return 0.0, 0.0

    hdd_total = 0.0
    cdd_total = 0.0

    for _, row in daily_df.iterrows():
        # Simplified NOAA method: T_mean = (T_max + T_min) / 2.
        # Bias is typically ±0.5 per day vs integration-based methods.
        tmean = (row["temp_max"] + row["temp_min"]) / 2.0
        if tmean < base_temp:
            hdd_total += (base_temp - tmean)
        elif tmean > base_temp:
            cdd_total += (tmean - base_temp)

    return hdd_total, cdd_total


def plot_solar_potential(df: pd.DataFrame) -> go.Figure:
    """Plot Global Horizontal (GHI), Direct Normal (DNI), and Diffuse (DHI) Irradiance overlay."""
    fig = go.Figure()
    if df.empty or "ghi_forecast" not in df.columns:
        return fig

    # Filled area for Global Horizontal Irradiance (GHI)
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["ghi_forecast"],
        mode="lines",
        name="Global Horizontal (GHI)",
        line=dict(color="#fbbf24", width=2),
        fill="tozeroy",
        fillcolor="rgba(251, 191, 36, 0.2)"
    ))

    # Line for Direct Normal Irradiance (DNI)
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["dni_forecast"],
        mode="lines",
        name="Direct Normal (DNI)",
        line=dict(color="#f97316", width=2, dash="dot")
    ))

    # Line for Diffuse Horizontal Irradiance (DHI)
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["dhi_forecast"],
        mode="lines",
        name="Diffuse Horizontal (DHI)",
        line=dict(color="#60a5fa", width=2, dash="dash")
    ))

    fig.update_layout(
        title="Solar Radiation Forecast (W/m²)",
        yaxis_title="Irradiance (W/m²)",
        xaxis_title="Time",
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=False, linecolor="rgba(255,255,255,0.2)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    return fig
