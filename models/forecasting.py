from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:  # pragma: no cover - fallback when statsmodels missing
    SARIMAX = None

import live_sensors as ls


@dataclass
class ForecastResult:
    forecast: pd.DataFrame
    bias_comparison: Optional[pd.DataFrame] = None
    forecast_fig: Optional[go.Figure] = None
    bias_fig: Optional[go.Figure] = None
    overheating_fig: Optional[go.Figure] = None


def load_sensor_data() -> pd.DataFrame:
    """Load stored sensor readings and ensure an hourly index."""
    df = ls.load_sensor_data().copy()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp")
    hourly = df.resample("1H").mean().dropna(how="all")
    return hourly


def load_epw_climatology(cdf: pd.DataFrame) -> pd.DataFrame:
    """Build the EPW climatology with day-of-year and hour bins."""
    return ls.build_epw_climatology(cdf)


def _fit_series(
    series: pd.Series,
    horizon: int,
    confidence_level: float,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    series = series.dropna()
    if len(series) < 48:
        idx = pd.date_range(series.index.max() + pd.Timedelta(hours=1), periods=horizon, freq="1H")
        fallback = pd.Series(series.iloc[-1] if len(series) else np.nan, index=idx)
        return fallback, fallback.copy(), fallback.copy()

    try:
        if SARIMAX is None:
            raise RuntimeError("SARIMAX unavailable")
        alpha = 1.0 - float(confidence_level)
        alpha = float(np.clip(alpha, 0.01, 0.99))
        model = SARIMAX(
            series,
            order=(1, 1, 1),
            seasonal_order=(0, 1, 1, 24),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)
        forecast = fitted.get_forecast(steps=horizon)
        pred = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=alpha)
        lower = conf_int.iloc[:, 0]
        upper = conf_int.iloc[:, 1]
        return pred, lower, upper
    except Exception:
        idx = pd.date_range(series.index.max() + pd.Timedelta(hours=1), periods=horizon, freq="1H")
        fallback = pd.Series(series.iloc[-1], index=idx)
        return fallback, fallback.copy(), fallback.copy()


def build_forecast_model(
    sensor_df: pd.DataFrame,
    horizon_hours: int = 72,
    training_days: int = 14,
    confidence_level: float = 0.8,
) -> pd.DataFrame:
    """Train SARIMAX models per variable and return a consolidated forecast dataframe."""
    if sensor_df.empty:
        return pd.DataFrame(columns=[
            "timestamp", "temp_forecast", "rh_forecast", "ghi_forecast", "lower_ci", "upper_ci"
        ])

    cutoff = sensor_df.index.max() - pd.Timedelta(days=training_days)
    window = sensor_df.loc[sensor_df.index >= cutoff].copy()

    idx = pd.date_range(window.index.max() + pd.Timedelta(hours=1), periods=horizon_hours, freq="1H")
    out = pd.DataFrame({"timestamp": idx})

    temp_pred, lower, upper = _fit_series(
        window.get("temperature", pd.Series(dtype=float)), horizon_hours, confidence_level
    )
    out["temp_forecast"] = temp_pred.reindex(idx).to_numpy(dtype=float)
    out["lower_ci"] = lower.reindex(idx).to_numpy(dtype=float)
    out["upper_ci"] = upper.reindex(idx).to_numpy(dtype=float)

    if "relative_humidity" in window.columns:
        rh_pred, _, _ = _fit_series(window["relative_humidity"], horizon_hours, confidence_level)
        out["rh_forecast"] = rh_pred.reindex(idx).to_numpy(dtype=float)
    else:
        out["rh_forecast"] = np.nan

    if "ghi" in window.columns:
        ghi_pred, _, _ = _fit_series(window["ghi"], horizon_hours, confidence_level)
        out["ghi_forecast"] = ghi_pred.reindex(idx).to_numpy(dtype=float)
    else:
        out["ghi_forecast"] = np.nan

    return out


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
    confidence_level: float = 0.8,
    recent_history: Optional[pd.Series] = None,
) -> go.Figure:
    fig = go.Figure()
    if df_forecast.empty:
        fig.update_layout(title="No forecast available")
        return fig
    band_pct = int(round(confidence_level * 100))
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
        name="Temperature forecast",
        line=dict(color="#60a5fa", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([df_forecast["timestamp"], df_forecast["timestamp"][::-1]]),
        y=pd.concat([df_forecast["upper_ci"], df_forecast["lower_ci"][::-1]]),
        fill="toself",
        fillcolor="rgba(96,165,250,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        name=f"{band_pct}% confidence band"
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
        "lower": float(row.get("lower_ci", np.nan)) if "lower_ci" in row.index else np.nan,
        "upper": float(row.get("upper_ci", np.nan)) if "upper_ci" in row.index else np.nan,
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
        name=">=30°C"
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title="°C",
        template="plotly_dark"
    )
    return fig
