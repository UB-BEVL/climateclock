"""Annual hourly resource charts with consistent thresholds and missing data."""
from __future__ import annotations

import calendar
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEFAULT_THRESHOLDS = {
    "solar": [100.0, 300.0, 500.0, 700.0],
    "humidity": [40.0, 60.0, 80.0],
    "wind": [1.5, 4.5],
}


def _range_labels(edges, unit):
    return ([f"<{edges[0]:g}{unit}"]
            + [f"{a:g}–<{b:g}{unit}" for a, b in zip(edges, edges[1:])]
            + [f"≥{edges[-1]:g}{unit}"])


def resource_grids(frame: pd.DataFrame, thresholds: dict | None = None):
    """Aggregate actual values before categorizing; never change the source frame."""
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.empty:
        return {}, pd.DatetimeIndex([])
    thresholds = thresholds or DEFAULT_THRESHOLDS
    for values in thresholds.values():
        if not np.isfinite(values).all() or any(a >= b for a, b in zip(values, values[1:])):
            raise ValueError("Thresholds must be finite and strictly increasing.")
    leap = bool(((frame.index.month == 2) & (frame.index.day == 29)).any())
    year = 2020 if leap else 2021
    dates = pd.date_range(f"{year}-01-01", periods=366 if leap else 365, freq="D")
    # EPW typical years may use a different source year for each month.
    normalized = pd.to_datetime({"year": year, "month": frame.index.month, "day": frame.index.day})
    day = normalized.dt.dayofyear.to_numpy()
    specs = [
        ("Dry Bulb Temperature", ["drybulb", "dry_bulb_temperature", "temp", "db"], [10, 20, 26.6, 35], "°C", (-100, 70), ["#6baed6", "#bdd7e7", "#f1f5f9", "#fd8d3c", "#e31a1c"]),
        ("Solar Radiation", ["glohorrad", "global_horizontal_radiation", "solar"], thresholds["solar"], " W/m²", (0, 2000), ["#feedde", "#fdbe85", "#fd8d3c", "#e6550d", "#a63603"]),
        ("Humidity", ["relhum", "relative_humidity", "rh"], thresholds["humidity"], "%", (0, 100), ["#fd8d3c", "#fef0d9", "#bdd7e7", "#3182bd"]),
        ("Precipitation", ["liq_precip_depth", "liqprecipdepth", "liquid_precipitation_depth", "liquidprecipitationdepth", "liquid_precip_depth", "precip_depth", "precipdepth", "precipitation_depth", "precipitation", "rainfall", "rain", "precip"], [0.1, 2.5, 10], " mm", (0, 900), ["#f1f5f9", "#93c5fd", "#2563eb", "#1e3a8a"]),
        ("Wind Speed", ["windspd", "wind_speed", "wspd"], thresholds["wind"], " m/s", (0, 100), ["#bde0fe", "#f1f5f9", "#74c476"]),
    ]
    grids = {}
    for title, aliases, edges, unit, valid, colors in specs:
        column = next((name for name in aliases if name in frame), None)
        if column is None:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        values = values.where(values.between(*valid)).to_numpy()
        work = pd.DataFrame({"hour": frame.index.hour, "day": day, "value": values})
        pivot = work.pivot_table(index="hour", columns="day", values="value", aggfunc="mean").reindex(index=range(24), columns=range(1, len(dates) + 1))
        if not np.isfinite(pivot.to_numpy()).any():
            continue
        raw = pivot.to_numpy()
        codes = np.searchsorted(edges, raw, side="right").astype(float)
        codes[~np.isfinite(raw)] = np.nan
        labels = _range_labels(edges, unit)
        hover = np.full(raw.shape, None, dtype=object)
        for i, label in enumerate(labels):
            hover[codes == i] = label
        grids[title] = {"codes": codes, "raw": raw, "labels": labels, "hover": hover, "colors": colors, "unit": unit}
    column = next((name for name in ["winddir", "wind_direction", "wd", "wdir", "wind_dir", "HourlyWindDirection"] if name in frame), None)
    if column:
        compass = dict(zip(["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"], np.arange(0, 360, 22.5)))
        text = frame[column].astype(str).str.strip().str.upper()
        values = pd.to_numeric(text, errors="coerce").fillna(text.map(compass))
        values = values.where(values.between(0, 360))
        codes = ((values % 360 + 22.5) // 45) % 8
        work = pd.DataFrame({"hour": frame.index.hour, "day": day, "value": codes.to_numpy()})
        pivot = work.pivot_table(index="hour", columns="day", values="value", aggfunc=lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan).reindex(index=range(24), columns=range(1, len(dates) + 1))
        raw = pivot.to_numpy()
        if np.isfinite(raw).any():
            labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
            hover = np.full(raw.shape, None, dtype=object)
            for i, label in enumerate(labels):
                hover[raw == i] = label
            grids["Wind Direction"] = {"codes": raw, "raw": raw, "labels": labels, "hover": hover, "colors": ["#4c6fff", "#3fb3ff", "#36d1a8", "#8bd36b", "#f6c445", "#f08c42", "#e15b9a", "#9d6bff"], "unit": ""}
    return grids, dates


def build_resource_heatmap_figure(frame: pd.DataFrame, thresholds: dict | None = None, *, metrics=None):
    grids, dates = resource_grids(frame, thresholds)
    if metrics is not None:
        grids = {title: grid for title, grid in grids.items() if title in metrics}
    if not grids:
        return None
    count = len(grids)
    # Printed strips need extra room between month ticks and the next title.
    spacing = 0.12 if metrics is not None else 0.065
    fig = make_subplots(rows=count, cols=1, shared_xaxes=True, vertical_spacing=spacing, subplot_titles=list(grids))
    for row, (title, grid) in enumerate(grids.items(), 1):
        n = len(grid["colors"])
        scale = [[p, color] for i, color in enumerate(grid["colors"]) for p in [i / n, (i + 1) / n]]
        axis = getattr(fig.layout, f"yaxis{row if row > 1 else ''}")
        bottom, top = axis.domain
        fig.add_trace(go.Heatmap(
            z=grid["codes"], x=dates, y=list(range(24)), customdata=grid["hover"],
            zmin=-0.5, zmax=n - 0.5, zsmooth=False, hoverongaps=False,
            colorscale=scale, colorbar=dict(x=1.01, y=(bottom + top) / 2, len=top - bottom, thickness=13,
                tickvals=list(range(n)), ticktext=grid["labels"], tickfont=dict(size=10)),
            hovertemplate=f"<b>%{{x|%b %d}} · %{{y}}:00</b><br>{title}: %{{customdata}}<extra></extra>",
        ), row=row, col=1)
        fig.update_yaxes(tickvals=[0, 12, 23], ticktext=["00:00", "12:00", "23:00"], autorange="reversed", row=row, col=1)
        fig.update_xaxes(tickvals=[pd.Timestamp(dates[0].year, m, 15) for m in range(1, 13)],
            ticktext=list(calendar.month_abbr)[1:], showticklabels=True, showgrid=False, row=row, col=1)
    fig.update_layout(autosize=True, height=170 * count + 70, margin=dict(l=48, r=138, t=45, b=40),
        template="plotly_white", font=dict(family="Arial", size=11, color="#334155"), showlegend=False)
    fig.update_annotations(font=dict(size=13, color="#334155"), x=0, xanchor="left")
    return fig


def build_resource_report_figures(frame: pd.DataFrame, thresholds: dict | None = None):
    """Use two landscape pages so six legends remain readable in print."""
    result = {}
    for title, metrics in (
        ("Annual Diurnal Resource Heatmap", ["Dry Bulb Temperature", "Solar Radiation", "Humidity"]),
        ("Annual Diurnal Resource Heatmap — Rain and Wind", ["Precipitation", "Wind Speed", "Wind Direction"]),
    ):
        fig = build_resource_heatmap_figure(frame, thresholds, metrics=metrics)
        if fig is not None:
            fig.update_layout(width=1000, height=560, font=dict(size=13), margin=dict(l=55, r=145, t=45, b=35))
            fig.update_xaxes(tickfont=dict(size=12))
            fig.update_yaxes(tickfont=dict(size=12))
            fig.update_annotations(font=dict(size=15))
            fig.update_traces(colorbar_tickfont_size=12)
            result[title] = fig
    return result
