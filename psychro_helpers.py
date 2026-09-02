"""
Psychrometric chart helpers — Givoni bioclimatic zones, heatmap grid, thermo calcs.
Used by render_psychrometrics_page() in app.py.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ──────────────── Thermo helpers (SI) ────────────────

def p_ws_kPa(TC):
    """Saturation vapor pressure over water [kPa] (Magnus/Tetens)."""
    TC = np.asarray(TC, dtype=float)
    return 0.61094 * np.exp(17.625 * TC / (TC + 243.04))

def w_from_Pv_kPa(Pv_kPa, P_kPa):
    Pv_kPa = np.clip(np.asarray(Pv_kPa, dtype=float), 0.0, 0.999 * P_kPa)
    return 0.62198 * Pv_kPa / (P_kPa - Pv_kPa)

def gpkg(w):
    return 1000.0 * np.asarray(w, dtype=float)

def w_sat(TC, P_kPa):
    return w_from_Pv_kPa(p_ws_kPa(TC), P_kPa)

def dew_point_C(TC, RH):
    a, b = 17.625, 243.04
    TC, RH = np.asarray(TC, float), np.asarray(RH, float)
    gamma = np.log(np.clip(RH, 1e-6, 100) / 100.0) + (a * TC) / (b + TC)
    return (b * gamma) / (a - gamma)

def wet_bulb_C(TC, RH):
    TC, RH = np.asarray(TC, float), np.clip(np.asarray(RH, float), 1e-6, 100)
    return (TC * np.arctan(0.151977 * np.sqrt(RH + 8.313659))
            + np.arctan(TC + RH) - np.arctan(RH - 1.676331)
            + 0.00391838 * (RH ** 1.5) * np.arctan(0.023101 * RH) - 4.686035)

def enthalpy_kJkg(TC, w):
    return 1.006 * TC + w * (2501.0 + 1.86 * TC)

def specific_vol(TC, w, P_kPa):
    return 0.287042 * (TC + 273.15) * (1 + 1.6078 * w) / P_kPa

# ──────────────── Givoni bioclimatic zone polygons ────────────────
# Each zone is a list of (T_db °C, w g/kg) vertices forming a closed polygon.
# Based on Givoni (1992) / Milne-Givoni standard references.
# The comfort zone shifts with mean outdoor temperature (Trm).

def _comfort_zone(Trm=20.0):
    """ASHRAE 55-style adaptive comfort zone in dry-bulb/humidity-ratio space."""
    neutral = 0.31 * float(Trm) + 17.8
    t_lo = max(18.0, neutral - 3.5)
    t_hi = min(30.0, neutral + 3.5)
    return [(t_lo, 4.0), (t_lo + 0.5, 12.0), (t_hi, 12.0), (t_hi + 0.5, 4.0)]

def givoni_zones(P_kPa=101.325, Trm=20.0):
    """Return dict of zone_name -> list of (T, w_gpkg) polygon vertices.
    Non-overlapping tiled zones based on Givoni (1992)."""
    cz = _comfort_zone(Trm)
    cz_tlo, cz_thi = cz[0][0], cz[2][0]
    
    def ws(t):
        return gpkg(w_sat(np.array([t]), P_kPa))[0]
    
    zones = {}
    
    # 1. Comfort Zone (core)
    zones["COMFORT ZONE"] = [
        (cz_tlo, 4.0), (cz_tlo, 12.0), (cz_thi, 12.0), (cz_thi, 4.0),
    ]
    
    # 2. Natural Ventilation (above comfort, high humidity)
    zones["NATURAL\nVENTILATION"] = [
        (cz_tlo, 12.0), (cz_tlo, min(ws(cz_tlo), 17.0)),
        (cz_thi + 2, min(ws(cz_thi + 2), 17.0)), (cz_thi + 2, 12.0),
    ]
    
    # 3. Internal Gains (just below comfort temp, same humidity band)
    zones["INTERNAL\nGAINS"] = [
        (12.0, 4.0), (12.0, 12.0), (cz_tlo, 12.0), (cz_tlo, 4.0),
    ]
    
    # 4. Passive Solar Heating (cold side)
    zones["PASSIVE SOLAR\nHEATING"] = [
        (5.0, 4.0), (5.0, 12.0), (12.0, 12.0), (12.0, 4.0),
    ]
    
    # 5. Active Solar (very cold)
    zones["ACTIVE\nSOLAR"] = [
        (-5.0, 4.0), (-5.0, 12.0), (5.0, 12.0), (5.0, 4.0),
    ]
    
    # 6. Heating (coldest region, above humidification)
    zones["HEATING"] = [
        (-15.0, 4.0), (-15.0, 12.0), (-5.0, 12.0), (-5.0, 4.0),
    ]
    
    # 7. Humidification (dry bottom strip, cold to comfort)
    zones["HUMIDIFICATION"] = [
        (-15.0, 0.0), (-15.0, 4.0), (cz_thi, 4.0), (cz_thi, 0.0),
    ]
    
    # 8. Evaporative Cooling (hot + dry)
    zones["EVAPORATIVE\nCOOLING"] = [
        (cz_thi, 0.0), (cz_thi, 4.0), (44.0, 4.0), (44.0, 0.0),
    ]
    
    # 9. Mass Cooling (hot, moderate humidity)
    zones["MASS\nCOOLING"] = [
        (cz_thi, 4.0), (cz_thi, 12.0), (36.0, 12.0), (36.0, 4.0),
    ]
    
    # 10. Mass Cooling & Night Ventilation (very hot)
    zones["NIGHT VENT\n& MASS COOL"] = [
        (36.0, 4.0), (36.0, 12.0), (44.0, 12.0), (44.0, 4.0),
    ]
    
    # 11. Air-Conditioning & Dehumidification (hot + humid, above comfort)
    zones["A/C &\nDEHUMIDIFICATION"] = [
        (cz_thi + 2, 12.0), (cz_thi + 2, min(ws(cz_thi + 2), 25.0)),
        (44.0, min(ws(44.0), 25.0)), (44.0, 12.0),
    ]
    
    return zones


# ──────────────── 2D Heatmap Grid ────────────────

def build_psychro_heatmap(T_pts, Y_gpkg, metric_vals=None,
                          t_step=1.0, w_step=1.0,
                          t_range=(-10, 50), w_range=(0, 30)):
    """Bin hourly data into a 2D frequency/metric grid.
    
    Returns t_edges, w_edges, grid_values (2D array).
    If metric_vals is None, grid_values = frequency count.
    Otherwise grid_values = mean of metric_vals per bin.
    """
    t_edges = np.arange(t_range[0], t_range[1] + t_step, t_step)
    w_edges = np.arange(w_range[0], w_range[1] + w_step, w_step)
    
    T_pts = np.asarray(T_pts, float)
    Y_gpkg = np.asarray(Y_gpkg, float)
    
    if metric_vals is None:
        grid, _, _ = np.histogram2d(T_pts, Y_gpkg, bins=[t_edges, w_edges])
    else:
        metric_vals = np.asarray(metric_vals, float)
        count, _, _ = np.histogram2d(T_pts, Y_gpkg, bins=[t_edges, w_edges])
        total, _, _ = np.histogram2d(T_pts, Y_gpkg, bins=[t_edges, w_edges],
                                      weights=metric_vals)
        with np.errstate(divide='ignore', invalid='ignore'):
            grid = np.where(count > 0, total / count, np.nan)
    
    return t_edges, w_edges, grid.T  # transpose so rows=humidity, cols=temp


def count_hours_in_zones(T_pts, W_gpkg, zones):
    """Count how many hourly points fall inside each Givoni zone polygon."""
    from matplotlib.path import Path
    results = {}
    pts = np.column_stack([T_pts, W_gpkg])
    for name, verts in zones.items():
        poly = Path(verts + [verts[0]])  # close polygon
        inside = poly.contains_points(pts)
        results[name] = int(inside.sum())
    return results


def classify_points_to_zones(T_pts, W_gpkg, zones, priority_order=None):
    """Assign each hourly point to a Givoni zone. Returns array of zone labels.
    Points not in any zone get label 'Unclassified'."""
    from matplotlib.path import Path
    n = len(T_pts)
    labels = np.array(["Unclassified"] * n, dtype=object)
    pts = np.column_stack([T_pts, W_gpkg])
    # Process zones in order (later zones override earlier if overlap)
    order = priority_order or list(zones.keys())
    for name in order:
        verts = zones[name]
        poly = Path(verts + [verts[0]])
        inside = poly.contains_points(pts)
        labels[inside] = name
    return labels


# ──────────────── Chart background line builders ────────────────

def rh_curve(T_axis, rh_pct, P_kPa):
    """Return absolute humidity g/kg for a constant RH% line."""
    Pv = (rh_pct / 100.0) * p_ws_kPa(T_axis)
    return gpkg(w_from_Pv_kPa(Pv, P_kPa))

def enthalpy_w_line(T_axis, h_kJkg):
    """w from enthalpy: h = 1.006T + w(2501+1.86T)"""
    w = (h_kJkg - 1.006 * T_axis) / (2501.0 + 1.86 * T_axis)
    return gpkg(w)

def volume_w_line(T_axis, v_m3kg, P_kPa):
    """w from specific volume."""
    R = 0.287042
    w = (v_m3kg * P_kPa / (R * (T_axis + 273.15)) - 1.0) / 1.6078
    return gpkg(w)

def wetbulb_curve(T_axis, twb_target, P_kPa, n_pts=200):
    """Points along a constant wet-bulb line on the psychrometric chart.
    Returns (T_array, w_gpkg_array) for plotting."""
    # Sweep RH from 100% down to find T,RH pairs where Twb ≈ target
    t_out, w_out = [], []
    for t in T_axis:
        if t < twb_target - 1:
            continue
        # Binary search for RH that gives this wet-bulb at this T
        lo, hi = 0.1, 100.0
        for _ in range(30):
            mid = (lo + hi) / 2
            twb = float(wet_bulb_C(np.array([t]), np.array([mid]))[0])
            if twb < twb_target:
                lo = mid
            else:
                hi = mid
        rh_found = (lo + hi) / 2
        pv = (rh_found / 100.0) * float(p_ws_kPa(np.array([t]))[0])
        w_val = float(w_from_Pv_kPa(np.array([pv]), P_kPa)[0])
        t_out.append(t)
        w_out.append(gpkg(np.array([w_val]))[0])
    return np.array(t_out), np.array(w_out)


def get_design_strategy_polygons(P_kPa=101.325, Trm=20.0):
    """Return dict of strategy_name -> list of (T, w_gpkg) polygon vertices.
    These define the boundary polygons for the 4 climate design strategies.
    """
    cz = _comfort_zone(Trm)
    cz_tlo, cz_thi = cz[0][0], cz[2][0]
    
    def ws(t):
        return gpkg(w_sat(np.array([t]), P_kPa))[0]

    strategies = {}
    
    # 1. Natural Ventilation (covers comfort zone + ventilation cooling zone)
    # Typically cz_tlo to cz_thi + 2°C, and humidity 4.0 to min(ws(T), 12.0) g/kg
    strategies["Natural Ventilation"] = [
        (cz_tlo, 4.0),
        (cz_tlo, min(ws(cz_tlo), 12.0)),
        (cz_thi + 2, min(ws(cz_thi + 2), 12.0)),
        (cz_thi + 2, 4.0)
    ]
    
    # 2. Direct Evaporative Cooling (hot, dry region)
    # Typically cz_thi to 44°C, humidity 0 to 8-10 g/kg
    strategies["Direct Evaporative Cooling"] = [
        (cz_thi, 0.0),
        (cz_thi, 8.0),
        (34.0, 10.0),
        (44.0, 4.0),
        (44.0, 0.0)
    ]
    
    # 3. Heating (all cold hours below comfort limit)
    # Typically -15°C to cz_tlo, humidity 0 to 12.0 g/kg
    strategies["Heating"] = [
        (-15.0, 0.0),
        (-15.0, 12.0),
        (cz_tlo, 12.0),
        (cz_tlo, 0.0)
    ]
    
    # 4. Dehumidification (warm, humid region above comfort limits)
    # Typically cz_tlo to 44°C, humidity > 12.0 g/kg up to saturation line (clipped at 25 g/kg)
    t_vals = np.linspace(cz_tlo, 44.0, 8)
    sat_pts = [(t, min(ws(t), 25.0)) for t in t_vals]
    strategies["Dehumidification"] = [
        (cz_tlo, 12.0)
    ] + sat_pts + [
        (44.0, 12.0)
    ]
    
    return strategies

# ──────────────── Unified Strategy Zones ────────────────

# Color palette for the 16 strategies (high-contrast, dark‑theme friendly)
strategy_color_map = {
    "Unclassified": "#94a3b8",
    "Dehumidification": "#9467bd",
    "Active Solar": "#e377c2",
    "Passive Solar Heating": "#bcbd22",
    "Internal Gains": "#8c564b",
    "Comfort Zone": "#2ca02c",
    "Mass Cooling": "#ff7f0e",
    "Evaporative Cooling": "#1f77b4",
    "Natural Ventilation": "#17becf",
    "Heating": "#f97316",
    "Humidification": "#3b82f6",
    "Direct Evaporative Cooling": "#ff9896",
    "Night Ventilation & Mass Cooling": "#c5b0d5",
    "A/C & Dehumidification": "#8c564b",
    "Passive Solar Heating (Alt)": "#d62728",
    "Active Solar (Alt)": "#2ca02c",
}

def get_all_strategy_zones(P_kPa=101.325, Trm=20.0):
    """Return the 16 Climate Consultant-style design strategy polygons."""
    cz = _comfort_zone(Trm)
    cz_tlo = cz[0][0]
    cz_thi = cz[2][0]

    def ws(t, cap=28.0):
        return min(float(gpkg(w_sat(np.array([t]), P_kPa))[0]), cap)

    strategies = [
        (
            "2",
            "Sun Shading of Windows",
            [(cz_thi - 1.0, 12.0), (cz_thi + 5.0, 12.0), (cz_thi + 6.0, ws(cz_thi + 6.0, 24.0)), (cz_thi, ws(cz_thi, 22.0))],
            "#ff0000",
        ),
        (
            "3",
            "High Thermal Mass",
            [(cz_thi - 0.5, 4.0), (cz_thi - 0.5, 12.0), (35.0, 12.0), (35.0, 4.0)],
            "#ff9900",
        ),
        (
            "4",
            "High Thermal Mass Night Flushed",
            [(cz_thi + 1.5, 4.0), (cz_thi + 1.5, 12.0), (40.0, 12.0), (40.0, 4.0)],
            "#ff7a00",
        ),
        (
            "5",
            "Direct Evaporative Cooling",
            [(cz_thi, 0.0), (cz_thi, 9.0), (35.0, 10.5), (40.0, 5.0), (40.0, 0.0)],
            "#0066ff",
        ),
        (
            "6",
            "Two-Stage Evaporative Cooling",
            [(cz_thi, 0.0), (cz_thi, 13.0), (40.0, 13.0), (40.0, 0.0)],
            "#0033cc",
        ),
        (
            "7",
            "Adaptive Comfort Ventilation",
            [(cz_thi - 0.5, 7.0), (cz_thi - 0.5, 17.0), (29.0, 17.0), (29.0, 7.0)],
            "#008000",
        ),
        (
            "8",
            "Fan-Forced Ventilation Cooling",
            [(cz_thi, 8.0), (cz_thi, 19.0), (32.0, 19.0), (32.0, 8.0)],
            "#00a000",
        ),
        (
            "9",
            "Internal Heat Gain",
            [(12.0, 4.0), (12.0, 12.0), (cz_tlo, 12.0), (cz_tlo, 4.0)],
            "#cc6600",
        ),
        (
            "10",
            "Passive Solar Direct Gain Low Mass",
            [(5.0, 4.0), (5.0, 12.0), (12.0, 12.0), (12.0, 4.0)],
            "#ff00ff",
        ),
        (
            "11",
            "Passive Solar Direct Gain High Mass",
            [(0.0, 4.0), (0.0, 12.0), (12.0, 12.0), (12.0, 4.0)],
            "#9900ff",
        ),
        (
            "12",
            "Wind Protection of Outdoor Spaces",
            [(-10.0, 4.0), (-10.0, 12.0), (5.0, 12.0), (5.0, 4.0)],
            "#555500",
        ),
        (
            "13",
            "Humidification Only",
            [(-10.0, 0.0), (-10.0, 4.0), (cz_thi, 4.0), (cz_thi, 0.0)],
            "#00cccc",
        ),
        (
            "14",
            "Dehumidification Only",
            [(cz_tlo, 12.0), (cz_tlo, 24.0), (cz_thi + 2.0, 24.0), (cz_thi + 2.0, 12.0)],
            "#00a6ff",
        ),
        (
            "15",
            "Cooling, add Dehumidification if needed",
            [(cz_thi + 2.0, 12.0), (cz_thi + 2.0, 28.0), (40.0, 28.0), (40.0, 12.0)],
            "#ff0000",
        ),
        (
            "16",
            "Heating, add Humidification if needed",
            [(-10.0, 0.0), (-10.0, 12.0), (cz_tlo, 12.0), (cz_tlo, 0.0)],
            "#ff0000",
        ),
    ]

    return {
        sid: {"name": name, "polygon": polygon, "color": color}
        for sid, name, polygon, color in strategies
    }

def compute_centroids(zones_dict):
    """Compute simple centroid (average of vertices) for each zone.
    Returns a dict mapping zone id (string) to (x, y) coordinates.
    """
    centroids = {}
    for zid, info in zones_dict.items():
        verts = info["polygon"]
        if not verts:
            continue
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        centroids[zid] = (sum(xs) / len(xs), sum(ys) / len(ys))
    return centroids
