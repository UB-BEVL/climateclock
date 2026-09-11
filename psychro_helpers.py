"""
Psychrometric properties (ASHRAE 2017 SI equations) and screening overlays.
Used by render_psychrometrics_page() in app.py.

Thermodynamic equations: ASHRAE Handbook Fundamentals (2017), chapter 1,
as documented by https://psychrometrics.github.io/psychrolib/api_docs.html.
The legacy strategy polygons below are illustrative, not validated comfort
or building-performance models. Polygon membership is only climate screening.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ──────────────── Thermo helpers (SI) ────────────────

def p_ws_kPa(TC):
    """Saturation pressure [kPa], over ice below 0.01 C, water above.

    ASHRAE 2017 ch. 1, equations 5 and 6; validity -100 to 200 C.
    Invalid inputs return NaN rather than extrapolated physical properties.
    """
    TC = np.asarray(TC, dtype=float)
    valid = np.isfinite(TC) & (TC >= -100) & (TC <= 200)
    t = np.where(valid, TC, 0.) + 273.15
    ice = (-5.6745359e3/t + 6.3925247 - 9.677843e-3*t + 6.2215701e-7*t**2
           + 2.0747825e-9*t**3 - 9.484024e-13*t**4 + 4.1635019*np.log(t))
    water = (-5.8002206e3/t + 1.3914993 - 4.8640239e-2*t + 4.1764768e-5*t**2
             - 1.4452093e-8*t**3 + 6.5459673*np.log(t))
    return np.where(valid, np.exp(np.where(TC <= .01, ice, water))/1000., np.nan)

def w_from_Pv_kPa(Pv_kPa, P_kPa):
    pv, pressure = np.broadcast_arrays(np.asarray(Pv_kPa, float), np.asarray(P_kPa, float))
    valid = np.isfinite(pv) & np.isfinite(pressure) & (pressure > 0) & (pv >= 0) & (pv < pressure)
    return np.where(valid, .621945 * pv / np.where(valid, pressure-pv, 1.), np.nan)

def gpkg(w):
    return 1000.0 * np.asarray(w, dtype=float)

def w_sat(TC, P_kPa):
    return w_from_Pv_kPa(p_ws_kPa(TC), P_kPa)

def dew_point_C(TC, RH):
    """Invert the same saturation equation used by the chart (ice/water)."""
    t, rh = np.broadcast_arrays(np.asarray(TC, float), np.asarray(RH, float))
    pv = rh / 100. * p_ws_kPa(t)
    valid = np.isfinite(pv) & (rh > 0) & (rh <= 100) & (pv >= p_ws_kPa(-100.))
    lo, hi = np.full(t.shape, -100.), np.where(valid, t, 0.)
    for _ in range(40):
        mid = (lo + hi) / 2
        below = p_ws_kPa(mid) < pv
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return np.where(valid, (lo + hi)/2, np.nan)

def _w_from_wet_bulb(TC, TWB, P_kPa):
    """Humidity ratio from thermodynamic wet bulb, ASHRAE eqs. 33/35."""
    t, tw = np.asarray(TC, float), np.asarray(TWB, float)
    ws = w_sat(tw, P_kPa)
    water = ((2501.-2.326*tw)*ws - 1.006*(t-tw)) / (2501.+1.86*t-4.186*tw)
    ice = ((2830.-.24*tw)*ws - 1.006*(t-tw)) / (2830.+1.86*t-2.1*tw)
    return np.where(tw >= 0, water, ice)


def wet_bulb_C(TC, RH, P_kPa=101.325):
    """Pressure-aware thermodynamic wet bulb, solved to <0.001 C."""
    t, rh, p = np.broadcast_arrays(np.asarray(TC, float), np.asarray(RH, float), np.asarray(P_kPa, float))
    w = w_from_Pv_kPa(rh/100. * p_ws_kPa(t), p)
    valid = np.isfinite(w) & (rh >= 0) & (rh <= 100) & (p_ws_kPa(t) < p)
    lo, hi = np.full(t.shape, -100.), np.where(valid, t, 0.)
    for _ in range(40):
        mid = (lo + hi)/2
        below = _w_from_wet_bulb(t, mid, p) < w
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return np.where(valid, (lo + hi)/2, np.nan)

def enthalpy_kJkg(TC, w):
    return 1.006 * TC + w * (2501.0 + 1.86 * TC)

def specific_vol(TC, w, P_kPa):
    return 0.287042 * (TC + 273.15) * (1 + 1.607858 * w) / P_kPa

# ──────────────── Givoni bioclimatic zone polygons ────────────────
# Each zone is a list of (T_db °C, w g/kg) vertices forming a closed polygon.
# Illustrative legacy polygons inspired by bioclimatic charts; no claim that
# these vertices reproduce Givoni, Climate Consultant or ASHRAE 55 boundaries.

def _comfort_zone(Trm=20.0):
    """Illustrative dry-bulb reference, not an ASHRAE 55 compliance zone.

    Temperature limits use the adaptive 80% band; humidity bounds 4-12 g/kg
    are separate screening assumptions. Outdoor dry bulb is not indoor
    operative temperature. Use the interactive workspace for comfort models.
    """
    if not 10.0 <= float(Trm) <= 33.5:
        raise ValueError("Outdoor reference temperature must be between 10 and 33.5 C.")
    neutral = 0.31 * float(Trm) + 17.8
    t_lo, t_hi = neutral - 3.5, neutral + 3.5
    return [(t_lo, 4.0), (t_lo, 12.0), (t_hi, 12.0), (t_hi, 4.0)]

def givoni_zones(P_kPa=101.325, Trm=20.0):
    """Return dict of zone_name -> list of (T, w_gpkg) polygon vertices.
    Illustrative regions only; they may overlap."""
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
    w = (v_m3kg * P_kPa / (R * (T_axis + 273.15)) - 1.0) / 1.607858
    return gpkg(w)

def wetbulb_curve(T_axis, twb_target, P_kPa, n_pts=200):
    """Points along a constant wet-bulb line on the psychrometric chart.
    Returns (T_array, w_gpkg_array) for plotting."""
    t = np.asarray(T_axis, float)
    w = _w_from_wet_bulb(t, twb_target, P_kPa)
    valid = (t >= twb_target) & np.isfinite(w) & (w >= 0) & (w <= w_sat(t, P_kPa))
    return t[valid], gpkg(w[valid])


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
    """Return 15 illustrative strategy regions; membership is not comfort."""
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


def prepare_hourly(frame, pressure_kpa):
    """Clean EPW sentinels and calculate all chart properties at one pressure."""
    t = pd.to_numeric(frame['drybulb'], errors='coerce')
    rh = pd.to_numeric(frame['relhum'], errors='coerce')
    valid = t.between(-100, 99.8) & rh.between(0, 100)
    t, rh = t[valid].to_numpy(float), rh[valid].to_numpy(float)
    w = w_from_Pv_kPa(rh / 100. * p_ws_kPa(t), pressure_kpa)
    out = pd.DataFrame({'Dry bulb (C)': t, 'RH (%)': rh, 'Humidity ratio (g/kg)': gpkg(w),
                        'Dew point (C)': dew_point_C(t, rh), 'Wet bulb (C)': wet_bulb_C(t, rh, pressure_kpa),
                        'Enthalpy (kJ/kg)': enthalpy_kJkg(t, w),
                        'Specific volume (m3/kg)': specific_vol(t, w, pressure_kpa)}, index=frame.index[valid])
    return out.loc[np.isfinite(out['Humidity ratio (g/kg)']) & np.isfinite(out['Wet bulb (C)'])]


def clip_region_to_saturation(vertices, pressure_kpa):
    """Clip a convex illustrative polygon to physical moist-air states.

    Sample vertical polygon sections so the upper boundary follows the curved
    saturation limit instead of joining only a few clipped corner points.
    """
    polygon = np.asarray(vertices, float)
    xs = np.unique(np.r_[np.linspace(polygon[:, 0].min(), polygon[:, 0].max(), 180), polygon[:, 0]])
    lower, upper, kept = [], [], []
    for x in xs:
        crossings = []
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            if abs(a[0]-b[0]) < 1e-10:
                if abs(x-a[0]) < 1e-8:
                    crossings.extend([a[1], b[1]])
            elif min(a[0], b[0])-1e-9 <= x <= max(a[0], b[0])+1e-9:
                crossings.append(a[1]+(b[1]-a[1])*(x-a[0])/(b[0]-a[0]))
        if not crossings:
            continue
        lo, hi = max(0., min(crossings)), min(max(crossings), float(gpkg(w_sat(x, pressure_kpa))))
        if np.isfinite(hi) and hi >= lo:
            kept.append(x); lower.append(lo); upper.append(hi)
    if len(kept) < 2:
        return []
    return list(zip(kept, lower)) + list(zip(kept[::-1], upper[::-1]))


def strategy_screening(points, pressure_kpa, reference_t=20.):
    from matplotlib.path import Path
    zones = {'Reference band': {'polygon': _comfort_zone(reference_t), 'color': '#318477'}}
    for info in get_all_strategy_zones(pressure_kpa, reference_t).values():
        zones[info['name']] = {'polygon': info['polygon'], 'color': info['color']}
    xy = points[['Dry bulb (C)', 'Humidity ratio (g/kg)']].to_numpy(float)
    masks = {}
    for name, zone in zones.items():
        vertices = clip_region_to_saturation(zone['polygon'], pressure_kpa)
        zone['polygon'] = vertices
        masks[name] = Path(vertices + [vertices[0]]).contains_points(xy, radius=1e-9) if vertices else np.zeros(len(points), bool)
    return zones, masks


def strategy_figure(points, pressure_kpa, zones, selected, *, fit=True,
                    show_rh=True, show_enthalpy=False, show_volume=False, show_wetbulb=False):
    """Chart-first presentation; every plotted property uses the same SI basis."""
    t, y = points['Dry bulb (C)'].to_numpy(float), points['Humidity ratio (g/kg)'].to_numpy(float)
    xlo = min(-10., float(t.min())-3) if fit else -10.
    xhi = max(40., float(t.max())+3) if fit else 40.
    ymax = max(28., float(y.max())+2) if fit else 28.
    axis = np.linspace(max(-100., xlo), min(99.8, xhi), 650)
    sat = gpkg(w_sat(axis, pressure_kpa))
    fig = go.Figure()

    def line(tx, wy, name, color, dash=None):
        tx, wy = np.asarray(tx), np.asarray(wy)
        valid = np.isfinite(wy) & (wy >= 0) & (wy <= ymax) & (wy <= gpkg(w_sat(tx, pressure_kpa))+1e-8)
        fig.add_trace(go.Scatter(x=np.where(valid, tx, np.nan), y=np.where(valid, wy, np.nan),
                                 mode='lines', name=name, line=dict(color=color, width=.8, dash=dash),
                                 showlegend=False, hovertemplate=name+'<extra></extra>', connectgaps=False))

    for name in selected:
        zone = zones[name]
        vertices = zone['polygon']
        if not vertices:
            continue
        xy = np.asarray(vertices+[vertices[0]])
        colour = zone['color']
        rgb = ','.join(str(int(colour[i:i+2], 16)) for i in (1, 3, 5))
        fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode='lines', fill='toself',
                                 line=dict(color=colour, width=1.5), fillcolor=f'rgba({rgb},0.10)',
                                 name=name, hovertemplate=name+' (illustrative)<extra></extra>'))
    line(axis, sat, 'Saturation (100% RH)', '#243c4b')
    if show_rh:
        for rh in (20, 40, 60, 80):
            line(axis, rh_curve(axis, rh, pressure_kpa), f'{rh}% RH', '#a3b8bd', 'dot')
    if show_enthalpy:
        for h in range(-20, 121, 10):
            line(axis, enthalpy_w_line(axis, h), f'h = {h} kJ/kg', '#c7a566', 'dash')
    if show_volume:
        # Select volumes appropriate to this station's pressure.
        volumes = specific_vol(np.array([xlo, xhi]), np.array([0., ymax/1000]), pressure_kpa)
        for v in np.linspace(volumes.min(), volumes.max(), 8):
            line(axis, volume_w_line(axis, v, pressure_kpa), f'v = {v:.2f} m3/kg', '#9fa9c4', 'dot')
    if show_wetbulb:
        for tw in range(-20, 36, 5):
            tx, wy = wetbulb_curve(axis, tw, pressure_kpa)
            line(tx, wy, f'Wet bulb = {tw} C', '#89adb8', 'dash')
    custom = points[['RH (%)', 'Wet bulb (C)', 'Dew point (C)', 'Enthalpy (kJ/kg)', 'Specific volume (m3/kg)']].to_numpy(float)
    fig.add_trace(go.Scatter(x=t, y=y, mode='markers', name='Hourly weather',
                             marker=dict(size=4, color='#235b75', opacity=.35), customdata=custom,
                             text=points.index.strftime('%b %d %H:%M') if isinstance(points.index, pd.DatetimeIndex) else points.index.astype(str),
                             hovertemplate='%{text}<br>Dry bulb: %{x:.1f} C<br>Humidity ratio: %{y:.2f} g/kg<br>RH: %{customdata[0]:.1f}%<br>Wet bulb: %{customdata[1]:.2f} C<br>Dew point: %{customdata[2]:.2f} C<br>Enthalpy: %{customdata[3]:.2f} kJ/kg<br>Specific volume: %{customdata[4]:.3f} m3/kg<extra></extra>'))
    fig.update_layout(title=dict(text='Climate Strategies · Psychrometric Chart', font=dict(size=18)),
                      height=650, margin=dict(l=60, r=25, t=55, b=170),
                      xaxis=dict(title='Dry-bulb temperature (°C)', range=[xlo, xhi], gridcolor='#edf1f3', zeroline=False, color='#304757'),
                      yaxis=dict(title='Humidity ratio (g/kg dry air)', range=[0, ymax], gridcolor='#edf1f3', zeroline=False, color='#304757'),
                      legend=dict(orientation='h', y=-.17, x=0, font=dict(size=11, color='#304757')),
                      paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#304757'),
                      hovermode='closest', meta=dict(preserve_plot_style=True))
    fig.add_annotation(text='Illustrative regions: overlapping hours are not predicted comfort or energy savings.',
                       x=0, y=-.27, xref='paper', yref='paper', xanchor='left', showarrow=False,
                       font=dict(size=10, color='#536675'))
    return fig
