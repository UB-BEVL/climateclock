from __future__ import annotations
# standard libs next
import os
import calendar
from pathlib import Path
import io, zipfile, csv, math, argparse, datetime, base64, hashlib
from contextlib import nullcontext
import requests
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple, Optional, Union
import re
# third-party
import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import pydeck as pdk
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for headless environments (Streamlit Cloud)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
try:
    import pvlib
    PVLIB_AVAILABLE = True
    PVLIB_IMPORT_ERROR = None
except Exception as _pvlib_import_exc:
    pvlib = None  # type: ignore[assignment]
    PVLIB_AVAILABLE = False
    PVLIB_IMPORT_ERROR = _pvlib_import_exc
# local modules
from metrics import comfort_energy as ce
import live_sensors as ls
import psychro_helpers as psh

# Patch platform processor to avoid Windows WMI KeyError during h5py/pvlib import
import platform as _platform

try:
    import streamlit.runtime.scriptrunner as _sr
except Exception:
    _sr = None
# utils/pdf_exporter.py
from fpdf import FPDF
import tempfile

try:
    _kaleido_scope = getattr(pio, "kaleido", None)
    _kaleido_scope = getattr(_kaleido_scope, "scope", None)
    if _kaleido_scope is not None:
        chromium_args = list(getattr(_kaleido_scope, "chromium_args", []) or [])
        for arg in ("--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"):
            if arg not in chromium_args:
                chromium_args.append(arg)
        _kaleido_scope.chromium_args = chromium_args
        if hasattr(_kaleido_scope, "mathjax"):
            _kaleido_scope.mathjax = None
except Exception:
    pass


_COMPASS_TO_DEG = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}

def _normalize_wind_dir(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip().str.upper()
    num = pd.to_numeric(raw, errors="coerce")
    txt = raw.map(_COMPASS_TO_DEG)
    out = num.fillna(txt)
    out = out.mask(out >= 999)
    return out % 360

def _session_is_ready() -> bool:
    if _sr is None:
        return True
    try:
        ctx = _sr.get_script_run_ctx()
        return ctx is not None
    except Exception:
        return True


if not hasattr(st, "fragment"):
    if hasattr(st, "experimental_fragment"):
        st.fragment = st.experimental_fragment
    else:
        def noop_fragment(func=None, *, run_every=None):
            if func is not None:
                return func
            return lambda f: f
        st.fragment = noop_fragment
elif hasattr(st, "experimental_fragment") and getattr(st.fragment, "__name__", "") == "_safe_fragment":
    # Recover from older hot-reloaded sessions where st.fragment was monkey-patched.
    st.fragment = st.experimental_fragment


from collections import defaultdict

def safe_format_caption(template, variables):
    """Format caption variables safely, rounding scalar floats before substitution."""
    clean_variables = {}
    for key, value in dict(variables or {}).items():
        if isinstance(value, np.floating):
            clean_variables[key] = round(float(value), 1)
        elif isinstance(value, float):
            clean_variables[key] = round(value, 1)
        else:
            clean_variables[key] = value
    try:
        return str(template).format_map(defaultdict(lambda: "[data unavailable]", clean_variables))
    except (KeyError, ValueError, TypeError):
        return str(template)

def placeholder_figure(message):
    """Returns a blank Plotly figure with a centered message for missing data."""
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=16, color="gray"))
    fig.update_layout(xaxis_visible=False, yaxis_visible=False,
                      plot_bgcolor="white", height=400)
    return fig

def resolve_longwave_column(df):
    """Resolves longwave radiation column across TMY3/CWEC/IWEC EPW formats."""
    candidates = ['horirsky', 'hor_ir_sky', 'horizontal_ir_sky', 'et_hr_sky_rad',
                  'horz_ir_intens', 'ir_hoz', 'horizontal_ir',
                  'longwave_radiation', 'lw_hoz', 'infrared_rad_hoz']
    return next((c for c in candidates if c in df.columns), None)

@st.cache_data
def compute_all_utci_scenarios(tdb, tr, v, rh):
    """Computes all 4 UTCI variants in ONE cached pass. Call once, use for Figs I, J, K."""
    tdb = np.asarray(tdb, dtype=float)
    tr = np.asarray(tr, dtype=float)
    v = np.asarray(v, dtype=float)
    rh = np.asarray(rh, dtype=float)

    def _utci_from_arrays(tdb_arr, tr_arr, v_arr, rh_arr):
        wind_arr = np.clip(np.asarray(v_arr, dtype=float), 0.1, None)
        rh_arr = np.clip(np.asarray(rh_arr, dtype=float), 0, 100)
        try:
            from pythermalcomfort.models import utci
            result = utci(tdb=tdb_arr, tr=tr_arr, v=wind_arr, rh=rh_arr)
            if hasattr(result, "utci"):
                return np.asarray(result.utci, dtype=float)
            return np.asarray(result, dtype=float)
        except Exception:
            vp = (rh_arr / 100.0) * 6.105 * np.exp((17.27 * tdb_arr) / (237.7 + tdb_arr))
            return (
                tdb_arr + 0.607562 + 0.022771 * tdb_arr + 0.000806 * (tdb_arr**2)
                + 0.002 * vp - 0.065 * wind_arr + 0.001 * tdb_arr * wind_arr
                - 0.015 * tdb_arr * vp / 100.0 - 0.00025 * vp * wind_arr
            )

    return {
        'baseline': _utci_from_arrays(tdb, tr, v, rh),
        'shaded':   _utci_from_arrays(tdb, tr - 15.0, v, rh),
        'calm':     _utci_from_arrays(tdb, tr, np.full_like(v, 0.5), rh),
        'neutral':  _utci_from_arrays(tdb, tr, v, np.full_like(rh, 50.0)),
    }


def _prepare_advanced_figure_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Add normalized aliases expected by the advanced publication figures."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    def _first_column(aliases: List[str], exclude_cols: Optional[List[str]] = None) -> Optional[str]:
        lower_to_col = {str(c).lower(): c for c in out.columns}
        for alias in aliases:
            if alias.lower() in lower_to_col:
                return lower_to_col[alias.lower()]
        try:
            return get_metric_column(out, aliases, exclude_cols=exclude_cols or [])
        except Exception:
            return None

    alias_map = {
        "drybulb_C": ["drybulb", "dry_bulb", "temp_air", "temperature", "tdb"],
        "dewpoint_C": ["dewpoint", "dew_point", "temp_dew", "dew_temperature", "dew"],
        "rh_pct": ["relhum", "relative_humidity", "rel_humidity", "humidity", "rh"],
        "ghi_Wm2": ["glohorrad", "ghi", "global_horizontal", "global_horiz", "solar", "radiation"],
        "dni_Wm2": ["dirnorrad", "dni", "direct_normal", "dir_nor_rad"],
        "dhi_Wm2": ["difhorrad", "dhi", "diffuse_horizontal", "dif_hor_rad"],
        "wind_speed_ms": ["windspd", "wind_speed", "windspeed", "wspd", "ws"],
    }
    for target, aliases in alias_map.items():
        if target not in out.columns:
            src = _first_column(aliases)
            if src:
                out[target] = pd.to_numeric(out[src], errors="coerce")
        else:
            out[target] = pd.to_numeric(out[target], errors="coerce")

    if "wind_dir_deg" not in out.columns:
        src = _first_column(["winddir", "wind_direction", "wind_dir", "wdir", "wd", "HourlyWindDirection"])
        if src:
            out["wind_dir_deg"] = _normalize_wind_dir(out[src])

    if not isinstance(out.index, pd.DatetimeIndex):
        if {"month", "day", "hour"}.issubset(out.columns):
            year_src = out["year"] if "year" in out.columns else pd.Series(2021, index=out.index)
            hour = pd.to_numeric(out["hour"], errors="coerce").fillna(0).astype(int).clip(0, 23)
            out.index = pd.to_datetime(
                dict(
                    year=pd.to_numeric(year_src, errors="coerce").fillna(2021).astype(int),
                    month=pd.to_numeric(out["month"], errors="coerce").fillna(1).astype(int).clip(1, 12),
                    day=pd.to_numeric(out["day"], errors="coerce").fillna(1).astype(int).clip(1, 31),
                    hour=hour,
                ),
                errors="coerce",
            )

    if isinstance(out.index, pd.DatetimeIndex):
        try:
            # Force all timestamps to a single reference leap year (2020) to align multi-year or TMY datasets into a single 12-month sequence
            new_index = pd.to_datetime(dict(
                year=2020,
                month=out.index.month,
                day=out.index.day,
                hour=out.index.hour,
                minute=out.index.minute
            ))
            out.index = new_index
            out = out.sort_index()
            # De-duplicate by taking the average for any overlapping hourly timestamps
            out = out.groupby(level=0).mean()
        except Exception:
            pass

    return out


def _advanced_day_hour_matrix(df: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        local = pd.DataFrame({"doy": df.index.dayofyear, "hour": df.index.hour, "value": values}, index=df.index)
    else:
        local = pd.DataFrame({
            "doy": pd.to_numeric(df.get("doy"), errors="coerce"),
            "hour": pd.to_numeric(df.get("hour"), errors="coerce"),
            "value": values,
        }, index=df.index)
    local = local.dropna()
    if local.empty:
        return pd.DataFrame()
    mat = local.pivot_table(index="hour", columns="doy", values="value", aggfunc="mean")
    return mat.reindex(index=range(24), columns=range(1, 367))


advance_day_hour_matrix = _advanced_day_hour_matrix


_orig_proc_get = getattr(getattr(_platform, "_Processor", None), "get", None)
if _orig_proc_get:
    def _safe_proc_get(self=None, *args, **kwargs):
        try:
            return _orig_proc_get(self, *args, **kwargs) if _orig_proc_get else ""
        except Exception:
            return os.environ.get("PROCESSOR_IDENTIFIER", "") or ""
    try:
        _platform._Processor.get = _safe_proc_get  # type: ignore[attr-defined]

    except Exception:
        pass  # silently ignore – this is just a platform compat patch


def _temp_unit() -> str:
    return "F" if st.session_state.get("temperature_unit") == "F" else "C"


def _c_to_f(value: float) -> float:
    return (float(value) * 9.0 / 5.0) + 32.0


def _f_to_c(value: float) -> float:
    return (float(value) - 32.0) * 5.0 / 9.0


def format_temperature(value, digits: int = 1) -> str:
    if value is None or (isinstance(value, (float, int)) and pd.isna(value)):
        return "—"
    display = float(value)
    if _temp_unit() == "F":
        display = _c_to_f(display)
        suffix = "°F"
    else:
        suffix = "°C"
    return f"{display:.{digits}f} {suffix}"


def format_temperature_delta(value, digits: int = 1, show_sign: bool = True) -> str:
    if value is None or (isinstance(value, (float, int)) and pd.isna(value)):
        return "—"
    delta = float(value)
    if _temp_unit() == "F":
        delta = delta * 9.0 / 5.0
        suffix = "°F"
    else:
        suffix = "°C"
    fmt = f"{{:{'+' if show_sign else ''}.{digits}f}} {{}}"
    return fmt.format(delta, suffix)


def format_threshold_label(temp_c: float, direction: str = ">", digits: int = 0) -> str:
    display = convert_threshold_for_display(temp_c)
    suffix = "°F" if _temp_unit() == "F" else "°C"
    return f"{direction} {display:.{digits}f} {suffix}"


def convert_threshold_for_display(temp_c: float) -> float:
    return _c_to_f(temp_c) if _temp_unit() == "F" else temp_c


# ========== UI COMPONENTS ==========
from contextlib import contextmanager
import time

@contextmanager
def cloud_loader(message: str = "Loading…"):
    placeholder = st.empty()
    placeholder.markdown(
        f"""
        <div style="
            position: fixed;
            right: 16px;    
            top: 16px;
            z-index: 9999;
            padding: 6px 10px;
            background: rgba(15,23,42,0.92);
            border-radius: 999px;
            display: flex;
            align-items: center;
            gap: 8px;
            box-shadow: 0 12px 30px rgba(0,0,0,0.6);
            font-size: 12px;
            color: #e5e7eb;
        ">
            <div class="cloud-loader">
                <span></span><span></span><span></span>
            </div>
            <span>{message}</span>
        </div>

        <style>
        .cloud-loader {{
            position: relative;
            width: 26px;
            height: 18px;
        }}
        .cloud-loader span {{
            position: absolute;
            display: block;
            background: #38bdf8;
            border-radius: 999px;
            opacity: 0.85;
            animation: cloud-pulse 1.4s infinite ease-in-out;
        }}
        .cloud-loader span:nth-child(1) {{
            width: 14px; height: 14px;
            left: 0; bottom: 0;
        }}
        .cloud-loader span:nth-child(2) {{
            width: 18px; height: 18px;
            left: 8px; bottom: 0;
            animation-delay: 0.12s;
        }}
        .cloud-loader span:nth-child(3) {{
            width: 12px; height: 12px;
            left: 18px; bottom: 2px;
            animation-delay: 0.24s;
        }}
        @keyframes cloud-pulse {{
            0%, 100% {{ transform: translateY(0); opacity: 0.8; }}
            50%      {{ transform: translateY(-2px); opacity: 1.0; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    try:
        yield
    finally:
        time.sleep(0.05)
        placeholder.empty()

# ========== IMPROVED LAYOUT ==========
st.set_page_config(
    page_title="Climate Analysis Pro",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ── Session state defaults ──────────────────────────────────────
for _k, _v in [
    ("cdf", None),
    ("header", {}),
    ("epw_path", None),
    ("uploaded_epw", None),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v
# Prevent footer "ghosting" during reruns/navigation: hide any old <footer> ASAP.
# The app footer we render later uses class "bevl-footer" and stays visible.
# Prevent footer "ghosting" during reruns/navigation: hide any old <footer> ASAP.
# The app footer we render later uses class "bevl-footer" and stays visible.
st.markdown(
    """
    <style>
        footer { display: none !important; }
        footer.bevl-footer { display: block !important; }
        /* Hide Streamlit's default running spinner icon */
        [data-testid="stStatusWidget"] {
            display: none !important;
        }
        
        /* Lock the sidebar open at a compact, narrow width. */
        section[data-testid="stSidebar"][aria-expanded="true"] > div {
            width: 248px !important;
            min-width: 248px !important;
            max-width: 248px !important;
        }
        section[data-testid="stSidebar"][aria-expanded="true"] {
            width: 248px !important;
            min-width: 248px !important;
            max-width: 248px !important;
        }
        section[data-testid="stSidebar"][aria-expanded="false"],
        section[data-testid="stSidebar"][aria-expanded="false"] > div {
            width: 248px !important;
            min-width: 248px !important;
            max-width: 248px !important;
            transform: translateX(0) !important;
            visibility: visible !important;
            margin-left: 0 !important;
        }
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapseButton"],
        button[title="Close sidebar"],
        button[title="Open sidebar"],
        button[aria-label="Close sidebar"],
        button[aria-label="Open sidebar"] {
            display: none !important;
        }

        /* Target Streamlit's specific heading anchor links without hiding normal links */
        a.header-anchor,
        .stMarkdown a.header-anchor,
        h1 > a:empty,
        h2 > a:empty,
        h3 > a:empty,
        h4 > a:empty,
        h5 > a:empty,
        h6 > a:empty {
            display: none !important;
        }

        /* Hide the specific SVG anchor icon next to headers */
        h1 svg, h2 svg, h3 svg, h4 svg, h5 svg, h6 svg {
            display: none !important;
        }

        /* ── Sidebar navigation: highlight active page ── */
        [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label {
            border-radius: 8px !important;
            padding: 0.45rem 0.75rem !important;
            margin: 2px 0 !important;
            transition: all 0.2s ease !important;
            border-left: 3px solid transparent !important;
            cursor: pointer !important;
        }
        /* Hover for non-selected items */
        [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
            background: rgba(59, 130, 246, 0.08) !important;
        }
        /* Active/selected item */
        [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:has(input[type="radio"]:checked) {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.18) 0%, rgba(99, 102, 241, 0.12) 100%) !important;
            border-left: 3px solid #3b82f6 !important;
            font-weight: 600 !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:has(input[type="radio"]:checked) p {
            color: #60a5fa !important;
            font-weight: 600 !important;
        }
        /* Hide the default radio dot entirely for a cleaner nav look */
        [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
            display: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.session_state.setdefault("temperature_unit", "C")
st.session_state.setdefault("epw_ready", False)
# Controller page mode (strict gating): "select_station" | "dashboard"
if st.session_state.get("active_page") not in ("select_station", "dashboard"):
    st.session_state["active_page"] = "select_station"
st.session_state.setdefault("pending_station_id", None)
st.session_state.setdefault("pending_station", None)
st.session_state.setdefault("load_requested", False)
st.session_state.setdefault("is_loading", False)
st.session_state.setdefault("last_loaded_station_id", None)
st.session_state.setdefault("_map_slot", None)
st.session_state.setdefault("_clear_map_on_next_run", False)
st.session_state.setdefault("pdf_capture_running", False)
st.session_state.setdefault("pdf_capture_pages", [])
st.session_state.setdefault("pdf_capture_index", 0)
st.session_state.setdefault("pdf_capture_origin_page", "Select weather file")
st.session_state.setdefault("pdf_download_bytes", None)
st.session_state.setdefault("pdf_download_name", None)
st.session_state.setdefault("pdf_download_error", None)
st.session_state.setdefault("pdf_dashboard_autobuild_pending", False)
st.session_state.setdefault("pdf_figures", {})
st.session_state.setdefault("pdf_figures_auto", {})
st.session_state.setdefault("pdf_captions", {})


# Navigation definitions
DEFAULT_PAGE = "Select weather file"

PRIMARY_NAV_GROUPS = [
    (
        "Start",
        [
            ("Map", "Select weather file"),
            ("Overview", "Overview"),
            ("Dashboard", "Dashboard"),
        ],
    ),
    (
        "Forecasts",
        [
            ("Short-Term Prediction", "Short-Term Prediction (24–72h)"),
            ("Long-Term Prediction", "Future Climate (2050 / 2080 SSP)"),
        ],
    ),
    (
        "Live Data",
        [
            ("EPW vs Live EPW", "Live Data vs EPW"),
            ("Sensor Comparison", "Sensor Comparison"),
        ],
    ),
    (
        "Outputs",
        [
            ("Export", "Export"),
        ],
    ),
]

NAV_ITEMS = [item for _, group_items in PRIMARY_NAV_GROUPS for item in group_items]
INTERNAL_PAGES: List[str] = []
REPORT_CAPTURE_PAGES = {
    "Dashboard",
    "Overview",
}

FROZEN_NAV_LABELS: set[str] = set()
FROZEN_PAGES: set[str] = set()

LABEL_TO_PAGE = {label: page for label, page in NAV_ITEMS}
PAGE_TO_LABEL = {page: label for label, page in NAV_ITEMS}
ALLOWED_PAGES = list(dict.fromkeys(list(PAGE_TO_LABEL.keys()) + INTERNAL_PAGES))

# UI navigation page (keeps existing sidebar labels/behavior)
_legacy_nav_page = st.session_state.get("nav_page")
if _legacy_nav_page not in ALLOWED_PAGES:
    _legacy_nav_page = st.session_state.get("active_page")
st.session_state.setdefault("nav_page", _legacy_nav_page if _legacy_nav_page in ALLOWED_PAGES else DEFAULT_PAGE)

# (nav_page initialization moved above to support legacy sessions)

THEME_BASE = "light"
try:
    theme_option = st.get_option("theme.base")
    if isinstance(theme_option, str):
        THEME_BASE = theme_option.lower()
except Exception:
    pass


CHART_FONT_FAMILY = "Inter, Poppins, Segoe UI, Helvetica, Arial, sans-serif"
CHART_COLORWAY = [
    "#3b82f6",  # blue
    "#f97316",  # orange
    "#06b6d4",  # cyan
    "#8b5cf6",  # violet
    "#22c55e",  # green
    "#e11d48",  # rose
]
CHART_DARK_BG = "#0f172a"
CHART_DARK_TEXT = "#e2e8f0"
CHART_DARK_GRID = "rgba(148, 163, 184, 0.24)"
CHART_LIGHT_BG = "#f8fafc"
CHART_LIGHT_TEXT = "#1e293b"
CHART_LIGHT_GRID = "#cbd5e1"
# Detect Streamlit Cloud (limited to ~1 GB RAM) to reduce Kaleido/Chromium memory use.
_IS_STREAMLIT_CLOUD = os.environ.get("STREAMLIT_SHARING_MODE") == "true" or os.environ.get("IS_STREAMLIT_CLOUD") == "true" or os.path.isdir("/mount/src")

# On cloud, export at 1× scale and smaller viewport to stay within memory budget.
REPORT_EXPORT_WIDTH = 900 if _IS_STREAMLIT_CLOUD else 1280
REPORT_EXPORT_HEIGHT = 500 if _IS_STREAMLIT_CLOUD else 720
REPORT_EXPORT_SCALE = 1 if _IS_STREAMLIT_CLOUD else 2


def _build_accessible_plotly_template(mode: str = "dark") -> go.layout.Template:
    """Ensure high-contrast Plotly defaults across the app."""
    dark_mode = mode == "dark"
    background = CHART_DARK_BG if dark_mode else CHART_LIGHT_BG
    font_color = CHART_DARK_TEXT if dark_mode else CHART_LIGHT_TEXT
    grid_color = CHART_DARK_GRID if dark_mode else CHART_LIGHT_GRID

    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor=background,
        plot_bgcolor=background,
        font=dict(color=font_color, family=CHART_FONT_FAMILY, size=12),
        title=dict(font=dict(color=font_color, family=CHART_FONT_FAMILY, size=18), x=0.01, xanchor="left"),
        xaxis=dict(
            gridcolor=grid_color,
            zerolinecolor=grid_color,
            linecolor=grid_color,
            tickfont=dict(color=font_color),
            title_font=dict(color=font_color),
            automargin=True,
            mirror=True,
        ),
        yaxis=dict(
            gridcolor=grid_color,
            zerolinecolor=grid_color,
            linecolor=grid_color,
            tickfont=dict(color=font_color),
            title_font=dict(color=font_color),
            automargin=True,
            mirror=True,
        ),
        legend=dict(font=dict(color=font_color, family=CHART_FONT_FAMILY, size=11), bgcolor=background, bordercolor=grid_color, borderwidth=0),
        hoverlabel=dict(bgcolor=background, font=dict(color=font_color, family=CHART_FONT_FAMILY)),
        scene=dict(
            bgcolor=background,
            xaxis=dict(showbackground=False, backgroundcolor=background, gridcolor=grid_color, zerolinecolor=grid_color, tickfont=dict(color=font_color), title_font=dict(color=font_color)),
            yaxis=dict(showbackground=False, backgroundcolor=background, gridcolor=grid_color, zerolinecolor=grid_color, tickfont=dict(color=font_color), title_font=dict(color=font_color)),
            zaxis=dict(showbackground=False, backgroundcolor=background, gridcolor=grid_color, zerolinecolor=grid_color, tickfont=dict(color=font_color), title_font=dict(color=font_color)),
        ),
        margin=dict(l=48, r=32, t=60, b=48),
    )
    template.layout.colorway = CHART_COLORWAY
    return template


pio.templates["bevl_dark"] = _build_accessible_plotly_template("dark")
pio.templates["bevl_light"] = _build_accessible_plotly_template("light")

DEFAULT_TEMPLATE = "bevl_dark"
try:
    # The app UI is custom dark-themed, so keep Plotly aligned with that surface
    # instead of inheriting Streamlit's default light theme.
    chosen = DEFAULT_TEMPLATE
    px.defaults.template = chosen
    pio.templates.default = chosen
    PLOTLY_TEMPLATE = chosen
except Exception:
    px.defaults.template = DEFAULT_TEMPLATE
    pio.templates.default = DEFAULT_TEMPLATE
    PLOTLY_TEMPLATE = DEFAULT_TEMPLATE


def _dashboard_template_name() -> str:
    return DEFAULT_TEMPLATE


def _clone_dashboard_figure(fig_obj: object) -> object:
    """Return a styled Plotly Figure snapshot using the same theme as the dashboard."""
    try:
        fig = fig_obj if isinstance(fig_obj, go.Figure) else go.Figure(fig_obj)
        fig = go.Figure(fig)
        return _apply_global_plot_style(fig)
    except Exception:
        return fig_obj


def _apply_global_plot_style(fig_obj: object) -> object:
    """Apply the app-wide Plotly template and ensure non-transparent backgrounds and readable fonts.

    This overrides per-figure transparent backgrounds so charts render consistently for
    both on-screen display and PDF export.
    """
    try:
        import plotly.graph_objects as go
        # Normalize to a Figure
        if not isinstance(fig_obj, go.Figure):
            fig_obj = go.Figure(fig_obj)

        template_name = _dashboard_template_name()
        # Apply template (this also sets font colors, bg colors, axes defaults)
        try:
            fig_obj.update_layout(template=template_name)
        except Exception:
            pass

        # Ensure backgrounds are not transparent (some figures explicitly set rgba(0,0,0,0)).
        try:
            t = pio.templates[template_name]
        except Exception:
            t = None
        bg = CHART_DARK_BG
        plotbg = CHART_DARK_BG
        fontcol = CHART_DARK_TEXT
        gridcol = CHART_DARK_GRID
        if t and getattr(t, "layout", None):
            bg = getattr(t.layout, "paper_bgcolor", None) or bg
            plotbg = getattr(t.layout, "plot_bgcolor", None) or plotbg
            fontcol = getattr(getattr(t.layout, "font", None), "color", None) or fontcol
            gridcol = getattr(getattr(t.layout, "xaxis", None), "gridcolor", None) or gridcol
            if bg is not None:
                fig_obj.layout.paper_bgcolor = bg
            if plotbg is not None:
                fig_obj.layout.plot_bgcolor = plotbg
            if fontcol is not None:
                try:
                    fig_obj.update_layout(
                        font=dict(color=fontcol, family=CHART_FONT_FAMILY),
                        hoverlabel=dict(font=dict(color=fontcol, family=CHART_FONT_FAMILY)),
                        legend=dict(
                            bgcolor=bg,
                            bordercolor=gridcol,
                            borderwidth=0,
                            font=dict(color=fontcol, family=CHART_FONT_FAMILY),
                        ),
                    )
                except Exception:
                    pass

        # Ensure axes tick/title colors follow font color when possible
        try:
            fc = fig_obj.layout.font.color if getattr(fig_obj.layout, "font", None) else None
            if fc:
                fig_obj.update_xaxes(
                    tickfont=dict(color=fc, family=CHART_FONT_FAMILY),
                    title_font=dict(color=fc, family=CHART_FONT_FAMILY),
                    linecolor=gridcol,
                    gridcolor=gridcol,
                    zerolinecolor=gridcol,
                )
                fig_obj.update_yaxes(
                    tickfont=dict(color=fc, family=CHART_FONT_FAMILY),
                    title_font=dict(color=fc, family=CHART_FONT_FAMILY),
                    linecolor=gridcol,
                    gridcolor=gridcol,
                    zerolinecolor=gridcol,
                )
                fig_obj.update_layout(
                    legend=dict(bgcolor=bg, bordercolor=gridcol, borderwidth=0, font=dict(color=fc, family=CHART_FONT_FAMILY)),
                    title=dict(font=dict(color=fc, family=CHART_FONT_FAMILY)),
                )
        except Exception:
            pass

        def _clean_plotly_text(value: object) -> str:
            if value is None:
                return ""
            text = str(value)
            cleaned = re.sub(r"\b(undefined|none|nan)\b", "", text, flags=re.IGNORECASE)
            cleaned = re.sub(r"(<br\s*/?>|\s|[:;,\-|/])+", " ", cleaned, flags=re.IGNORECASE).strip()
            return cleaned

        def _text_is_missing(value: object) -> bool:
            return _clean_plotly_text(value) == ""

        # Guard against Plotly rendering blank/missing title objects as "undefined"
        # in colorbars, chart titles, axes, trace names, or facet annotations.
        try:
            title_text = getattr(getattr(fig_obj.layout, "title", None), "text", None)
            if title_text is None or _text_is_missing(title_text):
                fig_obj.update_layout(title_text="")
        except Exception:
            pass

        try:
            layout_keys = list(fig_obj.layout.to_plotly_json().keys())
            for axis_name in [k for k in layout_keys if k.startswith(("xaxis", "yaxis"))]:
                axis_obj = getattr(fig_obj.layout, axis_name, None)
                title_obj = getattr(axis_obj, "title", None) if axis_obj is not None else None
                title_text = getattr(title_obj, "text", None) if title_obj is not None else None
                if title_text is None or _text_is_missing(title_text):
                    try:
                        axis_obj.title.text = ""
                    except Exception:
                        axis_obj.title = dict(text="")
        except Exception:
            pass

        try:
            layout_height = getattr(fig_obj.layout, "height", None)
            if layout_height is None or float(layout_height) < 420:
                fig_obj.update_layout(height=420, autosize=False)
        except Exception:
            try:
                fig_obj.update_layout(height=420, autosize=False)
            except Exception:
                pass

        # Plotly annotations, colorbars, selectors, and 3D scenes do not always
        # inherit template fonts/backgrounds, so normalize them explicitly.
        try:
            annotations = []
            for ann in fig_obj.layout.annotations or []:
                ann_dict = ann.to_plotly_json()
                ann_text = ann_dict.get("text")
                if ann_text is None or _text_is_missing(ann_text):
                    ann_dict["text"] = ""
                ann_font = dict(ann_dict.get("font") or {})
                ann_font.update({"color": fontcol, "family": CHART_FONT_FAMILY})
                ann_dict["font"] = ann_font
                annotations.append(ann_dict)
            if annotations:
                fig_obj.update_layout(annotations=annotations)
        except Exception:
            pass

        def _style_colorbar(colorbar_obj: object) -> None:
            if colorbar_obj is None:
                return
            try:
                colorbar_obj.tickfont = dict(color=fontcol, family=CHART_FONT_FAMILY)
            except Exception:
                pass
            try:
                title_text = getattr(getattr(colorbar_obj, "title", None), "text", None)
                if title_text is None or _text_is_missing(title_text):
                    title_text = ""
                colorbar_obj.title = dict(text=title_text, font=dict(color=fontcol, family=CHART_FONT_FAMILY))
            except Exception:
                pass

        try:
            layout_json = fig_obj.layout.to_plotly_json()
        except Exception:
            layout_json = {}

        try:
            # Style coloraxis-based keys only when the figure already uses one.
            # Adding a layout coloraxis everywhere can create stray keys on maps.
            coloraxis_names = [
                name for name in layout_json.keys()
                if name == "coloraxis" or name.startswith("coloraxis")
            ]
            for coloraxis_name in coloraxis_names:
                fig_obj.update_layout(
                    **{
                        f"{coloraxis_name}_colorbar": dict(
                            tickfont=dict(color=fontcol, family=CHART_FONT_FAMILY),
                            title=dict(text="", font=dict(color=fontcol, family=CHART_FONT_FAMILY)),
                        )
                    }
                )
        except Exception:
            pass

        try:
            colorbar_trace_types = {"heatmap", "histogram2d", "histogram2dcontour", "contour", "surface", "mesh3d"}
            for trace in fig_obj.data or []:
                trace_json = trace.to_plotly_json()
                trace_type = str(trace_json.get("type", "")).lower()
                try:
                    trace_name = getattr(trace, "name", None)
                    if _text_is_missing(trace_name):
                        trace.name = ""
                        trace.showlegend = False
                    legendgroup_title = getattr(trace, "legendgrouptitle", None)
                    legendgroup_title_text = getattr(legendgroup_title, "text", None) if legendgroup_title is not None else None
                    if legendgroup_title is not None and _text_is_missing(legendgroup_title_text):
                        trace.legendgrouptitle = dict(text="")
                except Exception:
                    pass

                trace_uses_colorbar = (
                    trace_json.get("showscale") is True
                    or "colorbar" in trace_json
                    or (trace_type in colorbar_trace_types and trace_json.get("showscale") is not False)
                )
                if trace_uses_colorbar:
                    _style_colorbar(getattr(trace, "colorbar", None))
                else:
                    try:
                        trace.colorbar = None
                    except Exception:
                        pass

                marker = getattr(trace, "marker", None)
                if marker is None:
                    continue
                marker_json = marker.to_plotly_json()
                marker_uses_colorbar = marker_json.get("showscale") is True
                if marker_uses_colorbar:
                    _style_colorbar(getattr(marker, "colorbar", None))
                else:
                    try:
                        marker.colorbar = None
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            fig_obj.update_scenes(
                bgcolor=plotbg,
                xaxis=dict(showbackground=False, backgroundcolor=plotbg, gridcolor=gridcol, zerolinecolor=gridcol, tickfont=dict(color=fontcol), title_font=dict(color=fontcol)),
                yaxis=dict(showbackground=False, backgroundcolor=plotbg, gridcolor=gridcol, zerolinecolor=gridcol, tickfont=dict(color=fontcol), title_font=dict(color=fontcol)),
                zaxis=dict(showbackground=False, backgroundcolor=plotbg, gridcolor=gridcol, zerolinecolor=gridcol, tickfont=dict(color=fontcol), title_font=dict(color=fontcol)),
            )
        except Exception:
            pass

        try:
            for axis_name in [name for name in fig_obj.layout.to_plotly_json().keys() if name.startswith("xaxis")]:
                axis_obj = getattr(fig_obj.layout, axis_name, None)
                selector = getattr(axis_obj, "rangeselector", None) if axis_obj is not None else None
                if selector is not None and getattr(selector, "buttons", None):
                    selector.bgcolor = bg
                    selector.activecolor = gridcol
                    selector.font = dict(color=fontcol, family=CHART_FONT_FAMILY)
                slider = getattr(axis_obj, "rangeslider", None) if axis_obj is not None else None
                if slider is not None and getattr(slider, "visible", None):
                    slider.bgcolor = bg
                    slider.bordercolor = gridcol
        except Exception:
            pass

        return fig_obj
    except Exception:
        return fig_obj


def _st_plotly_chart(fig_obj: object, *args, **kwargs):
    """Wrapper around `st.plotly_chart` that applies the global plot style first.

    Use this everywhere instead of `st.plotly_chart` to ensure consistent
    background and font colors across the app and exported images.
    """
    try:
        fig_obj = _apply_global_plot_style(fig_obj)
    except Exception:
        pass
    config = kwargs.get("config")
    if config is None:
        kwargs["config"] = {"responsive": False}
    elif isinstance(config, dict):
        merged_config = dict(config)
        merged_config.setdefault("responsive", False)
        kwargs["config"] = merged_config
    return st.plotly_chart(fig_obj, *args, **kwargs)

# Default station source fallback (can be overridden)
STATION_SOURCE = "https://raw.githubusercontent.com/CenterForTheBuiltEnvironment/clima/main/assets/data/epw_location.json"
STATION_INDEX_TIMEOUT = (4, 10)
EPW_FETCH_TIMEOUT = (5, 20)
MAX_EPW_BYTES = 100_000_000


def _fallback_station_index() -> pd.DataFrame:
    """Small built-in station catalog used when the remote index is unavailable."""
    return pd.DataFrame([
        {
            "name": "USA_Buffalo.725280_TMY3",
            "country": "USA",
            "lat": 42.94,
            "lon": -78.73,
            "elevation_m": 215,
            "timezone": "UTC-5",
            "zip_url": "https://energyplus-weather.s3.amazonaws.com/north_america_wmo_region_4/USA/NY/Buffalo/Buffalo_Greater_International_AP_725280_TMY3.epw",
            "period": "TMY3",
            "heating_db": -17.5,
            "cooling_db": 29.5,
        },
        {
            "name": "USA_Phoenix.722780_TMY3",
            "country": "USA",
            "lat": 33.43,
            "lon": -112.02,
            "elevation_m": 337,
            "timezone": "UTC-7",
            "zip_url": "https://energyplus-weather.s3.amazonaws.com/north_america_wmo_region_4/USA/AZ/Phoenix/Phoenix_Sky_Harbor_Intl_Airport_722780_TMY3.epw",
            "period": "TMY3",
            "heating_db": 1.7,
            "cooling_db": 42.2,
        },
        {
            "name": "USA_Chicago.725300_TMY3",
            "country": "USA",
            "lat": 41.98,
            "lon": -87.90,
            "elevation_m": 201,
            "timezone": "UTC-6",
            "zip_url": "https://energyplus-weather.s3.amazonaws.com/north_america_wmo_region_4/USA/IL/Chicago/Chicago_OHare_Intl_Airport_725300_TMY3.epw",
            "period": "TMY3",
            "heating_db": -16.6,
            "cooling_db": 31.3,
        },
        {
            "name": "USA_Miami.722020_TMY3",
            "country": "USA",
            "lat": 25.82,
            "lon": -80.28,
            "elevation_m": 8,
            "timezone": "UTC-5",
            "zip_url": "https://energyplus-weather.s3.amazonaws.com/north_america_wmo_region_4/USA/FL/Miami/Miami_Intl_Airport_722020_TMY3.epw",
            "period": "TMY3",
            "heating_db": 8.0,
            "cooling_db": 33.0,
        },
    ])

# Streamlit cache decorator (data) compatible alias
try:
    CACHE = st.cache_data
except Exception:
    CACHE = lambda *args, **kwargs: (lambda fn: fn)

try:
    CACHE_RESOURCE = st.cache_resource
except Exception:
    CACHE_RESOURCE = CACHE


def _downcast_float32(df: Optional[pd.DataFrame], cols: Optional[Iterable[str]] = None) -> Optional[pd.DataFrame]:
    """Downcast selected float columns to float32 to reduce memory footprint.
    Does NOT modify the input dataframe.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()
    if cols is None:
        cols = out.select_dtypes(include=[np.floating]).columns.tolist()
    for c in cols:
        if c in out.columns:
            try:
                out[c] = out[c].astype(np.float32)
            except Exception:
                pass  # silently skip – column downcast is optional
    return out

def render_virtualized_table(
    df: pd.DataFrame,
    *,
    height: int = 360,
    key: str = "table",
    page_size: int = 50,
) -> None:
    """Render a large dataframe using Streamlit's native dataframe widget.

    Previously used AgGrid, but st_aggrid triggers a 'SessionInfo before
    it was initialized' error on Streamlit >= 1.30.  The native widget
    in Streamlit 1.40+ supports sorting and column resizing out of the box.
    """
    st.dataframe(df, use_container_width=True, height=height)

# --- Streamlit compat shims ---
if hasattr(st, "rerun"):
    def _rerun():
        if not _session_is_ready():
            return
        try:
            st.rerun()
        except Exception:
            return
else:
    def _rerun():
        if not _session_is_ready():
            return
        try:
            st.experimental_rerun()
        except Exception:
            return


def _capture_plotly_figure(fig_obj: Any) -> Any:
    """Capture Plotly figures rendered in-app so PDF export can include all charts."""
    if fig_obj is None:
        return fig_obj

    try:
        fig = fig_obj if isinstance(fig_obj, go.Figure) else go.Figure(fig_obj)
        fig = _clone_dashboard_figure(fig)
    except Exception:
        return fig_obj

    current_page = str(st.session_state.get("nav_page", "") or "").strip()
    if current_page not in REPORT_CAPTURE_PAGES:
        return fig_obj

    def _is_map_figure(fig_obj_local: go.Figure) -> bool:
        map_trace_types = {
            "scattermapbox", "choroplethmapbox", "densitymapbox", "scattergeo", "choropleth", "densitygeo"
        }
        for tr in getattr(fig_obj_local, "data", []) or []:
            if str(getattr(tr, "type", "")).lower() in map_trace_types:
                return True
        return False

    def _infer_title(fig_obj_local: go.Figure, store_local: Dict[str, Any]) -> str:
        raw_title = ""
        try:
            raw_title = (fig_obj_local.layout.title.text or "").strip() if fig_obj_local.layout and fig_obj_local.layout.title else ""
        except Exception:
            raw_title = ""
        if raw_title:
            return raw_title

        trace_types = [str(getattr(tr, "type", "")).lower() for tr in (getattr(fig_obj_local, "data", []) or [])]
        first_type = trace_types[0] if trace_types else ""
        if "heatmap" in first_type:
            base = "Heatmap"
        elif "histogram" in first_type:
            base = "Histogram"
        elif "bar" in first_type:
            base = "Bar chart"
        elif "scatter" in first_type:
            base = "Scatter plot"
        elif "surface" in first_type or "mesh3d" in first_type:
            base = "3D chart"
        elif first_type:
            base = f"{first_type.title()} chart"
        else:
            base = "Chart"

        used = {str(k).strip().lower() for k in store_local.keys()}
        idx = 1
        candidate = f"{base} {idx}"
        while candidate.lower() in used:
            idx += 1
            candidate = f"{base} {idx}"
        return candidate

    store = st.session_state.get("pdf_figures_auto", {})
    fingerprints = st.session_state.get("pdf_figure_fingerprints", set())

    try:
        fig_json = fig.to_json()
    except Exception:
        fig_json = ""

    if fig_json and fig_json in fingerprints:
        return fig

    title = _infer_title(fig, store)
    title_low = title.lower()
    exclude_keywords = ["station", "location picker", "map picker", "select weather", "setup"]
    if any(keyword in title_low for keyword in exclude_keywords) or _is_map_figure(fig):
        return fig_obj

    key = title
    if key in store:
        suffix = 2
        while f"{title} ({suffix})" in store:
            suffix += 1
        key = f"{title} ({suffix})"

    store[key] = fig
    if fig_json:
        fingerprints.add(fig_json)
        
    st.session_state["pdf_figures_auto"] = store
    st.session_state["pdf_figure_fingerprints"] = fingerprints
    return fig


def _install_plotly_capture_hook() -> None:
    """Monkey-patch st.plotly_chart once so all Plotly charts are captured for PDF export."""
    if not hasattr(st, "_original_plotly_chart"):
        setattr(st, "_original_plotly_chart", st.plotly_chart)

    if getattr(st.plotly_chart, "__name__", "") == "_plotly_chart_with_capture":
        return

    original_plotly_chart = getattr(st, "_original_plotly_chart")

    def _plotly_chart_with_capture(figure_or_data=None, *args, **kwargs):
        # on_select charts are managed by Streamlit's event protocol; avoid wrapping them.
        if kwargs.get("on_select"):
            return original_plotly_chart(figure_or_data, *args, **kwargs)
        fig_for_render = _capture_plotly_figure(figure_or_data)
        return original_plotly_chart(fig_for_render, *args, **kwargs)

    st.plotly_chart = _plotly_chart_with_capture


def _restore_plotly_chart_if_hooked() -> None:
    """Restore native Streamlit plotly renderer if a prior run monkey-patched it."""
    original_plotly_chart = getattr(st, "_original_plotly_chart", None)
    if original_plotly_chart is None:
        return
    if getattr(st.plotly_chart, "__name__", "") == "_plotly_chart_with_capture":
        st.plotly_chart = original_plotly_chart


# Guard against stale hot-reload state where plotly_chart remained monkey-patched.
_restore_plotly_chart_if_hooked()


def fix_station_url(url: str) -> List[str]:
    """
    Generate alternative URLs for common naming pattern issues.
    Returns a list of possible URLs to try.
    """
    if not isinstance(url, str):
        return []

    alternatives = [url]  # Always try the original first

    # Add .zip/.epw variant for all URLs
    if url.lower().endswith('.epw'):
        alternatives.append(url[:-4] + '.zip')
    elif url.lower().endswith('.zip'):
        alternatives.append(url[:-4] + '.epw')

    # Common pattern fixes for OneBuilding.org
    if "onebuilding.org" in url:
        parts = url.split("/")
        station_part = parts[-1] if parts else ""
        if "." in station_part and "_TMY" in station_part:
            base, ext = station_part.rsplit('.', 1)
            fixed_station = base.replace('.', '_') + '.' + ext
            fixed_url = "/".join(parts[:-1] + [fixed_station])
            alternatives.append(fixed_url)

        alternatives.append(url.lower())

    return list(dict.fromkeys(alternatives))


# Alternative EPW sources as fallbacks
ALTERNATIVE_EPW_SOURCES = [
    # Keep a minimal, general-purpose fallback list for manual selections
    "https://energyplus-weather.s3.amazonaws.com/north_america_wmo_region_4/USA/NY/Buffalo/Buffalo_Greater_International_AP_725280_TMY3.epw",
    "https://energyplus-weather.s3.amazonaws.com/north_america_wmo_region_4/USA/AZ/Phoenix/Phoenix_Sky_Harbor_Intl_Airport_722780_TMY3.epw",
    "https://energyplus-weather.s3.amazonaws.com/north_america_wmo_region_4/USA/IL/Chicago/Chicago_OHare_Intl_Airport_725300_TMY3.epw",
    "https://energyplus-weather.s3.amazonaws.com/north_america_wmo_region_4/USA/FL/Miami/Miami_Intl_Airport_722020_TMY3.epw",
]


@CACHE(show_spinner=False)
def fetch_epw_bytes_no_ui(url: str) -> Tuple[Optional[bytes], Optional[str]]:
    """Fetch EPW bytes (or extract from ZIP) with no Streamlit UI calls."""
    import requests
    import io
    import zipfile
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; ClimateAnalysisPro/1.0)",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate",
        }

        with requests.get(url, headers=headers, timeout=EPW_FETCH_TIMEOUT, stream=True) as r:
            r.raise_for_status()

            content_length = r.headers.get("content-length")
            if content_length and int(content_length) > MAX_EPW_BYTES:
                return None, f"File too large: {content_length} bytes"

            chunks = []
            total = 0
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_EPW_BYTES:
                    return None, f"File too large: exceeded {MAX_EPW_BYTES} bytes"
                chunks.append(chunk)

        content = b"".join(chunks)
        if url.lower().endswith(".zip") or zipfile.is_zipfile(io.BytesIO(content)):
            with zipfile.ZipFile(io.BytesIO(content), "r") as z:
                epws = [m for m in z.namelist() if m.lower().endswith(".epw")]
                if not epws:
                    return None, "ZIP file contains no EPW files"
                epws.sort(key=lambda m: z.getinfo(m).file_size, reverse=True)
                with z.open(epws[0]) as f:
                    return f.read(), None
        return content, None
    except Exception as e:
        return None, str(e)


@CACHE(show_spinner=False, ttl=86400)
def load_station_index(source: Optional[Union[str, Path]] = None):
    import json, requests
    src = source or STATION_SOURCE

    def _finish(df: pd.DataFrame) -> pd.DataFrame:
        keep = [
            "name", "country", "lat", "lon",
            "elevation_m", "timezone", "zip_url",
            "period", "heating_db", "cooling_db"
        ]
        for c in keep:
            if c not in df.columns:
                df[c] = np.nan

        # Memory downcasting: keep geo/weather-like numeric columns as float32.
        for c in ["lat", "lon", "elevation_m", "heating_db", "cooling_db"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

        df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

        if "zip_url" not in df.columns:
            df["zip_url"] = np.nan

        def _clean_zip_url_val(v):
            if isinstance(v, str) and ('<a ' in v and 'href' in v):
                return _extract_url(v) or v
            return v

        df["zip_url"] = df["zip_url"].apply(_clean_zip_url_val)

        na_zip = df["zip_url"].isna()
        if na_zip.any():
            def _find_any_url(row):
                for c in row.index:
                    v = row[c]
                    if not isinstance(v, str):
                        continue
                    m = re.search(r'(https?://[^\s">]+)', v)
                    if not m:
                        continue
                    url = m.group(1)
                    low = url.lower()
                    if low.endswith(".zip") or low.endswith(".epw"):
                        return url
                return None
            df.loc[na_zip, "zip_url"] = df[na_zip].apply(_find_any_url, axis=1)

        return df

    # ---- remote JSON (handles FeatureCollection from CBE Clima)
    if isinstance(src, str) and src.startswith(("http://", "https://")) and src.endswith(".json"):
        try:
            r = requests.get(src, timeout=STATION_INDEX_TIMEOUT)
            r.raise_for_status()
            obj = r.json()
        except Exception:
            return _finish(_fallback_station_index())

        if isinstance(obj, dict) and "features" in obj:
            raw = pd.json_normalize(obj["features"])

            def _lat(x):
                return x[1] if isinstance(x, (list, tuple)) and len(x) >= 2 else np.nan

            def _lon(x):
                return x[0] if isinstance(x, (list, tuple)) and len(x) >= 2 else np.nan

            raw["lat"] = raw.get("geometry.coordinates", np.nan).apply(_lat)
            raw["lon"] = raw.get("geometry.coordinates", np.nan).apply(_lon)

            df = raw.rename(columns={
                "properties.name": "name",
                "properties.title": "name",
                "properties.country": "country",
                "properties.elevation": "elevation_m",
                "properties.timezone": "timezone",
                "properties.station_name": "name",
            })

            prop_cols = [c for c in df.columns if c.startswith("properties.")]
            candidates = prop_cols + [
                c for c in df.columns
                if any(k in c.lower() for k in ["url", "epw", "tmyx", "tmy3", "href"])
            ]

            def _find_url(row):
                for c in candidates:
                    v = row.get(c) if hasattr(row, "get") else (row[c] if c in row else None)
                    if isinstance(v, (list, tuple)):
                        v = " ".join(map(str, v))
                    elif isinstance(v, dict):
                        v = " ".join(map(str, v.values()))
                    if not isinstance(v, str):
                        continue
                    m = _href.search(v)
                    if not m:
                        m = re.search(r'href=([^\s>]+)', v, re.I)
                    s = m.group(1).strip() if m else v.strip()
                    m_url = re.search(r'(https?://[^\s"<>]+)', s)
                    s = m_url.group(1) if m_url else s
                    low = s.lower()
                    if (".epw" in low) or low.endswith(".zip"):
                        return s
                return None

            df["zip_url"] = df.apply(_find_url, axis=1)

            # Fallback country/name cleanup
            if "name" in df.columns and "properties.title" in raw.columns:
                df["name"] = df["name"].fillna(raw["properties.title"])
            if "country" in df.columns:
                df["country"] = df["country"].fillna(df.apply(best_country_from_row, axis=1))

            def clean_url(val):
                if isinstance(val, str) and ('<a ' in val and 'href' in val):
                    return _extract_url(val) or val
                return val

            df["zip_url"] = df["zip_url"].apply(clean_url)
            df = _finish(df)

            if df.empty:
                df = _finish(_fallback_station_index())
            return df

        df = pd.json_normalize(obj)
        df = df.rename(columns={
            "latitude": "lat",
            "longitude": "lon",
            "elevation": "elevation_m",
            "tz": "timezone",
            "time_zone": "timezone",
            "station_name": "name"
        })
        df = _finish(df)

        if df.empty:
            df = _finish(_fallback_station_index())
        return df

    if isinstance(src, (str, Path)) and str(src).endswith(".json") and Path(src).exists():
        with open(src, "r", encoding="utf-8") as f:
            recs = json.load(f)
        df = pd.json_normalize(recs)
        df = df.rename(columns={
            "latitude": "lat",
            "longitude": "lon",
            "elevation": "elevation_m",
            "tz": "timezone",
            "time_zone": "timezone",
            "station_name": "name"
        })
        return _finish(df)

    if isinstance(src, (str, Path)) and Path(src).exists() and str(src).endswith(".csv"):
        df = pd.read_csv(src)
        if "zip_url" not in df.columns:
            for c in ["url", "epw_url", "TMYx_url", "TMY3_url"]:
                if c in df.columns:
                    df["zip_url"] = df[c]
                    break
        df = df.rename(columns={"latitude": "lat", "longitude": "lon", "elevation": "elevation_m"})
        return _finish(df)

    return _finish(_fallback_station_index())


@CACHE(show_spinner=False, ttl=86400)
def load_station_index_for_map(source: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load and minimize station index for map display (float32 lat/lon, only needed columns)."""
    df = load_station_index(source)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["name", "country", "lat", "lon", "zip_url", "period", "elevation_m", "timezone", "heating_db", "cooling_db"])

    needed = [
        "name",
        "country",
        "lat",
        "lon",
        "zip_url",
        "period",
        "elevation_m",
        "timezone",
        "heating_db",
        "cooling_db",
        # Optional passthroughs if present (used by label parsing, but safe if missing)
        "raw_id",
        "source",
        "station_id",
        "country_iso3",
        "state_code",
        "city_raw",
        "country_name",
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    out = df[needed].copy()
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce").astype(np.float32)
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce").astype(np.float32)
    return out.dropna(subset=["lat", "lon"]).reset_index(drop=True)


# tiny helper to strip URL from <a href=...> anchors in the GeoJSON properties
import re
_href = re.compile(r'href\s*=\s*["\']([^"\']+)["\']', re.I)


def _extract_url(html_anchor: str) -> Optional[str]:
    if not html_anchor:
        return None
    m = _href.search(html_anchor)
    if not m:
        return None
    url = m.group(1).strip().strip('"').strip("'")
    return url

_COUNTRY_FROM_URL = re.compile(r"/([A-Z]{3})_([A-Za-z_]+?)/")  # e.g. /USA_United_States_of_America/

def country_from_zip_url(url: str) -> Optional[str]:
    if not isinstance(url, str) or not url:
        return None
    m = _COUNTRY_FROM_URL.search(url)
    if m:
        return m.group(2).replace("_", " ")
    # broader fallback – scan any "AAA_Country_Name" segment
    try:
        for seg in url.split("/"):
            if "_" in seg and "." not in seg and len(seg) > 4 and seg[:3].isupper() and seg[3] == "_":
                return seg[4:].replace("_", " ")

    except Exception:
        pass  # silently ignore – country detection is best-effort
    return None

def best_country_from_row(row: pd.Series) -> Optional[str]:
    # try any reasonable property first
    for key in ["properties.country", "country", "properties.admin", "properties.adm0name",
                "properties.Country", "properties.region"]:
        if key in row and isinstance(row[key], str) and row[key].strip():
            return row[key].strip()
    # then look across the row for OneBuilding-like URLs
    for c in row.index:
        v = row[c]
        if isinstance(v, str) and (".epw" in v.lower() or ".zip" in v.lower() or "onebuilding" in v.lower()):
            cn = country_from_zip_url(v)
            if cn:
                return cn
    return None




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


PVLIB_COLUMN_MAP = {
    "data_source_and_uncertainty_flags": "datasource",
    "dry_bulb": "drybulb",
    "temp_air": "drybulb",
    "dew_point": "dewpoint",
    "temp_dew": "dewpoint",
    "dew_temperature": "dewpoint",
    "rel_hum": "relhum",
    "relative_humidity": "relhum",
    "rel_humidity": "relhum",
    "atm_press": "atmos_pressure",
    "atmospheric_pressure": "atmos_pressure",
    "pressure": "atmos_pressure",
    "et_rad": "exthorrad",
    "et_dn_rad": "extdirrad",
    "et_hr_sky_rad": "horirsky",
    "hor_ir_sky": "horirsky",
    "glo_hor_rad": "glohorrad",
    "ghi": "glohorrad",
    "dir_nor_rad": "dirnorrad",
    "dni": "dirnorrad",
    "dif_hor_rad": "difhorrad",
    "dhi": "difhorrad",
    "glo_hor_illum": "glohorillum",
    "dir_nor_illum": "dirnorillum",
    "dif_hor_illum": "difhorillum",
    "zenith_luminance": "zenlum",
    "wind_direction": "winddir",
    "wind_dir": "winddir",
    "wind_speed": "windspd",
    "total_sky_cover": "totskycvr",
    "opaque_sky_cover": "opaqskycvr",
    "visibility": "visibility",
    "ceiling_height": "ceiling_hgt",
    "pres_wthr_obs": "presweathobs",
    "pres_wthr_codes": "presweathcodes",
    "precip_wtr": "precip_wtr",
    "aerosol_opt_depth": "aerosol_opt_depth",
    "snow_depth": "snowdepth",
    "days_last_snow": "days_last_snow",
    "albedo": "albedo",
    "liq_precip_depth": "liq_precip_depth",
    "liq_precip_rate": "liq_precip_rate",
}

def _parse_location(line: str):
    parts = [p.strip() for p in line.split(",")] + [""]*10
    def fnum(x):
        try:
            return float(x) if x != "" else None
        except (ValueError, TypeError):
            return None
    return dict(
        city=parts[1], state_province=parts[2], country=parts[3], source=parts[4], wmo=parts[5],
        latitude=fnum(parts[6]), longitude=fnum(parts[7]),
        timezone=fnum(parts[8]), elevation_m=fnum(parts[9])
    )

def read_epw_with_schema(epw_bytes_or_path: Union[bytes, str, Path]):
    # Return (header: dict, df: DataFrame indexed by timestamp, notes: list).
    notes: List[str] = []

    pvlib_result = _read_epw_via_pvlib(epw_bytes_or_path)
    if pvlib_result is not None:
        header, df = pvlib_result
        notes.append("Parsed via pvlib.iotools.read_epw().")
    else:
        if isinstance(epw_bytes_or_path, (str, Path)):
            text = Path(epw_bytes_or_path).read_text(encoding="latin-1", errors="replace")
        else:
            text = epw_bytes_or_path.decode("latin-1", errors="replace")

        lines = text.splitlines()
        header_lines, data_lines = lines[:8], lines[8:]
        header = {
            "location": _parse_location(header_lines[0]),
            "design_conditions": header_lines[1],
            "typical_extreme_periods": header_lines[2],
            "ground_temps": header_lines[3],
            "holidays_dst": header_lines[4],
            "data_periods": header_lines[5],
            "comments1": header_lines[6] if len(header_lines) > 6 else "",
            "comments2": header_lines[7] if len(header_lines) > 7 else "",
        }

        rows = list(csv.reader(data_lines))
        df = pd.DataFrame(rows).iloc[:, :len(EPW_COLUMNS)]
        df.columns = EPW_COLUMNS

        for c in ("year","month","day","hour","minute"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        for c in df.columns:
            if c in ("year","month","day","hour","minute","datasource","presweathobs","presweathcodes"):
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # EPW hour is end-of-hour; shift by -1h so 01:00 becomes 00:00–01:00 period start
        ts = pd.to_datetime(dict(year=df["year"],month=df["month"],day=df["day"],hour=df["hour"]), errors="coerce") \
             - pd.to_timedelta(1,"h")
        df.index = ts
        df.index.name = "timestamp"

    df, continuity_notes = _enforce_epw_hourly_profile(df)
    notes.extend(continuity_notes)

    # Memory downcasting: keep EPW weather columns as float32.
    df = _downcast_float32(df)

    return header, df, notes

def build_clima_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Adds derived fields and convenience columns; returns cdf.
    wx = df.copy()
    
    if isinstance(wx.index, pd.DatetimeIndex):
        try:
            # Force all timestamps to a single reference year (2021) to prevent day-of-year gaps
            # caused by leap/non-leap month transitions in typical meteorological year (TMY) datasets.
            new_years = np.full(len(wx.index), 2021)
            new_months = wx.index.month
            new_days = wx.index.day
            # Map Feb 29 to Feb 28 for non-leap year compatibility
            feb29_mask = (new_months == 2) & (new_days == 29)
            new_days = np.where(feb29_mask, 28, new_days)
            
            wx.index = pd.to_datetime(dict(
                year=new_years,
                month=new_months,
                day=new_days,
                hour=wx.index.hour,
                minute=wx.index.minute
            ))
            wx = wx.sort_index()
        except Exception:
            pass

    for c in ["drybulb","dewpoint","relhum","atmos_pressure","windspd","winddir",
              "glohorrad","dirnorrad","difhorrad","exthorrad","extdirrad"]:
        if c in wx: wx[c] = wx[c].interpolate(limit=3, limit_direction="both")
    if "relhum" in wx: wx["relhum"] = wx["relhum"].clip(0, 100)

    # vapor/absolute humidity + wet-bulb + humidity ratio + enthalpy
    T  = wx["drybulb"] if "drybulb" in wx else pd.Series(index=wx.index, dtype=float)
    RH = wx["relhum"]  if "relhum"  in wx else pd.Series(index=wx.index, dtype=float)
    es = 610.94 * np.exp(17.625*T/(T+243.04))
    wx["sat_press"] = es
    wx["vap_press"] = es*(RH/100.0)
    wx["abs_hum"]   = 216.7 * wx["vap_press"] / (T + 273.15)
        # Stull (2011) J. Appl. Meteorol. Climatol. 50(11): valid range T: 5-45°C, RH: 5-99%
    T_clamp = T.clip(5, 45)   # clamp for formula validity
    RH_clamp = RH.clip(5, 99)
    wx['twb'] = (T_clamp * np.arctan(0.151977 * np.sqrt(RH_clamp + 8.313659))
                 + np.arctan(T_clamp + RH_clamp)
                 - np.arctan(RH_clamp - 1.676331)
                 + 0.00391838 * (RH_clamp**1.5) * np.arctan(0.023101 * RH_clamp)
                 - 4.686035)
    # Mask out-of-range results with NaN instead of producing bad values
    out_of_range = (T < 5) | (T > 45) | (RH < 5) | (RH > 99)
    wx.loc[out_of_range, 'twb'] = np.nan

    P = wx["atmos_pressure"].fillna(101325.0) if "atmos_pressure" in wx else pd.Series(101325.0, index=wx.index)
    Pv = wx["vap_press"]
    wx["w"] = 0.62198 * Pv / (P - Pv)
    wx["h_kJ_per_kg_dry"] = 1.006 * T + wx["w"] * (2501.0 + 1.86 * T)

    cdf = wx.copy()
    cdf["month"] = cdf.index.month; cdf["day"] = cdf.index.day; cdf["hour"] = cdf.index.hour; cdf["doy"] = cdf.index.dayofyear

    # Memory downcasting: derived weather metrics can stay float32.
    cdf = _downcast_float32(cdf)
    return cdf


@CACHE(show_spinner=False)
def read_epw_with_schema_cached(epw_bytes_or_path: Union[bytes, str, Path]):
    return read_epw_with_schema(epw_bytes_or_path)


@CACHE(show_spinner=False)
def build_clima_dataframe_cached(df: pd.DataFrame) -> pd.DataFrame:
    return build_clima_dataframe(df)


def _hash_bytes(data: Optional[bytes]) -> Optional[str]:
    if not data:
        return None
    return hashlib.sha256(data).hexdigest()

@st.cache_data(show_spinner="Computing thermal comfort metrics...")
def build_comfort_package(cdf: pd.DataFrame) -> Dict[str, Optional[pd.DataFrame]]:
    # Compute comfort + load summaries used across multiple tabs.
    package: Dict[str, Optional[pd.DataFrame]] = {
        "di": None,
        "utci": None,
        "pmv": None,
        "heat_index": None,
        "humidex": None,
        "comfort_annual": pd.DataFrame(),
        "comfort_monthly": pd.DataFrame(),
        "degree_daily": pd.DataFrame(),
        "loads_annual": pd.DataFrame(),
    }

    if cdf is None or cdf.empty:
        return package

    def _log_comfort_warning(message: str) -> None:
        # Avoid Streamlit UI calls inside cached functions.
        print(f"[comfort-cache] {message}")

    di = None
    if {"drybulb", "relhum"}.issubset(cdf.columns):
        try:
            di = ce.compute_di(cdf)
            package["di"] = di
        except Exception as e:
            di = None
            _log_comfort_warning(f"DI failed: {e}")

    utci = None
    if {"drybulb", "relhum", "windspd"}.issubset(cdf.columns):
        try:
            utci = ce.compute_utci_approx(cdf)

            package["utci"] = utci
        except Exception as e:
            utci = None
            _log_comfort_warning(f"UTCI failed: {e}")

    pmv = None
    if _IS_STREAMLIT_CLOUD:
        _log_comfort_warning("PMV skipped on Streamlit Cloud lightweight mode")
    elif {"drybulb", "relhum", "windspd"}.issubset(cdf.columns):
        try:
            pmv = ce.compute_pmv(cdf)
            package["pmv"] = pmv
        except Exception as e:
            pmv = None
            _log_comfort_warning(f"PMV failed: {e}")

    heat_index = None
    if {"drybulb", "relhum"}.issubset(cdf.columns):
        try:
            heat_index = ce.compute_heat_index(cdf)
            package["heat_index"] = heat_index
        except Exception as e:
            heat_index = None
            _log_comfort_warning(f"Heat Index failed: {e}")

    humidex = None
    if {"drybulb", "relhum"}.issubset(cdf.columns):
        try:
            humidex = ce.compute_humidex(cdf)
            package["humidex"] = humidex
        except Exception as e:
            humidex = None
            _log_comfort_warning(f"Humidex failed: {e}")

    try:
        package["comfort_annual"] = ce.summarize_comfort(cdf, di, utci, freq="YE")
    except Exception as e:
        _log_comfort_warning(f"comfort_annual failed: {e}")

    try:
        package["comfort_monthly"] = ce.summarize_comfort(cdf, di, utci, freq="ME")
    except Exception as e:
        _log_comfort_warning(f"comfort_monthly failed: {e}")

    try:
        package["degree_daily"] = ce.compute_degree_metrics(cdf, freq="D")
    except Exception as e:
        _log_comfort_warning(f"degree_daily failed: {e}")

    try:
        deg_src = package["degree_daily"] if isinstance(package["degree_daily"], pd.DataFrame) else None
        if deg_src is None or deg_src.empty:
            deg_src = ce.compute_degree_metrics(cdf)
        package["loads_annual"] = ce.summarize_loads(deg_src, freq="YE")
    except Exception as e:
        _log_comfort_warning(f"loads_annual failed: {e}")

    return package

# -------------------- Optional: helpers for ZIP/URL --------------------
def read_epw_from_zip_bytes(zip_bytes: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as z:
        epws = [m for m in z.namelist() if m.lower().endswith('.epw')]
        if not epws:
            raise ValueError(
                'The uploaded ZIP file contains no EPW weather files. '
                'Please upload a ZIP that includes a .epw file.'
            )
        epws.sort(key=lambda m: z.getinfo(m).file_size, reverse=True)
        with z.open(epws[0]) as f:
            return f.read()

def compose_epw_text(header: dict, df: pd.DataFrame) -> bytes:
    # Rebuild an EPW text blob from header metadata and a data frame.

    # (_format_cell removed — unused dead code; _fmt below is used instead)

    def _fmt(value, precision=6):
        if value is None:
            return ""
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return ""
            return f"{value:.{precision}f}".rstrip("0").rstrip(".")
        return str(value)

    loc = header.get("location", {})
    loc_line = ",".join(
        [
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
        ]
    )
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

    def _fmt_series(series: pd.Series) -> pd.Series:
        """Vectorized formatter for a single column."""
        result = series.copy().astype(str)
        float_mask = series.apply(lambda x: isinstance(x, (float, np.floating)))
        result[float_mask] = series[float_mask].apply(
            lambda v: '' if pd.isna(v) else f'{v:.6f}'.rstrip('0').rstrip('.')
        )
        result[series.isna()] = ''
        return result

    formatted = pd.DataFrame({
        col: _fmt_series(df[col]) for col in EPW_COLUMNS
    })
    row_strings = formatted.apply(lambda row: ','.join(row), axis=1).tolist()
    data_csv = "\n".join(row_strings) + "\n"
    epw_text = "\n".join(header_lines) + "\n" + data_csv
    return epw_text.encode("latin-1", errors="replace")


def _meta_lookup(meta: Optional[dict], *keys: str) -> Optional[object]:
    if not isinstance(meta, dict):
        return None
    lowered = {k.lower(): v for k, v in meta.items()}
    for key in keys:
        if key.lower() in lowered:
            return lowered[key.lower()]
    # recurse into nested dicts
    for value in meta.values():
        if isinstance(value, dict):
            found = _meta_lookup(value, *keys)
            if found is not None:
                return found
    return None


def _build_header_from_pvlib_meta(meta: Optional[dict]) -> dict:
    location = {
        "city": _meta_lookup(meta, "city") or "",
        "state_province": _meta_lookup(meta, "state_province", "state" , "province") or "",
        "country": _meta_lookup(meta, "country") or "",
        "source": _meta_lookup(meta, "data_source", "source") or "pvlib",
        "wmo": _meta_lookup(meta, "wmo", "station_id", "wmo_code") or "",
        "latitude": _meta_lookup(meta, "latitude", "lat"),
        "longitude": _meta_lookup(meta, "longitude", "lon"),
        "timezone": _meta_lookup(meta, "timezone", "time_zone", "tz"),
        "elevation_m": _meta_lookup(meta, "elevation", "altitude"),
        "period": _meta_lookup(meta, "data_periods", "period"),
    }
    header = {
        "location": location,
        "design_conditions": _meta_lookup(meta, "design_conditions") or "",
        "typical_extreme_periods": _meta_lookup(meta, "typical_extreme_periods") or "",
        "ground_temps": _meta_lookup(meta, "ground_temps") or "",
        "holidays_dst": _meta_lookup(meta, "holidays_dst") or "",
        "data_periods": _meta_lookup(meta, "data_periods") or location.get("period", ""),
        "comments1": _meta_lookup(meta, "comments1") or "",
        "comments2": _meta_lookup(meta, "comments2") or "",
    }
    return header


def _read_epw_via_pvlib(epw_bytes_or_path: Union[bytes, str, Path]) -> Optional[Tuple[dict, pd.DataFrame]]:
    try:
        from pvlib.iotools import read_epw as pvlib_read_epw
    except Exception:
        return None

    source = epw_bytes_or_path
    buffer = None
    if isinstance(epw_bytes_or_path, (bytes, bytearray)):
        text = epw_bytes_or_path.decode("latin-1", errors="replace")
        buffer = io.StringIO(text)
        source = buffer
    elif isinstance(epw_bytes_or_path, Path):
        source = str(epw_bytes_or_path)

    try:
        pv_df, meta = pvlib_read_epw(source)
    except Exception:
        return None

    df = pv_df.copy()
    df.index = pd.to_datetime(df.index)
    df.index = df.index - pd.to_timedelta(1, "h")
    df.index.name = "timestamp"
    df.columns = [str(c).lower() for c in df.columns]
    df = df.rename(columns=PVLIB_COLUMN_MAP)
    if "datasource" not in df.columns:
        df["datasource"] = 0

    header = _build_header_from_pvlib_meta(meta if isinstance(meta, dict) else {})
    return header, df


def _infer_reference_year(df: pd.DataFrame) -> int:
    if "year" in df.columns:
        years = df["year"].dropna().astype(int)
        if not years.empty:
            return int(years.iloc[0])
    idx = getattr(df, "index", None)
    if isinstance(idx, pd.DatetimeIndex) and len(idx):
        return int(idx[0].year)
    return 2001


def _refresh_calendar_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    for col_name, values in {
        "year": df.index.year,
        "month": df.index.month,
        "day": df.index.day,
        "hour": df.index.hour,
        "minute": df.index.minute,
    }.items():
        df[col_name] = pd.Series(values, index=df.index, dtype="Int64")
    return df


def _enforce_epw_hourly_profile(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    notes: List[str] = []
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
        return df, notes

    df = df.sort_index()

    # Remove duplicate timestamps
    duplicated = df.index.duplicated()
    if duplicated.any():
        removed = int(duplicated.sum())
        df = df[~duplicated]
        notes.append(f"Removed {removed} duplicate hourly rows in EPW file.")

    record_count = len(df)

    # Handle leap-year EPWs by dropping Feb 29
    if record_count == 8784:
        leap_mask = (df.index.month == 2) & (df.index.day == 29)
        removed = int(leap_mask.sum())
        if removed:
            df = df.loc[~leap_mask]
            notes.append("Detected leap-year EPW (8784 hours). Dropped Feb 29 data to keep 8760 hours.")
        record_count = len(df)

    if record_count != 8760:
        reference_year = _infer_reference_year(df)
        target_index = pd.date_range(
            start=pd.Timestamp(reference_year, 1, 1, 0, 0, 0),
            periods=8760,
            freq="h",
        )
        df = df.reindex(target_index)
        float_cols = df.select_dtypes(include=[np.floating]).columns
        if len(float_cols):
            df[float_cols] = df[float_cols].interpolate(limit_direction="both")
        int_cols = df.select_dtypes(include=["Int64", "int32", "int64"]).columns
        for col in int_cols:
            df[col] = df[col].ffill().bfill()
        object_cols = df.select_dtypes(include=["object"]).columns
        for col in object_cols:
            df[col] = df[col].ffill().bfill()
        notes.append(
            f"EPW contained {record_count} records. Reindexed to 8760 hours and filled gaps via interpolation."
        )

    df = _refresh_calendar_columns(df)
    return df, notes



# ========== PREMIUM CUSTOM STYLING ==========
#"""====================
#PHASE A — CONTROLLER
#Runs first on every rerun. NO UI rendering.
#===================="""


# NOTE: Do NOT cache run_controller — it reads/mutates session state indirectly
# and caching causes stale results when the same inputs recur.
def run_controller(
    *,
    cdf_present: bool,
    load_requested: bool,
    is_loading: bool,
    pending_station_id: Optional[str],
    sel_station_url: Optional[str],
    pending_station: Optional[dict],
    sel_station_alt_urls: Tuple[str, ...],
    raw_epw_bytes: Optional[bytes],
    last_parsed_epw_hash: Optional[str],
    was_epw_ready: bool,
) -> Dict[str, object]:
    """Pure-ish controller: returns the session-state updates to apply."""
    out: Dict[str, object] = {
        "epw_ready": bool(cdf_present),
        "active_page": "dashboard" if cdf_present else "select_station",
        "should_rerun": False,
        "set": {},
        "pop": [],
        "station_load_error": None,
        "station_load_debug": None,
    }

    set_updates: Dict[str, object] = {}
    pop_keys: List[str] = []

    # If we're already ready, keep routing locked to dashboard and clear any pending selection state.
    if cdf_present:
        set_updates.update({
            "load_requested": False,
            "is_loading": False,
            "pending_station_id": None,
        })
        pop_keys.extend([
            "pending_station",
            "sel_station_url",
            "sel_station",
            "selected_station",
            "sel_station_alt_urls",
            "loading_station_name",
        ])
        out["set"] = set_updates
        out["pop"] = pop_keys
        return out

    # Downloads are synchronous in Streamlit. If a previous cloud run was
    # interrupted while the flag was true, clear it so the app does not sit in
    # a permanent loading state on the next render.
    if is_loading:
        set_updates["is_loading"] = False
        is_loading = False
        if not load_requested:
            pop_keys.append("loading_station_name")

    # One-shot station download+parse state machine.
    should_start_station_load = (
        out["active_page"] == "select_station"
        and bool(load_requested)
        and not bool(is_loading)
        and bool(pending_station_id)
        and bool(sel_station_url)
    )

    if should_start_station_load:
        set_updates["is_loading"] = True
        set_updates["load_requested"] = False
        pop_keys.extend(["station_load_error", "station_load_debug"])

        url = sel_station_url
        station_info = pending_station or {}
        station_name = str(station_info.get("name") or "selected station")

        if '<a ' in str(url) and 'href' in str(url):
            url = _extract_url(str(url)) or str(url)

        urls_to_try = fix_station_url(str(url))
        urls_to_try.extend([u for u in sel_station_alt_urls if u])

        attempted_urls: List[str] = []
        attempt_notes: List[str] = []
        fetched_bytes: Optional[bytes] = None
        successful_url: Optional[str] = None

        for i, test_url in enumerate(urls_to_try):
            attempted_urls.append(str(test_url))
            url_clean = re.search(r"https.*?\.epw", str(test_url))
            if url_clean:
                test_url = url_clean.group()
            fetched_bytes, _err = fetch_epw_bytes_no_ui(str(test_url))
            if fetched_bytes is not None:
                successful_url = str(test_url)
                break
            attempt_notes.append(f"Attempt {i+1}/{len(urls_to_try)} failed")

        if fetched_bytes is None:
            fallback_sources = ALTERNATIVE_EPW_SOURCES[:1] if _IS_STREAMLIT_CLOUD else ALTERNATIVE_EPW_SOURCES
            for j, alt_url in enumerate(fallback_sources, start=1):
                attempted_urls.append(str(alt_url))
                fetched_bytes, _err = fetch_epw_bytes_no_ui(str(alt_url))
                if fetched_bytes is not None:
                    successful_url = str(alt_url)
                    break
                attempt_notes.append(f"Alternate {j}/{len(fallback_sources)} failed")

        if fetched_bytes is None:
            set_updates["is_loading"] = False
            out["station_load_error"] = f"❌ Could not fetch EPW from **{station_name}**. All download attempts failed."
            out["station_load_debug"] = {
                "station_name": station_name,
                "original_url": str(url),
                "urls_to_try": urls_to_try,
                "successful_url": successful_url,
                "attempt_notes": attempt_notes,
                "attempted_urls": attempted_urls,
            }
        else:
            set_updates["raw_epw_bytes"] = fetched_bytes
            source_label = (
                f"Station: {station_name}"
                if successful_url == str(url)
                else f"Station: {station_name} (alt)"
            )
            set_updates["source_label"] = source_label
            set_updates["page_after_station"] = "Overview"

            # Parse/build only if bytes changed.
            epw_hash = _hash_bytes(fetched_bytes)
            if epw_hash and epw_hash != last_parsed_epw_hash:
                header, df, epw_notes = read_epw_with_schema_cached(fetched_bytes)
                cdf = build_clima_dataframe_cached(df)

                # Compute comfort package once per loaded file.
                comfort_pkg = build_comfort_package(cdf)

                set_updates.update({
                    "header": header,
                    "cdf": cdf,
                    "df": df,
                    "comfort_pkg": comfort_pkg,
                    "cdf_raw": cdf.copy(deep=True),
                    "_last_parsed_epw_hash": epw_hash,
                    "_last_epw_notes": epw_notes,
                })

            epw_ready_now = bool(set_updates.get("cdf") is not None) or bool(cdf_present)
            out["epw_ready"] = epw_ready_now

            if epw_ready_now:
                out["active_page"] = "dashboard"
                set_updates.update({
                    "_just_loaded_epw": True,
                    "nav_page": "Overview",
                    "sidebar_nav": "Overview",
                    "last_loaded_station_id": pending_station_id,
                    "pending_station_id": None,
                    "sel_station_url": None,
                    "is_loading": False,
                    "_clear_map_on_next_run": True,
                })
                pop_keys.extend([
                    "pending_station",
                    "sel_station",
                    "selected_station",
                    "sel_station_alt_urls",
                    "loading_station_name",
                ])
                if not was_epw_ready:
                    out["should_rerun"] = True

    # Upload path: if bytes exist but not parsed yet, parse once.
    if raw_epw_bytes is not None and not bool(out.get("epw_ready")) and not bool(is_loading):
        epw_hash = _hash_bytes(raw_epw_bytes)
        if epw_hash and epw_hash != last_parsed_epw_hash:
            header, df, epw_notes = read_epw_with_schema_cached(raw_epw_bytes)
            cdf = build_clima_dataframe_cached(df)
            comfort_pkg = build_comfort_package(cdf)

            set_updates.update({
                "header": header,
                "cdf": cdf,
                "df": df,
                "comfort_pkg": comfort_pkg,
                "cdf_raw": cdf.copy(deep=True),
                "_last_parsed_epw_hash": epw_hash,
                "_last_epw_notes": epw_notes,
            })

            out["epw_ready"] = True
            out["active_page"] = "dashboard"
            set_updates.update({
                "_just_loaded_epw": True,
                "nav_page": "Overview",
                "sidebar_nav": "Overview",
                "_clear_map_on_next_run": True,
            })
            if not was_epw_ready:
                out["should_rerun"] = True

    out["set"] = set_updates
    out["pop"] = pop_keys
    return out


ss = st.session_state

# Derive readiness from parsed data.
_was_epw_ready = bool(ss.get("cdf") is not None)
ss["epw_ready"] = _was_epw_ready

_should_sync = bool(ss.get("load_requested")) or (
    ss.get("raw_epw_bytes") is not None
    and not bool(ss.get("epw_ready"))
    and not bool(ss.get("is_loading"))
)

with st.spinner("Synchronizing Data...") if _should_sync else nullcontext():
    controller_out = run_controller(
        cdf_present=bool(ss.get("cdf") is not None),
        load_requested=bool(ss.get("load_requested")),
        is_loading=bool(ss.get("is_loading")),
        pending_station_id=ss.get("pending_station_id"),
        sel_station_url=ss.get("sel_station_url"),
        pending_station=ss.get("pending_station") or ss.get("sel_station") or None,
        sel_station_alt_urls=tuple(ss.get("sel_station_alt_urls") or ()),
        raw_epw_bytes=ss.get("raw_epw_bytes"),
        last_parsed_epw_hash=ss.get("_last_parsed_epw_hash"),
        was_epw_ready=_was_epw_ready,
    )

ss["epw_ready"] = bool(controller_out.get("epw_ready"))
ss["active_page"] = str(controller_out.get("active_page") or "select_station")

for k in controller_out.get("pop", []) or []:
    try:
        ss.pop(k, None)
    except Exception as e:
        st.warning(f"pop failed: {e}")

for k, v in (controller_out.get("set", {}) or {}).items():
    ss[k] = v

if controller_out.get("station_load_error"):
    ss["station_load_error"] = controller_out.get("station_load_error")
if controller_out.get("station_load_debug"):
    ss["station_load_debug"] = controller_out.get("station_load_debug")

# One-shot rerun after a successful parse/build so rendering always lands on the dashboard.
if bool(controller_out.get("should_rerun")):
    _rerun()

PREMIUM_CSS = '''
<style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Poppins:wght@600;700&display=swap");

:root {
    --bg: #0f172a;
    --panel: #111827;
    --panel-2: #1e293b;
    --text: #f1f5f9;
    --muted: #94a3b8;
    --primary: #3b82f6;
    --primary-2: #06b6d4;
    --accent: #8b5cf6;
    --accent-2: #ec4899;
    --glow: 0 12px 40px rgba(59, 130, 246, 0.35);
    --glass: rgba(30, 41, 59, 0.7);
}

* { font-family: "Inter", "Poppins", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
html, body, .main, .block-container { background: var(--bg); color: var(--muted); scroll-behavior: smooth; }
body::before {
    content: "";
    position: fixed;
    inset: 0;
    background: radial-gradient(circle at 20% 20%, rgba(59,130,246,0.2), transparent 35%),
                radial-gradient(circle at 80% 10%, rgba(8,47,73,0.25), transparent 35%),
                radial-gradient(circle at 50% 80%, rgba(236,72,153,0.15), transparent 35%);
    filter: blur(60px);
    opacity: 0.9;
    z-index: -2;
    animation: meshShift 18s ease-in-out infinite alternate;
}
body::after {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image: radial-gradient(rgba(255,255,255,0.05) 1px, transparent 1px);
    background-size: 120px 120px;
    opacity: 0.35;
    z-index: -1;
}

.block-container { padding: clamp(1rem, 3vw, 1.8rem) clamp(1.2rem, 5vw, 2.6rem); }
.main > .block-container { background: var(--bg); position: relative; z-index: 0; }

h1 { font-size: 2.2rem !important; font-weight: 800 !important; color: var(--text) !important; margin-bottom: 0.35rem !important; letter-spacing: -0.02em; }
h2 { font-size: 1.5rem !important; font-weight: 700 !important; color: #e2e8f0 !important; margin: 1.2rem 0 0.7rem 0 !important; letter-spacing: -0.02em; }
h3 { font-size: 1.15rem !important; font-weight: 650 !important; color: #d9e1ec !important; margin-bottom: 0.6rem !important; letter-spacing: -0.01em; }

.card { background: var(--panel); border-radius: 14px; padding: 1.15rem 1.25rem; border: 1px solid rgba(255,255,255,0.04); box-shadow: 0 20px 60px rgba(0,0,0,0.35); transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease; }
.card:hover { border-color: rgba(96,165,250,0.35); transform: translateY(-2px); box-shadow: 0 18px 46px rgba(59,130,246,0.18); }

[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 700 !important; color: var(--text) !important; letter-spacing: -0.02em; }
[data-testid="stMetricLabel"] { font-size: 0.82rem !important; color: var(--muted) !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.06em; }

.stTabs [data-baseweb="tab-list"] { gap: 10px; background: rgba(17,24,39,0.7); padding: 10px 6px 14px 6px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); box-shadow: 0 12px 30px rgba(0,0,0,0.35); }
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 10px; padding: 8px 12px; font-weight: 600; color: var(--muted); border: none; transition: color 0.15s ease, background 0.15s ease, transform 0.15s ease; }
.stTabs [data-baseweb="tab"]:hover { color: var(--text); transform: translateY(-1px); }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, var(--primary), var(--primary-2)) !important; color: #0b1220 !important; box-shadow: 0 12px 24px rgba(59,130,246,0.3); }
/* Tab content isolation - prevent bleed-through */
.stTabs [data-baseweb="tab-panel"] { overflow: hidden; position: relative; z-index: 1; background: rgba(15,23,42,0.96); border: 1px solid rgba(255,255,255,0.05); border-radius: 0 0 14px 14px; padding: 1rem 1rem 0.25rem; margin-top: 0.55rem; box-shadow: 0 18px 40px rgba(0,0,0,0.22); }
.stTabs [data-baseweb="tab-panel"] > div { overflow: hidden; position: relative; }
.stTabs > div:nth-of-type(2) > div > div { overflow: hidden; position: relative; }
[role="tabpanel"] { overflow: hidden !important; background: transparent !important; }
.stTabs [role="tabpanel"][hidden],
.stTabs [role="tabpanel"][aria-hidden="true"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    overflow: hidden !important;
}
.stTabs [role="tabpanel"]:not([hidden]):not([aria-hidden="true"]),
.stTabs [data-baseweb="tab-panel"]:not([hidden]),
.stTabs [data-baseweb="tab-panel"]:not([hidden]) > div {
    overflow: visible !important;
    min-height: 420px;
}

[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #0b1220, #0f172a 60%, #0b1220) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
    box-shadow: 12px 0 28px rgba(0,0,0,0.35);
}
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: #0b1220 !important;
    border: 1px solid rgba(148,163,184,0.28) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 14px 32px rgba(0,0,0,0.34) !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] details,
[data-testid="stSidebar"] [data-testid="stExpander"] summary,
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stVerticalBlock"] {
    background: #0b1220 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label { margin-bottom: 6px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); background: rgba(255,255,255,0.02); padding: 10px 12px; transition: all 0.2s ease; }
[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:hover { border-color: rgba(59,130,246,0.7); box-shadow: 0 8px 22px rgba(59,130,246,0.25); transform: translateX(4px); color: #e2e8f0; }
[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label [data-testid="stMarkdownContainer"] p { margin: 0; font-weight: 600; color: var(--muted); }
[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] { border-color: var(--primary); box-shadow: 4px 0 0 var(--primary) inset, 0 12px 28px rgba(59,130,246,0.35); background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(6,182,212,0.18)); color: var(--text); }

.sidebar-brand { margin-top: auto; padding: 1rem 0.5rem 0.75rem 0.5rem; color: var(--muted); font-size: 0.9rem; border-top: 1px solid rgba(255,255,255,0.06); }
.sidebar-brand strong { color: var(--text); }

.stButton button { background: linear-gradient(135deg, var(--primary), var(--primary-2)); color: #0b1220; border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 0.4rem 0.9rem; font-weight: 700; font-size: 0.95rem; transition: transform 0.15s ease, box-shadow 0.15s ease; height: 38px; box-shadow: var(--glow); }
.stButton button:hover { transform: translateY(-1px) scale(1.01); box-shadow: 0 14px 28px rgba(6,182,212,0.35); }
.stButton button:active { transform: translateY(0); box-shadow: none; }
.stButton button[data-testid="baseButton-secondary"] { background: rgba(255,255,255,0.06); color: var(--text); border: 1px solid rgba(255,255,255,0.12); box-shadow: none; }
.stButton button:disabled,
.stButton button[disabled],
.stDownloadButton button:disabled,
.stDownloadButton button[disabled],
button[data-testid="baseButton-secondary"]:disabled {
    background: rgba(51,65,85,0.62) !important;
    color: rgba(226,232,240,0.42) !important;
    border: 1px solid rgba(148,163,184,0.20) !important;
    box-shadow: none !important;
    opacity: 1 !important;
    transform: none !important;
    cursor: not-allowed !important;
    text-shadow: none !important;
}

div[data-testid="stFileUploader"] section { border: 1px dashed rgba(96,165,250,0.35); border-radius: 12px; padding: 0.75rem; background: rgba(17,24,39,0.8); box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02); }
.dataframe { border-radius: 10px; overflow: hidden; box-shadow: 0 16px 40px rgba(0,0,0,0.38); border: 1px solid rgba(255,255,255,0.03); }
.dataframe th { background: rgba(255, 255, 255, 0.04) !important; color: #e2e8f0 !important; font-weight: 700 !important; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.08em; padding: 0.75rem !important; }
.dataframe td { padding: 0.7rem 0.8rem !important; border-bottom: 1px solid rgba(255, 255, 255, 0.04) !important; color: #d3dae5 !important; }
.dataframe tr:hover { background: rgba(255, 255, 255, 0.02) !important; }
.stAlert { border-radius: 12px; box-shadow: 0 10px 28px rgba(0,0,0,0.35); border: 1px solid rgba(255,255,255,0.08); }
.js-plotly-plot { border-radius: 12px; background: #0f172a !important; box-shadow: 0 18px 50px rgba(0,0,0,0.38); }
.js-plotly-plot text,
.plotly .xtick text,
.plotly .ytick text,
.plotly .xaxislayer-above text,
.plotly .yaxislayer-above text,
.plotly .legend text { fill: #e2e8f0 !important; color: #e2e8f0 !important; }
.plotly .gridlayer path { stroke: rgba(148, 163, 184, 0.2) !important; }
.light-chart .plotly .xtick text,
.light-chart .plotly .ytick text,
.light-chart .plotly .legend text { fill: #1e293b !important; color: #1e293b !important; }
.station-info { background: var(--panel-2); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 1rem 1.1rem; margin: 0.65rem 0 1rem 0; font-weight: 500; color: #d5dbe7; box-shadow: 0 18px 48px rgba(0,0,0,0.32); }
.station-info-title { font-size: 1.2rem; font-weight: 700; color: var(--text); margin-bottom: 0.5rem; letter-spacing: -0.01em; display: flex; align-items: center; gap: 0.5rem; }
.station-info-detail { font-size: 0.95rem; line-height: 1.6; color: #cdd4df; }
.station-info-detail strong { color: #e2e8f0; font-weight: 650; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.05em; }
.station-country { font-size: 0.92rem; color: #e2e8f0; font-weight: 650; margin-bottom: 0.45rem; padding: 0.32rem 0.65rem; background: rgba(255, 255, 255, 0.03); border-radius: 10px; display: inline-block; border: 1px solid rgba(255, 255, 255, 0.08); }


/* ---- PROFESSIONAL LANDING LAYOUT ---- */
.block-container { padding-top: 0.5rem; }

/* Hide the Streamlit colored decoration bar at the very top */
[data-testid="stDecoration"],
header[data-testid="stHeader"] {
    display: none !important;
}

.app-header {
    margin: -1rem -1rem 1rem -1rem;
    padding: 0.85rem 2rem;
    background: linear-gradient(135deg, #020e1a 0%, #0a1929 50%, #0d2137 100%);
    border-bottom: 1px solid rgba(59, 130, 246, 0.15);
    box-shadow: 0 2px 16px rgba(0,0,0,0.45);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 2rem;
    z-index: 9999;
    min-width: 0;
}

.header-left {
    display: flex;
    align-items: center;
    gap: 1rem;
    min-width: 0;                       /* Allow text to shrink gracefully */
    flex: 1 1 auto;
}

.header-logo {
    max-height: 48px;
    width: auto;
    border-radius: 999px;
    box-shadow: 0 0 15px rgba(59,130,246,0.4);
    flex-shrink: 0;                     /* Never squish the logo */
}

.header-text {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    min-width: 0;                       /* Allow text truncation */
}

.header-title {
    font-size: 1.15rem;
    font-weight: 800;
    color: #f9fafb;
    letter-spacing: -0.01em;
    line-height: 1.1;
    white-space: nowrap;
}

.header-location {
    font-size: 0.8rem;
    color: #cbd5e1;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.header-right {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    flex-shrink: 0;                     /* Right side never squishes */
}

.header-ub {
    max-height: 44px;
    width: auto;
    opacity: 0.98;
    flex-shrink: 0;
}

/* make sure old center/glow layout is turned off */
.logo-glow,
.header-center,
.header-icons,
.brand-text,
.brand-title,
.brand-sub,
.header-sub {
    display: none !important;
}

/* ---- HERO SECTION (COMPACT & FOCUSED) ---- */
.hero-section {
    max-width: 680px;          /* Highly focused width */
    margin: 0 auto 2rem auto;
}

.hero-card {
    background: #020617;
    border-radius: 8px;
    border: 1px solid rgba(51, 65, 85, 0.6);
    padding: 0.4rem 0.9rem;
    margin-bottom: 0.25rem;
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    flex-wrap: wrap;
}

.hero-card h2 {
    margin: 0;
    font-size: 0.9rem;
    font-weight: 700;
    color: #e5e7eb;
    white-space: nowrap;
}

.hero-card p {
    margin: 0;
    font-size: 0.78rem;
    color: #64748b;
    line-height: 1.3;
}

.hero-btn {
    padding: 0.35rem 1.1rem;   /* sleek, ultra-minimal button footprint */
    border-radius: 999px;
    border: 1px solid rgba(59,130,246,0.6);
    background: linear-gradient(135deg,#3b82f6,#06b6d4);
    color: #020617;
    font-weight: 700;
    font-size: 0.82rem;
    cursor: pointer;
    transition: all 0.2s ease;
}

.hero-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(59,130,246,0.3);
}

.preview-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; margin-top: 0.8rem; }
.preview-card { background: var(--panel); border: 1px solid rgba(255,255,255,0.05); border-radius: 14px; padding: 1rem; box-shadow: 0 14px 38px rgba(0,0,0,0.4); position: relative; overflow: hidden; transition: transform 0.2s ease, border-color 0.2s ease; }
.preview-card:hover { transform: translateY(-3px); border-color: rgba(59,130,246,0.45); }
.preview-card h4 { color: var(--text); margin: 0 0 0.35rem 0; font-weight: 700; letter-spacing: -0.01em; }
.preview-chip { display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.3rem 0.55rem; border-radius: 10px; background: rgba(59,130,246,0.15); color: var(--text); font-size: 0.85rem; }

.skeleton { position: relative; overflow: hidden; background: linear-gradient(90deg, rgba(255,255,255,0.05), rgba(255,255,255,0.12), rgba(255,255,255,0.05)); background-size: 200% 100%; animation: shimmer 1.6s infinite; border-radius: 12px; min-height: 120px; }

.station-modal { position: relative; background: #0b1220; border: 1px solid rgba(148, 163, 184, 0.22); border-radius: 16px; padding: 1rem 1.2rem; box-shadow: 0 18px 50px rgba(0,0,0,0.55); margin: 1rem 0; }
.station-modal h4 { margin: 0 0 0.4rem 0; color: #e2e8f0; font-weight: 750; letter-spacing: -0.01em; }
.station-modal p { margin: 0.15rem 0; color: #cbd5e1; }
.station-modal .actions { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.5rem; margin-top: 0.75rem; }

.bevl-footer { background: #0a0f1f; border-top: 1px solid rgba(148, 163, 184, 0.12); padding: 1rem 2rem 0.75rem; margin-top: 2rem; color: #64748b; text-align: center; font-size: 0.8rem; }
.bevl-footer a { color: #94a3b8; text-decoration: underline; }
.bevl-footer a:hover { color: #e2e8f0; }

hr.page-separator { border: 0; height: 1px; background: rgba(148, 163, 184, 0.25); margin: 1.6rem 0; }

@keyframes shimmer { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }
@keyframes meshShift { 0% { transform: translateY(0); } 100% { transform: translateY(-12px) scale(1.02); } }
@keyframes floatY { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }
@keyframes pulseGlow { 0%, 100% { box-shadow: 0 0 0 0 rgba(59,130,246,0.45); } 50% { box-shadow: 0 0 0 12px rgba(59,130,246,0); } }
@keyframes spinSlow { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
@keyframes rain { 0% { transform: translateY(-10px); opacity: 0; } 20% { opacity: 1; } 100% { transform: translateY(22px); opacity: 0; } }

@media (max-width: 980px) {
    .app-header { grid-template-columns: 1fr; text-align: center; }
    .header-left, .header-right { justify-content: center; }
    .hero-content { grid-template-columns: 1fr; }
}
</style>
'''
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

SECONDARY_CSS = r'''
<style>
:root {
    --hero-title-size: clamp(1.6rem, 4.2vw, 2.4rem);
    --hero-subtitle-size: clamp(1rem, 2.6vw, 1.15rem);
}

.map-wrapper .js-plotly-plot { width: 100% !important; }
.section-gap { height: 12px; }
.section-gap-lg { height: 24px; }
.section-gap-xl { height: 32px; }
.line-row { display: flex; gap: 12px; align-items: center; }
.flat-bar { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 10px 14px; }
.chip-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; }
.chip-row .stButton>button { width: 100%; text-align: center; height: 36px; }
.nav-band { margin: 16px 0 16px 0; }
.map-wrapper { margin-top: 16px; }
.hairline { height: 1px; background: rgba(255,255,255,0.08); margin: 12px 0; }
.tab-guard { margin: 26px 0; }
.stAlert { background: rgba(17,24,39,0.85); border: 1px solid rgba(255,255,255,0.08); color: #d5dbe7; }

.landing-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }
.highlight-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07); border-radius: 14px; padding: 0.95rem; box-shadow: 0 14px 36px rgba(0,0,0,0.35); }
.recent-locations { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
.recent-card { background: rgba(17,24,39,0.82); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 0.8rem; box-shadow: 0 12px 32px rgba(0,0,0,0.32); }

@media (max-width: 768px) {
    .clima-alert { font-size: 0.92rem; padding: 0.75rem 0.9rem; }
    .station-info { padding: 0.9rem; }
    .station-info-title { font-size: 1.15rem; }
    .map-wrapper .js-plotly-plot { min-height: 440px !important; }
    .header-center .title { font-size: 1.5rem; }
}

/* Custom Sidebar Navigation Menu Styling (mimics clima.cbe) */
[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label {
    margin-bottom: 0.15rem !important; 
    padding: 0.35rem 0.6rem !important; 
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
    opacity: 0.85;
    display: block !important;
    width: 100% !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
    background-color: rgba(255, 255, 255, 0.08) !important;
    opacity: 1.0 !important;
    border-left: 3px solid #3b82f6 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) p {
    font-weight: 600 !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:hover:has(input:not(:checked)) {
    background-color: rgba(255, 255, 255, 0.04) !important;
    opacity: 1.0 !important;
}
</style>
'''
st.markdown(SECONDARY_CSS, unsafe_allow_html=True)

CLIMATE_INTELLIGENCE_CSS = r'''
<style>
:root {
    --ci-sidebar-width: 248px;
    --ci-bg: #070d12;
    --ci-bg-elevated: #0b141b;
    --ci-panel: #101a22;
    --ci-panel-2: #13212b;
    --ci-border: rgba(154, 174, 186, 0.16);
    --ci-border-strong: rgba(72, 187, 177, 0.46);
    --ci-text: #edf3f5;
    --ci-muted: #9aabba;
    --ci-subtle: #6f8290;
    --ci-primary: #48bbb1;
    --ci-primary-dark: #123a3b;
    --ci-amber: #e9a23b;
    --ci-coral: #d95d39;
    --ci-blue: #5ba4f3;
    --ci-shadow: 0 18px 48px rgba(0, 0, 0, 0.34);
}

body::before,
body::after {
    display: none !important;
}

html,
body,
.main,
.block-container,
.main > .block-container {
    background: var(--ci-bg) !important;
    color: var(--ci-muted) !important;
}

[data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] style) {
    display: none !important;
}

[data-testid="stElementContainer"]:has(#station-picker) {
    display: none !important;
}

.block-container {
    max-width: 1520px;
    padding: 0.85rem 1.5rem 2.5rem !important;
}

h1, h2, h3, h4 {
    color: var(--ci-text) !important;
    letter-spacing: 0 !important;
}

h1 {
    font-size: 2rem !important;
    line-height: 1.12 !important;
}

h2 {
    font-size: 1.45rem !important;
}

h3 {
    font-size: 1.12rem !important;
}

section[data-testid="stSidebar"][aria-expanded="true"],
section[data-testid="stSidebar"][aria-expanded="true"] > div,
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div:first-child {
    width: var(--ci-sidebar-width) !important;
    min-width: var(--ci-sidebar-width) !important;
    max-width: var(--ci-sidebar-width) !important;
    background: #0b141b !important;
    border-right: 1px solid rgba(154, 174, 186, 0.12) !important;
    box-shadow: 8px 0 28px rgba(0, 0, 0, 0.24) !important;
}

section[data-testid="stSidebar"] {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    height: 100vh !important;
    align-self: flex-start !important;
    overflow: hidden !important;
    z-index: 999 !important;
}

section[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebar"] > div {
    height: 100vh !important;
    overflow: hidden !important;
    scrollbar-width: none;
}

[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"] {
    padding: 1rem 0.9rem 1rem !important;
}

[data-testid="stSidebarHeader"] {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
}

@media (min-width: 769px) {
    [data-testid="stMain"],
    [data-testid="stAppViewContainer"] > .main,
    [data-testid="stAppViewContainer"] > section.main,
    div[data-testid="stAppViewContainer"] section.main {
        margin-left: var(--ci-sidebar-width) !important;
        width: calc(100% - var(--ci-sidebar-width)) !important;
        max-width: calc(100% - var(--ci-sidebar-width)) !important;
    }
}

[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[title="Close sidebar"],
button[title="Open sidebar"],
button[title*="sidebar" i],
button[aria-label="Close sidebar"],
button[aria-label="Open sidebar"],
button[aria-label*="sidebar" i] {
    display: none !important;
}

section[data-testid="stSidebar"][aria-expanded="false"],
section[data-testid="stSidebar"][aria-expanded="false"] > div,
section[data-testid="stSidebar"][aria-expanded="false"] > div:first-child,
section[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarContent"] {
    width: var(--ci-sidebar-width) !important;
    min-width: var(--ci-sidebar-width) !important;
    max-width: var(--ci-sidebar-width) !important;
    transform: translateX(0) !important;
    margin-left: 0 !important;
    left: 0 !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
}

section[data-testid="stSidebar"][aria-expanded="false"] *,
section[data-testid="stSidebar"][aria-expanded="false"] *::before,
section[data-testid="stSidebar"][aria-expanded="false"] *::after {
    transform: translateX(0) !important;
    visibility: visible !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.15rem !important;
}

/* Hide sidebar scrollbar */
[data-testid="stSidebar"] ::-webkit-scrollbar {
    display: none !important;
}

.cc-sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin: 0 0 0.35rem;
    padding: 0.2rem 0.1rem;
}

.cc-sidebar-mark {
    width: 30px;
    height: 30px;
    border-radius: 6px;
    display: grid;
    place-items: center;
    background: rgba(72, 187, 177, 0.14);
    border: 1px solid rgba(72, 187, 177, 0.32);
    color: var(--ci-text);
    font-weight: 800;
    font-size: 0.62rem;
    flex-shrink: 0;
}

.cc-sidebar-title {
    color: var(--ci-text);
    font-size: 0.85rem;
    font-weight: 700;
    line-height: 1.15;
}

.cc-sidebar-subtitle {
    color: var(--ci-muted);
    font-size: 0.68rem;
    margin-top: 0.08rem;
    max-width: 155px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.cc-sidebar-status {
    display: grid;
    grid-template-columns: 7px 1fr;
    gap: 0.4rem;
    align-items: center;
    padding: 0 0.1rem 0.35rem;
    margin-bottom: 0.4rem;
    border-bottom: 1px solid rgba(154, 174, 186, 0.10);
}

.cc-sidebar-status strong {
    display: block;
    color: var(--ci-text);
    font-size: 0.68rem;
    line-height: 1.15;
}

.cc-sidebar-status span:not(.cc-status-dot) {
    display: block;
    color: var(--ci-subtle);
    font-size: 0.6rem;
    line-height: 1.25;
    margin-top: 0.05rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.cc-status-dot {
    width: 6px;
    height: 6px;
    border-radius: 999px;
    background: #5a6875;
}

.cc-status-dot.is-ready {
    background: var(--ci-primary);
    box-shadow: 0 0 0 2px rgba(72, 187, 177, 0.12);
}

/* ── Sidebar nav buttons: flat Clima-CBE style ── */
[data-testid="stSidebar"] .stButton > button {
    min-height: 33px !important;
    height: auto !important;
    justify-content: flex-start !important;
    border-radius: 6px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.38rem 0.6rem !important;
    margin-bottom: 1px !important;
    box-shadow: none !important;
    transition: background 0.15s ease, color 0.15s ease !important;
}

/* Inactive (secondary) nav items: fully flat, no border, no bg */
section[data-testid="stSidebar"] .stButton button,
[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-secondary"],
section[data-testid="stSidebar"] .stButton button[data-testid*="secondary"] {
    color: var(--ci-muted, #9aabba) !important;
    background: transparent !important;
    border: none !important;
    border-color: transparent !important;
    box-shadow: none !important;
}

/* Inactive hover: very subtle tint, no shift */
[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-secondary"]:hover,
section[data-testid="stSidebar"] .stButton button[data-testid*="secondary"]:hover {
    color: var(--ci-text, #edf3f5) !important;
    background: rgba(255, 255, 255, 0.045) !important;
    border: none !important;
    transform: none !important;
}

/* Active (primary) nav item: subtle filled bg + thin left accent */
[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"],
section[data-testid="stSidebar"] .stButton button[kind="primary"],
section[data-testid="stSidebar"] .stButton button[data-testid="baseButton-primary"],
section[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"],
section[data-testid="stSidebar"] .stButton button[data-testid*="primary"] {
    color: var(--ci-text, #edf3f5) !important;
    background: rgba(72, 187, 177, 0.10) !important;
    border: none !important;
    border-left: 3px solid var(--ci-primary, #48bbb1) !important;
    box-shadow: none !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] .stButton button:disabled,
section[data-testid="stSidebar"] .stButton button[disabled] {
    color: rgba(154, 174, 186, 0.42) !important;
    background: transparent !important;
    border-color: transparent !important;
    cursor: not-allowed !important;
}

.cc-sidebar-hint {
    margin: 0.7rem 0 0.45rem;
    padding: 0.6rem 0.65rem;
    border-radius: 7px;
    background: rgba(72, 187, 177, 0.075);
    border: 1px solid rgba(72, 187, 177, 0.16);
    color: #b8c8cf;
    font-size: 0.72rem;
    line-height: 1.42;
}

[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.035) !important;
    border: 1px solid rgba(154, 174, 186, 0.14) !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}

.app-header {
    background: rgba(8, 16, 23, 0.96) !important;
    border-bottom: 1px solid rgba(154, 174, 186, 0.12) !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.24) !important;
    margin: -0.85rem -1.5rem 0.75rem !important;
    padding: 1rem 1.5rem !important;
}

.header-logo {
    max-height: 58px !important;
}

.header-title {
    font-size: 1.28rem !important;
}

.header-location {
    font-size: 0.86rem !important;
}

.header-ub {
    max-height: 50px !important;
}

.cc-station-intro {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 1.5rem;
    padding: 0.2rem 0 0.65rem;
    border-bottom: 1px solid rgba(154, 174, 186, 0.12);
    margin-bottom: 0.65rem;
}

.cc-station-intro h1 {
    margin: 0 !important;
    font-size: 1.55rem !important;
}

.cc-station-intro p:not(.cc-eyebrow) {
    margin: 0.35rem 0 0;
    max-width: 720px;
    color: var(--ci-muted);
    font-size: 0.92rem;
    line-height: 1.45;
}

.cc-map-heading {
    display: flex;
    align-items: end;
    justify-content: space-between;
    gap: 1rem;
    margin: 0.35rem 0 0.35rem;
}

.cc-map-heading h3 {
    margin: 0 !important;
    font-size: 1.02rem !important;
}

.cc-map-heading p {
    margin: 0.2rem 0 0;
    color: var(--ci-muted);
    font-size: 0.84rem;
}

div[data-testid="stFileUploader"] {
    margin-bottom: 0.25rem;
}

.st-key-main_epw_upload_primary {
    margin-bottom: 0 !important;
}

div[data-testid="stFileUploader"] section {
    background: rgba(11, 20, 27, 0.88) !important;
    border: 1px dashed rgba(72, 187, 177, 0.28) !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}

.cc-hero-panel,
.cc-page-intro,
.cc-panel,
.cc-mini-card,
.cc-export-note {
    background: linear-gradient(180deg, rgba(16, 26, 34, 0.98), rgba(11, 20, 27, 0.98));
    border: 1px solid var(--ci-border);
    border-radius: 8px;
    box-shadow: var(--ci-shadow);
}

.cc-hero-panel {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(240px, 360px);
    gap: 1.5rem;
    align-items: end;
    padding: 1.35rem 1.5rem;
    margin-bottom: 1rem;
}

.cc-page-intro {
    padding: 1.15rem 1.25rem;
    margin-bottom: 1rem;
}

.cc-eyebrow {
    margin: 0 0 0.35rem;
    color: var(--ci-primary);
    font-size: 0.76rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.cc-hero-panel h1,
.cc-page-intro h1 {
    margin: 0 !important;
}

.cc-hero-copy,
.cc-page-intro p:not(.cc-eyebrow) {
    max-width: 780px;
    margin: 0.55rem 0 0;
    color: var(--ci-muted);
    font-size: 0.98rem;
    line-height: 1.55;
}

.cc-hero-meta {
    display: grid;
    gap: 0.45rem;
}

.cc-hero-meta span {
    display: block;
    padding: 0.55rem 0.65rem;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(154, 174, 186, 0.12);
    color: #cbd8df;
    font-size: 0.84rem;
}

.cc-panel {
    padding: 1rem;
    margin-bottom: 1rem;
}

.cc-panel-head h3 {
    margin: 0 !important;
    color: var(--ci-text) !important;
    font-size: 1.05rem !important;
}

.cc-panel-head p {
    margin: 0.35rem 0 0;
    color: var(--ci-muted);
    font-size: 0.88rem;
    line-height: 1.45;
}

.cc-summary-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.75rem;
    margin-top: 1rem;
}

.cc-summary-grid div,
.cc-mini-card,
.cc-export-note {
    padding: 0.78rem;
}

.cc-summary-grid span,
.cc-mini-card span,
.cc-export-note span {
    display: block;
    color: var(--ci-subtle);
    font-size: 0.78rem;
    line-height: 1.4;
}

.cc-summary-grid strong,
.cc-mini-card strong,
.cc-export-note strong {
    display: block;
    color: var(--ci-text);
    font-size: 0.95rem;
    margin-top: 0.2rem;
}

div[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(16, 26, 34, 0.98), rgba(11, 20, 27, 0.98));
    border: 1px solid var(--ci-border);
    border-radius: 8px;
    padding: 0.9rem 0.95rem;
    box-shadow: 0 14px 32px rgba(0, 0, 0, 0.22);
}

[data-testid="stMetricLabel"] {
    color: var(--ci-subtle) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em !important;
}

[data-testid="stMetricValue"] {
    color: var(--ci-text) !important;
    font-size: 1.35rem !important;
}

div[role="radiogroup"][aria-label="Dashboard view"] {
    display: flex !important;
    flex-wrap: wrap;
    gap: 0.5rem !important;
    padding: 0.45rem !important;
    margin-bottom: 1rem;
    background: rgba(11, 20, 27, 0.8);
    border: 1px solid var(--ci-border);
    border-radius: 8px;
}

div[role="radiogroup"][aria-label="Dashboard view"] label {
    min-height: 38px;
    padding: 0.48rem 0.75rem !important;
    border: 1px solid transparent;
    border-radius: 7px;
    color: var(--ci-muted);
    font-weight: 650;
}

div[role="radiogroup"][aria-label="Dashboard view"] label > div:first-child {
    display: none !important;
}

div[role="radiogroup"][aria-label="Dashboard view"] label:has(input:checked) {
    color: var(--ci-text) !important;
    background: rgba(72, 187, 177, 0.16) !important;
    border-color: rgba(72, 187, 177, 0.4);
    box-shadow: inset 0 -2px 0 var(--ci-primary);
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(11, 20, 27, 0.8) !important;
    border: 1px solid var(--ci-border) !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    padding: 0.45rem !important;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important;
    color: var(--ci-muted) !important;
    font-weight: 650 !important;
}

.stTabs [aria-selected="true"] {
    background: rgba(72, 187, 177, 0.16) !important;
    color: var(--ci-text) !important;
    box-shadow: inset 0 -2px 0 var(--ci-primary) !important;
}

.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 1rem 0 0 !important;
    overflow: visible !important;
    min-height: 420px;
}

.stTabs [data-baseweb="tab-panel"] > div,
.stTabs > div:nth-of-type(2) > div > div,
[role="tabpanel"] {
    overflow: visible !important;
}

.js-plotly-plot {
    border-radius: 8px !important;
    border: 1px solid rgba(154, 174, 186, 0.12);
    box-shadow: 0 18px 48px rgba(0,0,0,0.28) !important;
    overflow: hidden;
}

[data-testid="stPlotlyChart"],
[data-testid="stPlotlyChart"] > div,
[data-testid="stPlotlyChart"] .js-plotly-plot,
[data-testid="stPlotlyChart"] .plot-container,
[data-testid="stPlotlyChart"] .svg-container {
    min-height: 420px !important;
}

[data-testid="stPlotlyChart"] svg.main-svg {
    min-height: 420px !important;
}

.stButton button,
.stDownloadButton button {
    border-radius: 8px !important;
    min-height: 38px !important;
    font-weight: 700 !important;
    box-shadow: none !important;
}

.stDownloadButton button,
.stButton button[data-testid="baseButton-primary"] {
    background: linear-gradient(90deg, #48bbb1, #5ba4f3) !important;
    color: #071017 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

.stButton button[data-testid="baseButton-secondary"] {
    background: rgba(255,255,255,0.055) !important;
    color: var(--ci-text) !important;
    border: 1px solid rgba(154, 174, 186, 0.14) !important;
}

.cc-export-note {
    margin-top: 1rem;
}

.cc-pdf-capture-screen {
    background: linear-gradient(180deg, rgba(16, 26, 34, 0.98), rgba(11, 20, 27, 0.98));
    border: 1px solid var(--ci-border);
    border-radius: 8px;
    box-shadow: var(--ci-shadow);
    padding: 1.25rem 1.35rem;
    margin-bottom: 1rem;
}

.cc-pdf-capture-screen h1 {
    margin: 0 !important;
}

.cc-pdf-capture-screen p:not(.cc-eyebrow) {
    margin: 0.55rem 0 0;
    color: var(--ci-muted);
}

@media (max-height: 760px) {
    .cc-sidebar-status,
    .sidebar-brand {
        display: none !important;
    }

    [data-testid="stSidebar"] .stButton > button {
        min-height: 36px !important;
        font-size: 0.9rem !important;
        padding-top: 0.42rem !important;
        padding-bottom: 0.42rem !important;
    }
}

@media (max-width: 1180px) {
    .cc-hero-panel {
        grid-template-columns: 1fr;
    }
    .cc-summary-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .cc-station-intro,
    .cc-map-heading {
        align-items: flex-start;
        flex-direction: column;
    }
}
</style>
'''
st.markdown(CLIMATE_INTELLIGENCE_CSS, unsafe_allow_html=True)


def _encode_image_to_base64(path: Union[str, Path]) -> str:
    """Embed local assets (e.g., logos) as base64 data URIs for consistent rendering."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


def _load_logos() -> tuple[str, str]:
    """Load and base64-encode logos once per process lifetime."""
    base = Path(__file__).parent
    primary   = _encode_image_to_base64(base / 'assets' / 'bevl_framework.png')
    secondary = _encode_image_to_base64(base / 'assets' / 'ub_framework.png')
    return primary, secondary

LOGO_PRIMARY, LOGO_SECONDARY = _load_logos()



def _is_valid_label_value(val: object) -> bool:
    """Return True if val is a non-empty, non-sentinel string."""
    if not val or not isinstance(val, str):
        return False
    return val.strip().lower() not in ("", "none", "null", "undefined", "nan", "n/a")


def get_location_label(default: str = "Location: N/A") -> str:
    """Robustly retrieve a 'City, Country' string from session state header."""
    header = st.session_state.get("header")
    if not isinstance(header, dict):
        return default

    loc = header.get("location") or {}
    city = loc.get("city") or loc.get("cityName") or header.get("city") or header.get("cityName")
    country = loc.get("country") or header.get("country") or header.get("countryName")

    city = city if _is_valid_label_value(city) else None
    country = country if _is_valid_label_value(country) else None

    if city or country:
        parts = [str(p).strip() for p in (city, country) if p]
        return ", ".join(parts)
    return default


def get_clean_city_name() -> str:
    """Retrieve just the city name for graph headers (e.g. 'San Francisco')."""
    ss = st.session_state
    
    # 1. Try selected station dict (most explicit)
    sel = ss.get("selected_station") or ss.get("sel_station")
    if isinstance(sel, dict):
        city = sel.get("city_name") or sel.get("cityname") or sel.get("city")
        if city: 
            return str(city).strip()
        
        # Fallback: parsing standardized name format "USA_CA_San.Francisco..."
        name = sel.get("name", "")
        if name and "_" in name:
            parts = name.split("_")
            if len(parts) >= 3:
                city_part = parts[2].split(".")[0]
                if city_part:
                    return city_part.replace("-", " ").strip()

    # 2. Try header metadata (EPW)
    header = ss.get("header", {})
    if isinstance(header, dict):
        loc = header.get("location", {})
        city = loc.get("city") or loc.get("cityName") or header.get("city")
        if city: 
            return str(city).strip()

    # 3. Fallback to extracting from formatted label
    full_label = get_location_label("Unknown Location")
    if "," in full_label:
        return full_label.split(",")[0].strip()
    
    return full_label


def render_header():
    logo_src = f"data:image/png;base64,{LOGO_PRIMARY}" if LOGO_PRIMARY else ""
    ub_src = f"data:image/png;base64,{LOGO_SECONDARY}" if LOGO_SECONDARY else ""
    
    loc_label = get_location_label()

    st.markdown(
        f"""
        <div class="app-header">
          <div class="header-left">
            <img class="header-logo" src="{logo_src}" alt="BEVL Lab" />
            <div class="header-text">
              <div class="header-title">Climate Analysis Pro</div>
              <div class="header-location">{loc_label}</div>
            </div>
          </div>
          <div class="header-right">
            <img class="header-ub" src="{ub_src}" alt="UB Framework" />
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



# ========== SIDEBAR WITH IMPROVED UX ==========
def render_landing_hero():
    st.markdown(
        """
        <section class="cc-station-intro">
          <div>
            <p class="cc-eyebrow">Weather source</p>
            <h1>Select weather file</h1>
            <p>Choose a station or upload an EPW/ZIP file to begin the analysis workspace.</p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )





def render_sidebar_filters(epw_loaded: bool) -> None:
    with st.expander("Settings & Filters", expanded=False):
        st.caption("Refine the analysis sandbox. Settings persist for this session.")
        temp_unit = st.radio(
            "Temperature units",
            options=["C", "F"],
            index=0 if _temp_unit() == "C" else 1,
            format_func=lambda u: "Celsius (°C)" if u == "C" else "Fahrenheit (°F)",
            key="temperature_unit",
        )
        current_threshold_c = float(st.session_state.get("custom_overheat_threshold", 30))
        unit_label = "°F" if temp_unit == "F" else "°C"
        if temp_unit == "F":
            threshold_slider = st.slider(
                f"Focus comfort threshold ({unit_label})",
                min_value=int(round(_c_to_f(24))),
                max_value=int(round(_c_to_f(36))),
                value=int(round(_c_to_f(current_threshold_c))),
                step=1,
                help="Adds this threshold across comfort analytics.",
            )
            threshold_c = _f_to_c(threshold_slider)
        else:
            threshold_slider = st.slider(
                f"Focus comfort threshold ({unit_label})",
                min_value=24,
                max_value=36,
                value=int(round(current_threshold_c)),
                step=1,
                help="Adds this threshold across comfort analytics.",
            )
            threshold_c = float(threshold_slider)
        st.session_state["custom_overheat_threshold"] = float(threshold_c)

        _phase_a_busy = bool(st.session_state.get("is_loading") or st.session_state.get("load_requested"))
        if not _phase_a_busy:
            st.caption("All temperature-derived charts now reflect this additional heat load until you toggle it off.")

        model_options = [
            "Auto SARIMAX (default)",
            "Persistence (naïve)",
            "Seasonal ETS (preview)",
        ]
        default_model = st.session_state.get("forecast_model_choice") or model_options[0]
        if default_model not in model_options:
            default_model = model_options[0]
        st.selectbox(
            "Forecast model",
            options=model_options,
            index=model_options.index(default_model),
            key="forecast_model_choice",
            help="Experiment with different short-term models. Non-default entries currently fall back to SARIMAX but make the intent explicit."
        )

        st.slider(
            "Month range",
            min_value=1,
            max_value=12,
            value=st.session_state.get("month_range", (1, 12)),
            step=1,
            disabled=not epw_loaded,
            key="month_range",
            help="Limit visualizations to a month window when data is loaded.",
        )

    base_cdf = st.session_state.get("cdf_raw")
    if base_cdf is not None:
        cdf_adjusted = base_cdf.copy(deep=True)
        
        # 1. Apply UHI Bias
        if st.session_state.get("apply_uhi_bias") and "drybulb" in cdf_adjusted.columns:
            delta = float(st.session_state.get("uhi_bias_delta", 1.5))
            cdf_adjusted["drybulb"] = cdf_adjusted["drybulb"] + delta
            
        # 2. Apply Month Range Filter (if valid datetime index)
        m_range = st.session_state.get("month_range", (1, 12))
        if m_range != (1, 12):
            # Apply month filter — works for both tz-aware and tz-naive indexes
            mask = ((cdf_adjusted.index.month >= m_range[0])
                    & (cdf_adjusted.index.month <= m_range[1]))
            cdf_adjusted = cdf_adjusted[mask]

                
        # Only recompute comfort if the adjusted cdf actually changed.
        # Use a hash of the relevant parameters as a cache key.
        _pkg_key = f"{st.session_state.get('apply_uhi_bias')}_{st.session_state.get('uhi_bias_delta')}_{st.session_state.get('month_range')}_{st.session_state.get('last_loaded_station_id')}"
        if st.session_state.get('_comfort_pkg_key') != _pkg_key:
            st.session_state.comfort_pkg = build_comfort_package(cdf_adjusted)
            st.session_state['_comfort_pkg_key'] = _pkg_key








def render_sidebar():
    """Premium sidebar navigation for the climate intelligence workflow."""
    with st.sidebar:
        epw_loaded = bool(st.session_state.get("cdf") is not None and st.session_state.get("header"))
        current_page = st.session_state.get("nav_page", DEFAULT_PAGE)
        if current_page not in ALLOWED_PAGES:
            current_page = DEFAULT_PAGE
        if not epw_loaded and current_page != DEFAULT_PAGE:
            current_page = DEFAULT_PAGE
        st.session_state["nav_page"] = current_page

        header = st.session_state.get("header", {})
        loc_meta = header.get("location", {}) if isinstance(header, dict) else {}
        loc_city = (
            (header.get("city") if isinstance(header, dict) else None)
            or loc_meta.get("city")
            or loc_meta.get("state_province")
            or ""
        )
        loc_country = (
            (header.get("country") if isinstance(header, dict) else None)
            or loc_meta.get("country")
            or ""
        )
        loc_label = ", ".join([p for p in [loc_city, loc_country] if p]) or "Weather Workspace"
        status_label = "EPW loaded" if epw_loaded else "No file loaded"
        source_label = st.session_state.get("source_label") or "Select a station or upload EPW"

        st.markdown(
            f"""
            <div class="cc-sidebar-brand">
                <div class="cc-sidebar-mark">CI</div>
                <div>
                    <div class="cc-sidebar-title">Climate Intelligence</div>
                    <div class="cc-sidebar-subtitle">{loc_label}</div>
                </div>
            </div>
            <div class="cc-sidebar-status">
                <span class="cc-status-dot {'is-ready' if epw_loaded else ''}"></span>
                <div>
                    <strong>{status_label}</strong>
                    <span>{source_label}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if current_page == "Dashboard" and st.session_state.get("pdf_dashboard_autobuild_pending"):
            st.info("Building the full report. You will return to Export automatically.")

        for label, page in NAV_ITEMS:
            disabled = (not epw_loaded and page != DEFAULT_PAGE)
            is_active = current_page == page
            button_type = "primary" if is_active else "secondary"
            key_slug = re.sub(r"[^a-zA-Z0-9_]+", "_", f"nav_{page}").strip("_").lower()
            if st.button(
                label,
                key=key_slug,
                type=button_type,
                use_container_width=True,
                disabled=disabled,
            ):
                st.session_state["nav_page"] = page
                st.session_state["active_page"] = "select_station" if page == DEFAULT_PAGE else "dashboard"
                _rerun()

        if not epw_loaded:
            st.markdown(
                "<div class='cc-sidebar-hint'>Load a station or upload an EPW/ZIP to unlock the workspace.</div>",
                unsafe_allow_html=True,
            )

        st.session_state["active_page"] = "select_station" if current_page == DEFAULT_PAGE else "dashboard"

        render_sidebar_filters(epw_loaded)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Reset Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            _rerun()

        st.markdown(
            """
            <div class="sidebar-brand">
                <strong>BEVL Lab</strong><br/>
                Weather intelligence for research and practice
            </div>
            """,
            unsafe_allow_html=True,
        )


# ========== MAIN CONTENT ==========

if st.session_state.get("active_page") == "select_station" and st.session_state.get("loading_station_name"):
    st.status(f"Loading data for {st.session_state['loading_station_name']}…", expanded=False)

if st.session_state.get("apply_uhi_bias") and st.session_state.get("cdf") is not None:
    st.info(
        f"Urban heat island bias of {format_temperature_delta(st.session_state.get('uhi_bias_delta', 1.5))} is currently applied to all temperature-dependent charts."
    )

ss = st.session_state
raw_epw_bytes = ss.get("raw_epw_bytes")
source_label = ss.get("source_label")

def _stage_station_and_load(station_info: dict):
    """Stage station selection; controller performs the one-shot download+parse."""
    if st.session_state.get("pdf_dashboard_autobuild_pending", False):
        return
    if st.session_state.get("nav_page") != "Select weather file":
        return

    station_id = station_info.get("station_id") or station_info.get("raw_id") or station_info.get("name")
    if st.session_state.get("is_loading"):
        return
    if station_id and station_id == st.session_state.get("pending_station_id"):
        return
    if station_id and station_id == st.session_state.get("last_loaded_station_id"):
        return

    st.session_state["loading_station_name"] = station_info.get("name", "selected station")
    st.session_state["sel_station"] = station_info
    st.session_state["selected_station"] = station_info
    zip_url = station_info.get("zip_url", "")
    if zip_url and ('<a ' in str(zip_url) and 'href' in str(zip_url)):
        zip_url = _extract_url(zip_url) or zip_url
    st.session_state["sel_station_url"] = zip_url
    display_label = station_info.get("display_label") or station_info.get("name") or "EPW"
    st.session_state["source_label"] = f"Station: {display_label}"
    st.session_state["pending_station_id"] = station_id
    st.session_state["pending_station"] = station_info
    st.session_state["load_requested"] = True
    st.session_state["page_after_station"] = "Overview"
    st.session_state.pop("raw_epw_bytes", None)
    _rerun()


def _station_df_signature(stations: pd.DataFrame) -> str:
    cols = [c for c in ["lat", "lon", "name", "country", "period"] if c in stations]
    if not cols:
        return ""
    subset = stations[cols].copy()
    for c in cols:
        subset[c] = subset[c].fillna("")
    import hashlib as _hl
    return _hl.sha256(subset.to_csv(index=False).encode()).hexdigest()


def _build_station_map_figure(stations: pd.DataFrame, map_height: int) -> go.Figure:
    sig = _station_df_signature(stations)
    cached_sig = st.session_state.get("_station_map_sig")
    cached_fig = st.session_state.get("_station_map_fig")
    if cached_sig == sig and cached_fig is not None:
        return cached_fig

    hover_text = stations["name"].fillna("").astype(str).tolist()
    center_lat = float(stations["lat"].median())
    center_lon = float(stations["lon"].median())

    fig_map = go.Figure(
        data=[
            go.Scattermapbox(
                lat=stations["lat"].tolist(),
                lon=stations["lon"].tolist(),
                mode="markers",
                marker=dict(size=10, color="#5fd4ff", opacity=0.82),
                hovertext=hover_text,
                hovertemplate="%{hovertext}<extra></extra>",
                name="",
                showlegend=False,
            )
        ]
    )

    fig_map.update_layout(
        mapbox=dict(
            style="open-street-map",
            bearing=0,
            pitch=0,
            center=dict(lat=center_lat, lon=center_lon),
            zoom=2.2,
        ),
        dragmode="pan",
        margin=dict(l=0, r=0, t=12, b=0),
        height=map_height,
        paper_bgcolor="#0b0f1a",
        plot_bgcolor="#0b0f1a",
        hovermode="closest",
        clickmode="event+select",
        showlegend=False,
        uirevision="station-map",
    )

    st.session_state["_station_map_sig"] = sig
    st.session_state["_station_map_fig"] = fig_map
    return fig_map



def _interactive_map_fragment() -> None:
    # Guard: ensure Streamlit session context is fully initialized.
    if not _session_is_ready():
        return
    try:
        _ = st.session_state  # lightweight probe
    except Exception:
        return

    stations = st.session_state.get("_stations_picker_df")
    if not isinstance(stations, pd.DataFrame) or stations.empty:
        st.info("Station map is not available right now.")
        return

    st.markdown(
        """
        <div class="cc-map-heading">
          <div>
            <h3>Interactive Map</h3>
            <p>Select a station from the global EPW index.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    map_height = 580
    map_slot = st.empty()
    st.session_state["_map_slot"] = map_slot
    fig_map = _build_station_map_figure(stations, map_height)
    pdf_pending = bool(st.session_state.get("pdf_dashboard_autobuild_pending", False))
    on_select_page = st.session_state.get("nav_page") == "Select weather file"

    # Interactive map when safe; static fallback during PDF generation.
    event = None
    with map_slot.container():
        try:
            if pdf_pending or not on_select_page:
                map_key = "station_map_pdf_pending" if pdf_pending else "station_map_static_only"
                _st_plotly_chart(fig_map, use_container_width=True, key=map_key)
                if pdf_pending:
                    st.caption("Map click-to-load is paused while PDF generation is in progress.")
                return

            event = _st_plotly_chart(
                fig_map,
                use_container_width=True,
                on_select="rerun",
                selection_mode=("points",),
                key="station_map_interactive",
            )
        except Exception:
            _st_plotly_chart(fig_map, use_container_width=True, key="station_map_static")
            st.caption("Map click-to-load is temporarily unavailable. Use Station Search below to load a station.")
            return

    # Extract selected point index from Streamlit event payload.
    selected_points = []
    if event and hasattr(event, "selection") and event.selection:
        sel = event.selection
        if hasattr(sel, "points"):
            selected_points = sel.points
        elif isinstance(sel, dict):
            selected_points = sel.get("points") or []

    if selected_points:
        point = selected_points[0]
        idx = point.get("point_index")
        if idx is None:
            idx = point.get("pointIndex")
        if idx is None and isinstance(point.get("point_indices"), list) and point.get("point_indices"):
            idx = point.get("point_indices")[0]

        if idx is not None and 0 <= int(idx) < len(stations):
            row = stations.iloc[int(idx)]
            station_info = {
                "name": row.get("name", "Unknown"),
                "country": row.get("country", "—"),
                "lat": row.get("lat", 0),
                "lon": row.get("lon", 0),
                "elevation_m": row.get("elevation_m", "—"),
                "timezone": row.get("timezone", "—"),
                "zip_url": row.get("zip_url", ""),
                "period": row.get("period", "—"),
                "heating_db": row.get("heating_db"),
                "cooling_db": row.get("cooling_db"),
                "display_label": row.get("display_label"),
                "station_id": row.get("station_id"),
                "raw_id": row.get("raw_id"),
            }
            station_id = row.get("station_id") or row.get("raw_id") or str(idx)
            last_loaded_id = st.session_state.get("last_loaded_station_id")
            pending_id = st.session_state.get("pending_station_id")
            if station_id not in (last_loaded_id, pending_id):
                _stage_station_and_load(station_info)


def _station_search_fragment() -> None:
    """Micro-rerun fragment for station search + load button."""
    if not _session_is_ready():
        return
    try:
        _ = st.session_state
    except Exception:
        return
    stations = st.session_state.get("_stations_picker_df")
    if not isinstance(stations, pd.DataFrame) or stations.empty:
        return

    st.subheader("Station Search", anchor=False)
    st.caption(f"{len(stations):,} verified download links. Search, then load once.")
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

    search_col, load_col = st.columns([3, 1])
    with search_col:
        search_query = st.text_input(
            "Search by station name, country, or period",
            key="station_selector_query",
            help="Type a city, ISO3 code, WMO station ID, source, or year range. Top 25 matches shown.",
            placeholder="e.g., Paris FRA 2021 TMYx",
        )

        matches = stations.head(0)
        choice_label = None
        search = (search_query or "").strip()

        if search:
            mask = (
                stations["display_label"].str.contains(search, case=False, na=False)
                | stations["city_name"].str.contains(search, case=False, na=False)
                | stations["country_name"].str.contains(search, case=False, na=False)
            )
            matches = stations[mask].head(25)

            if matches.empty:
                st.info("No stations match that search. Try a broader term or different year range.")
            else:
                choice_label = st.selectbox(
                    "Matches",
                    matches["display_label"],
                    key="station_selector",
                )
        else:
            st.caption("Start typing to search stations.")

    with load_col:
        chosen_row = None
        if search and not matches.empty and choice_label:
            chosen_row = matches[matches["display_label"] == choice_label].iloc[0]

        if st.button("Load Selected Station", type="primary", use_container_width=True):
            if chosen_row is not None:
                station_info = {
                    "name": chosen_row.get("name", "Unknown"),
                    "country": chosen_row.get("country", "—"),
                    "lat": chosen_row.get("lat", 0),
                    "lon": chosen_row.get("lon", 0),
                    "elevation_m": chosen_row.get("elevation_m", "—"),
                    "timezone": chosen_row.get("timezone", "—"),
                    "zip_url": chosen_row.get("zip_url", ""),
                    "period": chosen_row.get("period", "—"),
                    "heating_db": chosen_row.get("heating_db"),
                    "cooling_db": chosen_row.get("cooling_db"),
                    "display_label": chosen_row.get("display_label"),
                    "station_id": chosen_row.get("station_id"),
                    "raw_id": chosen_row.get("raw_id"),
                }
                _stage_station_and_load(station_info)


@CACHE(show_spinner=False, ttl=86400)
def _prepare_station_picker_dataframe(stations_in: pd.DataFrame) -> pd.DataFrame:
    """Prepare station dataframe for the picker/map (cached to avoid rerun lag)."""
    stations = stations_in.copy()
    # -------------------------------------------------------------------------
    # Helper: Normalize columns
    # -------------------------------------------------------------------------
    if "lat" not in stations.columns or "lon" not in stations.columns:
        return stations

    # Convert lat/lon
    stations["lat"] = pd.to_numeric(stations["lat"], errors="coerce")
    stations["lon"] = pd.to_numeric(stations["lon"], errors="coerce")
    stations = stations.dropna(subset=["lat", "lon"])
    
    # Helpers for display
    def _country_name_from_iso3(code: str) -> str:
        code = (code or "").strip()
        if not code:
            return ""
        try:
            import pycountry  # type: ignore
            c = pycountry.countries.get(alpha_3=code.upper())
            if c:
                return c.name
        except Exception:
            pass  # pycountry lookup is best-effort
        return code.upper()

    stations["country_disp"] = stations.get("country", pd.Series(dtype=str)).fillna("—")
    stations["elev_disp"] = pd.to_numeric(stations["elevation_m"], errors="coerce").round(0).astype("Int64")
    stations["tz_disp"] = stations["timezone"].astype(str).replace({"nan": "—"})
    stations["period_disp"] = stations.get("period", pd.Series(dtype=str)).fillna("—")
    stations["heating_disp"] = stations["heating_db"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
    stations["cooling_disp"] = stations["cooling_db"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
    stations["source_url"] = stations.get("zip_url", pd.Series(dtype=str)).fillna("")

    # --- VECTORIZED PARSING (Optimized) ---
    # Parse metadata from 'name' column if available (e.g. DZA_Algiers.603900_IWEC)
    if "name" in stations.columns:
        # Regex to capture: ISO, City (raw), WMO, Source (optional)
        # Assumes format: ISO_CityName.WMO_Source
        pattern = r"^(?P<iso_p>[A-Z]{3,})_(?P<city_p>[^.]+)\.(?P<wmo_p>\d+)(?:[._](?P<src_p>.+))?$"
        active_names = stations["name"].astype(str)
        extracted = active_names.str.extract(pattern)
        
        # Fill missing standard columns with parsed values where valid
        if "iso_p" in extracted.columns:
            if "country_iso3" not in stations.columns: stations["country_iso3"] = np.nan
            stations["country_iso3"] = stations["country_iso3"].fillna(extracted["iso_p"])
            
            if "city_name" not in stations.columns: stations["city_name"] = np.nan
            stations["city_name"] = stations["city_name"].fillna(extracted["city_p"])
            
            if "station_id" not in stations.columns: stations["station_id"] = np.nan
            stations["station_id"] = stations["station_id"].fillna(extracted["wmo_p"])
            
            if "source" not in stations.columns: stations["source"] = np.nan
            stations["source"] = stations["source"].fillna(extracted["src_p"])

    # Ensure columns exist and are string
    for col in ["country_name", "state_code", "source", "period", "city_name", "country_iso3", "station_id"]:
        if col not in stations.columns:
            stations[col] = ""
        stations[col] = stations[col].fillna("").astype(str)

    # Clean up City Names: "Algiers" is fine, "Al-Minya" is fine. 
    # Replace dots/underscores only if they look like delimiters we missed (vectorized)
    stations["city_name"] = stations["city_name"].str.replace("_", " ", regex=False).str.title()

    # Vectorized Country Name Lookup
    unique_isos = stations["country_iso3"].unique()
    iso_map = {}
    for iso in unique_isos:
        if iso:
            try:
                iso_map[iso] = _country_name_from_iso3(iso)
            except:
                iso_map[iso] = iso
    
    # Fill country_name using the map where missing
    mask_missing_country = (stations["country_name"] == "") | (stations["country_name"] == "nan")
    stations.loc[mask_missing_country, "country_name"] = stations.loc[mask_missing_country, "country_iso3"].map(iso_map).fillna("")

    # Construct Display Label (Vectorized)
    # Format: City, State, Country (WMO ID, Source, Period)
    
    label_parts = pd.DataFrame({
        "city": stations["city_name"],
        "state": stations["state_code"],
        "country": stations["country_name"]
    })
    # Filter empty strings
    loc_str = label_parts.apply(lambda x: ", ".join(s for s in x if s), axis=1)
    
    # Meta Part
    meta_parts = pd.DataFrame({
        "wmo": stations["station_id"].apply(lambda x: f"WMO {x}" if x else ""),
        "src": stations["source"],
        "prd": stations["period"]
    })
    meta_str = meta_parts.apply(lambda x: ", ".join(s for s in x if s), axis=1)
    
    # Combine
    full_labels = loc_str + " (" + meta_str + ")"
    # Handle missing location (fallback to Raw ID or 'Unknown')
    fallback = stations.get("raw_id", stations.get("name", "Unknown Station"))
    
    stations["display_label"] = np.where(loc_str != "", full_labels, fallback)
    
    # Clean parens if meta is empty
    stations["display_label"] = stations["display_label"].str.replace(" ()", "", regex=False)

    return stations


def render_station_picker():
    # Render station selection with quick picks, dropdown, and map.

    header = st.session_state.get("header", {}) if isinstance(st.session_state.get("header"), dict) else {}
    location_meta = header.get("location", {}) if isinstance(header, dict) else {}

    # (pending_station is managed by controller; do not clear here)

    # ---------- Station list + map ----------
    from pathlib import Path
    import re

    stations_raw = _PRELOADED_STATIONS.copy() if isinstance(_PRELOADED_STATIONS, pd.DataFrame) else load_station_index_for_map()
    stations = _prepare_station_picker_dataframe(stations_raw)
    st.session_state["_stations_picker_df"] = stations

    
    try:
        _interactive_map_fragment()
    except Exception as _frag_err:
        st.warning(f"Map fragment error (try refreshing): {_frag_err}")
    st.divider()
    try:
        _station_search_fragment()
    except Exception as _frag_err:
        st.warning(f"Search fragment error (try refreshing): {_frag_err}")


def handle_epw_upload(uploaded_file, picker_key: str = "sidebar") -> Optional[bytes]:
    """Process an uploaded EPW/ZIP and persist to session state.

    picker_key differentiates selectbox keys between sidebar/main uploads.
    Returns the raw EPW bytes or None if selection failed.
    """
    if uploaded_file is None:
        return None

    try:
        uploaded_file.seek(0)
    except Exception as e:
        st.warning(f"seek failed: {e}")

    filename = getattr(uploaded_file, "name", "uploaded.epw")
    raw_epw_bytes: Optional[bytes] = None

    file_bytes = uploaded_file.read()
    if not file_bytes:
        st.warning("Uploaded file is empty; please try again.")
        return None

    if filename.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(file_bytes), "r") as z:
            epws = [m for m in z.namelist() if m.lower().endswith(".epw")]
            if not epws:
                st.warning("ZIP file contains no EPW files")
                return None
            if len(epws) > 1:
                pick = st.selectbox(
                    "Select EPW inside ZIP",
                    epws,
                    index=0,
                    key=f"{picker_key}_zip_pick",
                    help="Multiple EPW files detected; choose one to analyze."
                )
                raw_epw_bytes = z.read(pick)
            else:
                epws.sort(key=lambda m: z.getinfo(m).file_size, reverse=True)
                raw_epw_bytes = z.read(epws[0])
    else:
        raw_epw_bytes = file_bytes

    st.session_state["raw_epw_bytes"] = raw_epw_bytes
    st.session_state["source_label"] = f"Uploaded: {filename}"
    st.session_state.pop("loading_station_name", None)
    return raw_epw_bytes


# Kick off station preload as soon as the script starts so the map is ready on first paint
_PRELOADED_STATIONS = None

# Strict routing: only one page renders per run.
controller_page = st.session_state.get("active_page", "select_station")

nav_page = st.session_state.get("nav_page", DEFAULT_PAGE)
if nav_page not in ALLOWED_PAGES:
    nav_page = DEFAULT_PAGE

if controller_page != "select_station" or st.session_state.pop("_clear_map_on_next_run", False):
    map_slot = st.session_state.pop("_map_slot", None)
    if map_slot is not None:
        map_slot.empty()
    st.session_state.pop("_station_map_fig", None)
    st.session_state.pop("_station_map_sig", None)

main_upload = None
def render_select_station_page():
    render_landing_hero()
    main_upload = st.file_uploader(
        "Upload EPW or ZIP file",
        type=["epw", "zip"],
        help="Upload an EnergyPlus Weather file or a ZIP containing EPWs",
        key="main_epw_upload_primary",
    )

    # Handle upload immediately (before any st.stop() gating in this branch).
    if main_upload is not None:
        with cloud_loader("Parsing EPW…"):
            raw_epw_bytes = handle_epw_upload(main_upload, picker_key="main")
            if raw_epw_bytes is not None:
                _rerun()

    st.markdown("<div id='station-picker'></div>", unsafe_allow_html=True)
    # render_station_picker handles the map and list
    render_station_picker()

    if st.session_state.get("station_load_error"):
        debug = st.session_state.get("station_load_debug") or {}
        station_name = debug.get("station_name", "selected station")
        url = debug.get("original_url", "")
        urls_to_try = debug.get("urls_to_try", [])
        successful_url = debug.get("successful_url")

        st.error(f"❌ Could not fetch EPW from **{station_name}**. All download attempts failed.")

        troubleshooting_md = (
            "### Troubleshooting Steps\n\n"
            "Immediate solutions:\n"
            "- Try one of the Quick Start verified stations above\n"
            "- Upload your own EPW file using the sidebar\n"
            "- Use the station selector dropdown instead of the map\n\n"
            "Common issues:\n"
            "- The climate data server might be temporarily down\n"
            "- The specific EPW file may have been moved or removed\n"
            "- Your network might be blocking the download\n\n"
            "Advanced options:\n"
            "- Manually download EPW files from EnergyPlus Weather Data (energyplus.net/weather)\n"
            "- Try the OneBuilding.org website directly (climate.onebuilding.org/)\n"
        )
        st.markdown(troubleshooting_md)

        with st.expander("Technical Details & Debug Info"):
            st.write(f"Station: {station_name}")
            st.write(f"Original URL: `{url}`")
            st.write("Alternative URLs tried:")
            for i, test_url in enumerate(urls_to_try):
                status = "Success" if test_url == successful_url else "Failed"
                st.write(f"{i+1}. `{test_url}` — {status}")
            about_error = (
                "About this error:\n"
                "This typically happens when climate data repositories reorganize their file structure. The station data exists, but the specific file path has changed."
            )
            st.markdown(about_error)

        st.session_state.pop("loading_station_name", None)

        st.markdown("### Recovery Options")
        col1, col2 = st.columns(2)
        if col1.button("Clear & Start Over", use_container_width=True):
            ss = st.session_state
            ss.pop("sel_station_url", None)
            ss.pop("sel_station", None)
            ss.pop("pending_station_id", None)
            ss.pop("pending_station", None)
            ss.pop("station_load_error", None)
            ss.pop("station_load_debug", None)
            _rerun()

        if col2.button("Show Working Stations", use_container_width=True):
            st.info(
                "Verified working stations:\n"
                "- Denver, CO (724695)\n"
                "- Chicago, IL (725346)\n"
                "- Phoenix, AZ (722780)\n"
                "- Los Angeles, CA (722950)\n"
                "- Miami, FL (722020)\n"
                "- Seattle, WA (727930)\n"
            )

    # Prevent fall-through rendering (no dashboard/widgets appended below).
    st.stop()
# Removed dangling else logic — controller handled in main()

if raw_epw_bytes is None:
    pass


def show_epw_status():
    if st.session_state.pop("_just_loaded_epw", False):
        # New weather file loaded: clear prior PDF captures/download state.
        st.session_state["pdf_figures"] = {}
        st.session_state["pdf_figures_auto"] = {}
        st.session_state["pdf_captions"] = {}
        st.session_state["pdf_figure_fingerprints"] = set()
        st.session_state["pdf_download_bytes"] = None
        st.session_state["pdf_download_name"] = None
        st.session_state["pdf_download_error"] = None

        header = st.session_state.get("header")
        df = st.session_state.get("df")
        epw_notes = st.session_state.get("_last_epw_notes")
        if epw_notes is None:
            epw_notes = []

        location_meta = header.get("location", {}) if isinstance(header, dict) else {}
        city = location_meta.get("city") or location_meta.get("state_province") or "Unknown"
        country = location_meta.get("country", "")
        period = location_meta.get("period") or location_meta.get("data_periods") or "—"
        domain = location_meta.get("source", "EPW")
        record_count = len(df) if isinstance(df, pd.DataFrame) else 0
        source_label = st.session_state.get("source_label", "")

        # --- DASHBOARD LOGIC ---
    
        # ------------------
        # TABS
        # ------------------
        st.session_state.pop("loading_station_name", None)


def setup_cdf():
    cdf = st.session_state.get("cdf")

    if cdf is None and st.session_state.get("nav_page") != DEFAULT_PAGE:
        pass  # nav_page is widget-owned; sidebar handles routing

    # Harmonize alternate column names that may come from different EPW parsers (e.g., pvlib)
    if cdf is not None:
        alias_columns = {
            "temp_air": "drybulb",
            "temp_dew": "dewpoint",
            "dew_temperature": "dewpoint",
            "relative_humidity": "relhum",
            "rel_humidity": "relhum",
            "ghi": "glohorrad",
            "dni": "dirnorrad",
            "dhi": "difhorrad",
            "windspeed": "windspd",
            "wind_speed": "windspd",
            "windspeedms": "windspd",
            "windspeed_ms": "windspd",
            "HourlyWindSpeed": "windspd",
            "liqprecipdepth": "liq_precip_depth",
            "liqpreciprate": "liq_precip_rate",
            "precipwtr": "precip_wtr",
            "precipdepth": "liq_precip_depth",
            "liqprecip": "liq_precip_depth",
            "pressure": "atmos_pressure",
            "atmospheric_pressure": "atmos_pressure",
        }
        for src, dest in alias_columns.items():
            if dest not in cdf.columns and src in cdf.columns:
                cdf[dest] = cdf[src]
    
    page = st.session_state.get("nav_page", DEFAULT_PAGE)
    if page not in ALLOWED_PAGES:
        page = DEFAULT_PAGE
    return page

# ========== HEATMAP HELPERS (ANNUAL DIURNAL RESOURCE) ==========

WIND_SPEED_ALIASES = [
    "windspeed", "windspd", "wind_speed", "wspd", "ws",
    "windspeedms", "wind speed", "windspeed_ms", "HourlyWindSpeed",
]
PRECIP_ALIASES = [
    "liqprecipdepth", "liqpreciprate", "precipwtr", "precipitation",
    "rain", "precip", "precipdepth", "liqprecip", "Precipitable Water",
    "liq_precip_depth", "liq_precip_rate", "precip_wtr",
]


def find_column_by_fuzzy_match(df: pd.DataFrame, keywords: List[str], exclude_cols: List[str] = None) -> Optional[str]:
    """Find a column in df by fuzzy matching against keywords (case-insensitive, substring)."""
    if exclude_cols is None:
        exclude_cols = []
    # Build a mapping from lowercase name to original name (filtered)
    col_map = {c.lower(): c for c in df.columns if c not in exclude_cols}
    for keyword in keywords:
        kw_lower = keyword.lower()
        for col_low, col_orig in col_map.items():
            if kw_lower in col_low or col_low in kw_lower:
                return col_orig
            kw_norm = re.sub(r"[^a-z0-9]+", "", kw_lower)
            col_norm = re.sub(r"[^a-z0-9]+", "", col_low)
            if kw_norm and (kw_norm in col_norm or col_norm in kw_norm):
                return col_orig
    return None


def get_metric_column(df: pd.DataFrame, keywords: List[str], exclude_cols: List[str] = None) -> Optional[str]:
    """Find a metric column and store debug context for missing wind/precip aliases."""
    keywords = list(keywords or [])
    key_norms = {re.sub(r"[^a-z0-9]+", "", str(k).lower()) for k in keywords}
    alias_group = None
    if key_norms.intersection({"wind", "windspeed", "windspeedms", "windspd", "windspdm", "wspd", "ws", "hourlywindspeed"}):
        alias_group = "wind_speed"
        keywords = list(dict.fromkeys(keywords + WIND_SPEED_ALIASES))
    elif key_norms.intersection({"precip", "precipitation", "rain", "liqprecipdepth", "liqpreciprate", "precipwtr", "precipdepth"}):
        alias_group = "precipitation"
        keywords = list(dict.fromkeys(keywords + PRECIP_ALIASES))

    match = find_column_by_fuzzy_match(df, keywords, exclude_cols)
    if match is None and alias_group and df is not None:
        debug = st.session_state.get("debug_missing_cols", {})
        debug[alias_group] = list(df.columns)
        st.session_state["debug_missing_cols"] = debug
    return match


def coerce_to_numeric(series: pd.Series) -> pd.Series:
    """Coerce a series to numeric, replacing errors with NaN."""
    return pd.to_numeric(series, errors="coerce")


def month_labels_at_midpoints(leap_year: bool = False) -> Tuple[List[float], List[str]]:
    """Return (doy_positions, month_letters) for month midpoints in a reference (non-leap) year.
    
    Uses a fixed reference year (e.g., 2021 for non-leap) to compute DOY positions.
    """
    ref_year = 2021  # Non-leap reference year
    month_letters = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    doy_midpoints = []
    for month in range(1, 13):
        # Use the 15th of each month as midpoint
        d = pd.Timestamp(ref_year, month, 15)
        doy = d.dayofyear
        doy_midpoints.append(float(doy))
    return doy_midpoints, month_letters


def month_boundaries_doy(leap_year: bool = False) -> List[float]:
    """Return DOY values for month boundaries in a reference (non-leap) year."""
    ref_year = 2021  # Non-leap reference year
    boundaries = []
    for month in range(1, 13):
        # First day of each month
        d = pd.Timestamp(ref_year, month, 1)
        boundaries.append(float(d.dayofyear))
    # Add boundary for next year start
    boundaries.append(366.0 if leap_year else 365.0)
    return boundaries


def bin_metric(values: pd.Series, metric_name: str) -> pd.Series:
    """Bin a metric series into discrete categories based on predefined thresholds.
    
    Returns a Series of category integers (0, 1, 2, ...).
    """
    metric_lower = metric_name.lower()
    
    if any(x in metric_lower for x in ["temp", "dry_bulb"]):
        # Temperature: <10, 10–20, 20–26, 26–35, >35
        bins = [-np.inf, 10, 20, 26, 35, np.inf]
        labels = ["<10°C", "10–20°C", "20–26°C", "26–35°C", ">35°C"]
    elif any(x in metric_lower for x in ["radiation", "solar", "ghi", "global"]):
        # Solar: <100, 100–300, 300–500, 500–700, >700
        bins = [-np.inf, 100, 300, 500, 700, np.inf]
        labels = ["<100", "100–300", "300–500", "500–700", ">700"]
    elif any(x in metric_lower for x in ["humidity", "rh", "relhum"]):
        # RH: <40, 40–60, 60–80, >80
        bins = [-np.inf, 40, 60, 80, np.inf]
        labels = ["<40%", "40–60%", "60–80%", ">80%"]
    elif any(x in metric_lower for x in ["wind", "speed", "wspd"]):
        # Wind speed: <1.5, 1.5–4.5, >4.5
        bins = [-np.inf, 1.5, 4.5, np.inf]
        labels = ["<1.5 m/s", "1.5–4.5 m/s", ">4.5 m/s"]
    else:
        # Default: quartiles
        q25, q50, q75 = values.quantile([0.25, 0.5, 0.75])
        bins = [-np.inf, q25, q50, q75, np.inf]
        labels = ["Q1", "Q2", "Q3", "Q4"]
    
    try:
        # pd.cut returns categorical; convert to numeric codes (0, 1, 2, ...)
        cat = pd.cut(values, bins=bins, labels=labels, right=False, duplicates="drop")
        codes = cat.cat.codes.astype(float)
        codes[codes < 0] = np.nan
        return codes
    except Exception:
        return pd.Series(np.nan, index=values.index)


@st.cache_data
def compute_heatmap_matrix(df: pd.DataFrame, metric_col: str, metric_name: str) -> Tuple[pd.DataFrame, dict]:
    """Compute pivot table (hod x doy) for a metric and return both raw and binned matrices.
    
    Returns: (raw_pivot, info_dict)
    """
    try:
        # Coerce metric to numeric
        metric_vals = coerce_to_numeric(df[metric_col])
        work = df.copy()
        work[metric_col] = metric_vals
        work = work.dropna(subset=[metric_col])
        
        if work.empty:
            return pd.DataFrame(), {"error": f"No valid data for {metric_name}"}
        
        # Extract hour-of-day and day-of-year
        work["hod"] = work.index.hour
        work["doy"] = work.index.dayofyear
        
        # Pivot: hod (rows) x doy (cols), aggregate with mean
        pivot_raw = work.pivot_table(index="hod", columns="doy", values=metric_col, aggfunc="mean")
        
        # Reindex to ensure full 0..23 hod and 1..366 doy
        pivot_raw = pivot_raw.reindex(index=range(24), columns=range(1, 367), fill_value=np.nan)
        
        # Bin the raw values
        binned_flat = pd.Series(pivot_raw.values.flatten())
        binned_series = bin_metric(binned_flat, metric_name)
        pivot_binned = binned_series.values.reshape(pivot_raw.shape)
        pivot_binned = pd.DataFrame(pivot_binned, index=pivot_raw.index, columns=pivot_raw.columns)
        
        info = {
            "metric": metric_name,
            "col": metric_col,
            "n_valid": len(work),
            "min": pivot_raw.min().min(),
            "max": pivot_raw.max().max(),
        }
        return pivot_binned, info
    except Exception as e:
        return pd.DataFrame(), {"error": str(e)}


def get_color_scale_for_metric(metric_name: str) -> Tuple[List[str], List[str]]:
    """Return (colors_list, labels_list) for a metric's discrete color scale."""
    metric_lower = metric_name.lower()
    
    if any(x in metric_lower for x in ["temp", "dry_bulb", "drybulb"]):
        colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c", "#8b0000"]
        labels = ["<10°C", "10–20°C", "20–26°C", "26–35°C", ">35°C"]
    elif any(x in metric_lower for x in ["radiation", "solar", "ghi", "global"]):
        colors = ["#1a1a2e", "#3498db", "#f39c12", "#e74c3c", "#fff700"]
        labels = ["<100", "100–300", "300–500", "500–700", ">700"]
    elif any(x in metric_lower for x in ["humidity", "rh", "relhum"]):
        # Blue-based palette for humidity bands
        colors = ["#d8eaff", "#7fb6ff", "#2f80d3", "#1b1c70"]
        labels = ["<40%", "40–60%", "60–80%", ">80%"]
    elif any(x in metric_lower for x in ["wind", "speed", "wspd"]):
        colors = ["#3498db", "#2ecc71", "#e74c3c"]
        labels = ["<1.5 m/s", "1.5–4.5 m/s", ">4.5 m/s"]
    else:
        colors = ["#e8f4f8", "#a9d6e5", "#51afc5", "#0d5f6f"]
        labels = ["Q1", "Q2", "Q3", "Q4"]
    
    return colors, labels


def build_diurnal_heatmap_figure(heatmap_dict: Dict, cdf: pd.DataFrame, header: dict) -> Optional[go.Figure]:
    """Build a multi-strip heatmap figure (subplots, one per metric).
    
    # Force reload to fix cache (Legends Update)
    heatmap_dict: {"metric_name": (pivot_binned, info_dict), ...}
    """
    if not heatmap_dict or all("error" in v[1] for v in heatmap_dict.values()):
        return None
    
    # Filter out metrics with errors
    valid_strips = {k: v for k, v in heatmap_dict.items() if "error" not in v[1]}
    if not valid_strips:
        return None
    
    n_strips = len(valid_strips)
    
    # Use real dates for the X-axis (Year 2021 as standard non-leap year)
    # pivot_binned has columns 1..365 (or 366). We map them to dates.
    # We'll generate a date range for 365 days.
    dates_2021 = pd.date_range(start="2021-01-01", periods=365, freq="D")
    
    fig = make_subplots(
        rows=n_strips, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[],  # We will add custom annotations below each plot
    )
    
    heatmap_indices = []

    for row, (strip_name, (pivot_binned, info)) in enumerate(valid_strips.items(), start=1):
        if pivot_binned.empty:
            continue
        
        # Ensure pivot aligns with our 365-day date range
        if pivot_binned.shape[1] > 365:
            pivot_plot_slice = pivot_binned.iloc[:, :365]
        else:
            pivot_plot_slice = pivot_binned.reindex(columns=range(1, 366))
            
        # Define discrete color scales and bins matching the printed style reference
        # keys: metric_name -> (bounds, colors, tick_labels)
        # Colors approximated from the user provided image
        discrete_scales = {
            "Dry Bulb Temperature": (
                [-100, 10, 20, 26.6, 35, 100], 
                # Blue -> Light Blue -> White -> Orange -> Red
                ["#6baed6", "#bdd7e7", "#ffffff", "#fd8d3c", "#e31a1c"], 
                ["<10°C", "<20°C", "Comf. Zone", ">26.6°C", ">35°C"]
            ),
            "Solar Radiation": (
                [0, 100, 300, 500, 700, 9999], 
                # Very Light Orange -> Light Orange -> Orange -> Dark Orange -> Brown
                ["#feedde", "#fdbe85", "#fd8d3c", "#e6550d", "#a63603"],
                ["<100 w/m²", "100-300 w/m²", "300-500 w/m²", "500-700 w/m²", ">700 w/m²"]
            ),
            # Approximate Absolute Humidity: Grey -> White -> Blue
            "Absolute Humidity": ( 
                [0, 5, 12, 100], # Guessing thresholds based on "occasional low moisture" & "standard"
                ["#cccccc", "#ffffff", "#6baed6"], 
                ["<?g/kg", "Comf. Zone", ">?g/kg"] 
            ),
            "Humidity": ( # Relative Humidity
                [0, 30, 70, 100], 
                # Orange -> White -> Blue
                ["#fd8d3c", "#ffffff", "#6baed6"], 
                ["<30%", "Comf. Zone", ">70%"]
            ),
            "Precipitation": (
                [0, 0.1, 2.5, 10.0, 999],
                ["#ffffff", "#93c5fd", "#2563eb", "#1e3a8a"],
                ["Dry", "Light", "Moderate", "Heavy"]
            ),
            "Wind Speed": (
                [0, 1.5, 4.5, 999], 
                ["#bde0fe", "#ffffff", "#74c476"], 
                ["<1.5 m/s", "1.5-4.5 m/s", ">4.5 m/s"] 
            ),
            "Wind Direction": (
                list(range(9)), # 0..8 bounds for 8 categories
                ["#4c6fff", "#3fb3ff", "#36d1a8", "#8bd36b", "#f6c445", "#f08c42", "#e15b9a", "#9d6bff"],
                ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
            )
        }

        # Determine colorscale and colorbar settings
        heatmap_colorscale = None
        zmin, zmax = None, None
        colorbar_dict = None
        show_scale = False
        active_tick_labels = []

        if strip_name in discrete_scales:
            bounds, colors, ticks = discrete_scales[strip_name]
            active_tick_labels = ticks
            
            # Special handling for Categorical Bins (Wind Speed, Solar, Wind Direction, Dry Bulb)
            if strip_name in ["Wind Speed", "Solar Radiation", "Wind Direction", "Dry Bulb Temperature", "Humidity", "Precipitation"]:
                # Categorical Logic
                n = len(colors)
                zmin = 0
                zmax = n 
                
                # Construct categorical colorscale
                scale = []
                for i, col in enumerate(colors):
                    low = i / n
                    high = (i + 1) / n
                    scale.append([low, col])
                    scale.append([high, col])
                
                heatmap_colorscale = scale
                
                # Tick placement: Center of each bin
                tick_vals = [i + 0.5 for i in range(n)]
                
            else:
                d_min = bounds[0]
                d_max = bounds[-1]
                zmin = d_min
                zmax = d_max
                
                if strip_name == "Humidity":
                    zmin, zmax = 0, 100
                    d_min, d_max = 0, 100

                # Build the stepped scale
                scale = []
                n_colors = len(colors)
                # Ensure bounds length matches n_colors + 1
                safe_bounds = bounds if len(bounds) == n_colors + 1 else np.linspace(d_min, d_max, n_colors+1)
                
                for i in range(n_colors):
                    val_min = safe_bounds[i]
                    val_max = safe_bounds[i+1]
                    lower_frac = (val_min - d_min) / (d_max - d_min)
                    upper_frac = (val_max - d_min) / (d_max - d_min)
                    lower_frac = max(0.0, min(1.0, lower_frac))
                    upper_frac = max(0.0, min(1.0, upper_frac))
                    scale.append([lower_frac, colors[i]])
                    scale.append([upper_frac, colors[i]])
                
                heatmap_colorscale = scale
                
                tick_vals = []
                for i in range(n_colors):
                   v1 = safe_bounds[i]
                   v2 = safe_bounds[i+1]
                   tick_vals.append((v1 + v2)/2)
           
            # Vertical Legend Position
            space = 0.04
            h = (1.0 - (n_strips - 1) * space) / n_strips
            y_top = 1.0 - (row - 1) * (h + space)
            y_center = y_top - h / 2
            
            # Adjust title for Wind Direction
            cb_title = None

            colorbar_dict = dict(
                orientation="v",
                x=1.01,
                y=y_center,
                yanchor="middle",
                xanchor="left",
                len=h,
                thickness=25,
                tickmode="array",
                tickvals=tick_vals,
                ticktext=ticks,
                tickfont=dict(size=9),
                title=cb_title
            )
            show_scale = True

        #else:
            # Fallback for Wind Direction or others: discrete interpolation
        #    colors_default, labels_default = get_color_scale_for_metric(info["metric"])
        #    colors = info.get("colors", colors_default[: len(info.get("labels", labels_default))])
        #    heatmap_colorscale = list(zip([i / (len(colors) - 1) for i in range(len(colors))], colors))
            
            # Special Compass Legend for Wind Direction
        #   if strip_name == "Wind Direction":
        #        # Use a cyclic colorscale (HSV)
        #        heatmap_colorscale = "HSV"
        #        zmin, zmax = 0, 360 # Explicit degrees
        #        # We need a separate legend for this maybe? 
        #        # For now, standard colorbar 0-360
        #        colorbar_dict = dict(
        #           orientation="v", x=1.01, y=y_center, yanchor="middle", len=h, thickness=15,
        #           tickmode="array", tickvals=[0, 90, 180, 270, 360], ticktext=["N", "E", "S", "W", "N"]
        #        )
        #        show_scale = True

        # Gentle DOY smoothing 
        if info["metric"] in {"Wind Direction", "Precipitation"}:
             pivot_plot = pd.DataFrame(pivot_plot_slice).copy()
        else:
            pivot_plot = pd.DataFrame(pivot_plot_slice).copy().T.rolling(window=5, center=True, min_periods=1).mean().T

        hover_labels = info.get("hover_labels")
        if hover_labels is not None:
             if hasattr(hover_labels, "shape") and hover_labels.shape[1] > 365:
                 hover_labels = hover_labels[:, :365]
        
        customdata = hover_labels if hover_labels is not None else None
        
        # Updated Hover Template: Remove Year
        hovertemplate = (
            "<b>%{x|%b %d} %{y}:00</b><br>" +
            f"{strip_name}: " +
            "%{customdata}<extra></extra>"
            if customdata is not None
            else "<b>%{x|%b %d} %{y}:00</b><br>Value: %{z:.2f}<extra></extra>"
        )

        trace = go.Heatmap(
            z=pivot_plot.values,
            x=dates_2021,       # Real DatetimeIndex for positioning
            y=pivot_plot.index, # HOD 0..23
            colorscale=heatmap_colorscale,
            showscale=show_scale,
            colorbar=colorbar_dict,
            zmin=zmin,
            zmax=zmax,
            customdata=customdata,
            hovertemplate=hovertemplate,
            showlegend=False,
            zsmooth=False, # Disable smoothing for crisp blocks
        )
        fig.add_trace(trace, row=row, col=1)
        heatmap_indices.append((len(fig.data) - 1, show_scale))
        
        # Configure y-axis (HOD)
        # 12am, Noon, 11pm
        fig.update_yaxes(
            tickmode="array",
            tickvals=[0, 12, 23],
            ticktext=["12:00am", "noon", "11:59pm"],
            gridcolor="rgba(128,128,128,0.1)",
            gridwidth=0.5,
            tickfont=dict(size=9, color="#666666"),
            row=row, col=1,
            autorange="reversed" # 0 at top (12am), 23 at bottom (11pm) matches image? 
            # Image shows 12:00am at top, noon middle, 11:59pm bottom. 
            # Plotly default: 0 at bottom. So "reversed" makes 0 top.
        )
        
        # Calculate percentage stats for the description
        total_valid_hours = float(np.sum(~np.isnan(pivot_plot_slice.values)))
        description = ""
        if total_valid_hours > 0:
            def _category_distribution_text(labels: List[str], sort_desc: bool = False) -> str:
                values = pd.to_numeric(pd.Series(np.ravel(pivot_plot_slice.values)), errors="coerce").dropna()
                total = float(len(values))
                if total <= 0 or not labels:
                    return ""
                rows = []
                for idx, label in enumerate(labels):
                    pct = float((values == idx).sum()) / total * 100.0
                    rows.append((idx, label, pct))
                if sort_desc:
                    rows.sort(key=lambda item: (-item[2], item[0]))
                return ", ".join(f"{label}: {pct:.1f}%" for _, label, pct in rows)

            if strip_name == "Dry Bulb Temperature":
                # Assuming Comfort Zone is 20°C to 26.6°C (bin index 2)
                # The data is binned 0-4. Bin 2 is Comf. Zone.
                comf_hours = np.sum(pivot_plot_slice.values == 2)
                pct = (comf_hours / total_valid_hours) * 100
                description = f"({pct:.1f}% of hours in Comf. Zone)"
            elif strip_name == "Solar Radiation":
                # Bin 3 is 500-700, Bin 4 is >700
                high_rad_hours = np.sum(pivot_plot_slice.values >= 3)
                pct = (high_rad_hours / total_valid_hours) * 100
                description = f"({pct:.1f}% of hours > 500 W/m²)"
            elif strip_name == "Wind Speed":
                # Assuming >4.5 m/s is the highest bin (Bin 2 based on provided config)
                high_wind_hours = np.sum(pivot_plot_slice.values == 2)
                pct = (high_wind_hours / total_valid_hours) * 100
                description = f"({pct:.1f}% of hours > 4.5 m/s)"
            elif strip_name == "Humidity":
                # RH comfort zone: 30-70% maps to bin index 1 (middle bin)
                comf_rh_hours = np.sum(pivot_plot_slice.values == 1)
                pct = (comf_rh_hours / total_valid_hours) * 100
                comf_rh_hours_int = int(comf_rh_hours)
                description = f"({comf_rh_hours_int} hours / {pct:.1f}% in Comfort Zone 30–70%)"

            elif strip_name == "Precipitation":
                description = f"({_category_distribution_text(active_tick_labels)})"
            elif strip_name == "Wind Direction":
                description = f"({_category_distribution_text(active_tick_labels, sort_desc=True)})"

        # Add Title Annotation BELOW the heatmap
        fig.add_annotation(
            xref=f"x{row if row > 1 else ''} domain",
            yref=f"y{row if row > 1 else ''} domain",
            x=0.0, 
            y=-0.15, # Position below, moved left
            text=f"{strip_name.upper()}: {description}", 
            showarrow=False,
            font=dict(size=10, color="#d1d5db", weight="bold"),
            xanchor="left",
            yshift=0
        )

        # Configure x-axis for this row - HIDE TICKS primarily
        fig.update_xaxes(
            showticklabels=False, 
            showgrid=False,
            zeroline=False,
            row=row, col=1,
        )
        
        # Add Vertical Month Lines
        month_starts = pd.date_range("2021-01-01", "2021-12-01", freq="MS")
        for date_val in month_starts:
            fig.add_vline(
                x=date_val.timestamp() * 1000, # Plotly needs ms for date axes sometimes, or just date string
                line_width=1, 
                line_dash="solid", 
                line_color="#333333", 
                opacity=0.3,
                row=row, col=1
            )

    # Add Month Initials ABOVE the top plot (Shared for all)
    month_starts = pd.date_range("2021-01-01", "2021-12-01", freq="MS")
    month_initials = list("JFMAMJJASOND")
    
    # Calculate mid-points for labels? Or just start? Image shows letter centered in month.
    # Approximate centers: +15 days
    month_centers = month_starts + pd.Timedelta(days=15)

    for date_val, label in zip(month_centers, month_initials):
        fig.add_annotation(
            x=date_val,
            y=1.02, 
            xref="x", 
            yref="paper",
            text=label,
            showarrow=False,
            font=dict(size=12, color="black", weight="bold"), 
            xanchor="center",
            yanchor="bottom"
        )

    fig.update_layout(
        autosize=False,
        width=1200,
        height=200 * n_strips + 80, 
        showlegend=False,
        title_text=None, # Clean look
        font=dict(size=10, family="Arial"),
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent
        paper_bgcolor="rgba(0,0,0,0)", # Transparent
        margin=dict(l=50, r=150, t=60, b=40),
    )
    
    return fig

SOLAR_COLORSCALE = [
    [0.00, "#ffffff"],
    [0.02, "#fff7bc"],
    [0.25, "#fec44f"],
    [0.55, "#fe9929"],
    [0.80, "#ec7014"],
    [1.00, "#cc4c02"],
]  # Unified zero-white colorscale for solar irradiance heatmaps

# ========== PDF REPORT EXPORT ==========
PDF_INK = (15, 23, 42)         # Near-black navy - primary text
PDF_MUTED = (71, 85, 105)       # Slate-500 - captions, labels
PDF_FAINT = (148, 163, 184)     # Slate-400 - dividers, faint metadata
PDF_ACCENT = (14, 116, 144)     # Teal-700 - section tags, figure labels, rule lines
PDF_RULE = (203, 213, 225)      # Slate-300 - thin rules
PDF_SOFT_BG = (241, 245, 249)   # Slate-100 - metadata boxes, alternating table rows
PDF_HIGHLIGHT = (224, 242, 254) # Sky-100 - cover accent band, figure label bg
COVER_STRIPE = (14, 116, 144)   # Teal - cover left stripe


def _pdf_page_choice() -> str:
    choice = str(st.session_state.get("export_pdf_page_size") or "A4 Landscape")
    valid = {"A4 Landscape", "A4 Portrait", "A3 Landscape", "A2 Landscape"}
    return choice if choice in valid else "A4 Landscape"


def _pdf_format_for_choice(choice: str) -> tuple[str, tuple[float, float]]:
    if choice == "A4 Portrait":
        return "P", (210, 297)
    if choice == "A4 Landscape":
        return "L", (210, 297)
    if choice == "A3 Landscape":
        return "L", (297, 420)
    return "L", (420, 594)


def _pdf_is_large_page(pdf: FPDF) -> bool:
    return float(getattr(pdf, "w", 210)) > 300


class ClimateReportPDF(FPDF):
    """Branded PDF report with clean, minimalist aesthetics inspired by true scientific journals."""

    def __init__(self, location_label: str, source_label: str, generated_on: str):
        page_choice = _pdf_page_choice()
        orientation, page_format = _pdf_format_for_choice(page_choice)
        super().__init__(orientation=orientation, unit="mm", format=page_format)
        self.location_label = location_label
        self.source_label = source_label
        self.generated_on = generated_on
        self.current_section = ""
        self.page_choice = page_choice
        self.report_title = str(st.session_state.get("export_report_title") or "Climate Analysis Report").strip()
        self.include_branding = bool(st.session_state.get("export_include_branding", True))
        self.white_label = bool(st.session_state.get("export_white_label", False))

    def header(self):
        # Skip header on cover page
        if self.page_no() == 1:
            return

        self.set_fill_color(255, 255, 255)
        self.rect(0, 0, self.w, self.h, "F")

        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(*PDF_MUTED)
        self.set_xy(16, 10)
        header_label = "CLIMATE REPORT" if self.white_label else "CLIMATE ANALYSIS DATABOOK"
        if not self.include_branding:
            header_label = "CLIMATE INTELLIGENCE REPORT"
        self.cell((self.w - 32) * 0.5, 5, _pdf_safe_text(header_label), ln=0, align="L")
        section_label = str(self.current_section or "").strip()
        if section_label.lower() in {"undefined", "none", "null"}:
            section_label = ""
        if section_label:
            self.set_font("Helvetica", "", 8.5)
            self.set_text_color(*PDF_MUTED)
            self.set_xy(self.w * 0.50, 10)
            self.cell(self.w * 0.45, 5, _pdf_safe_text(section_label[:72]), ln=0, align="R")

        self.set_draw_color(*PDF_RULE)
        self.set_line_width(0.18)
        self.line(16, 17, self.w - 16, 17)

    def footer(self):
        if self.page_no() == 1:
            return

        margin = 16
        content_w = self.w - (2 * margin)
        self.set_y(-12)
        # Thin teal rule above footer
        self.set_draw_color(*PDF_ACCENT)
        self.set_line_width(0.3)
        self.line(margin, self.h - 12, self.w - margin, self.h - 12)

        self.set_font("Helvetica", "", 8)
        self.set_text_color(*PDF_MUTED)
        self.set_xy(margin, self.h - 10)
        col_w = content_w / 3
        self.cell(col_w, 5, _pdf_safe_text(self.location_label[:64]), ln=0, align="L")
        self.cell(col_w, 5, f"Generated {self.generated_on}", ln=0, align="C")
        self.cell(col_w, 5, f"Page {self.page_no()} of {{nb}}", ln=0, align="R")


def _export_dimensions_for_figure(fig: go.Figure, width: int, height: int) -> Tuple[int, int]:
    """Choose a stable export viewport without changing the figure layout itself."""
    export_width = width or REPORT_EXPORT_WIDTH
    export_height = height or REPORT_EXPORT_HEIGHT

    try:
        layout_width = getattr(fig.layout, "width", None)
        if layout_width:
            export_width = int(layout_width)
    except Exception:
        pass

    try:
        layout_height = getattr(fig.layout, "height", None)
        if layout_height:
            export_height = int(layout_height)
    except Exception:
        pass

    try:
        trace_types = {str(getattr(tr, "type", "")).lower() for tr in (fig.data or [])}
        if "barpolar" in trace_types or "scatterpolar" in trace_types:
            export_height = max(export_height, min(export_width, 900))
        elif len(getattr(fig.layout, "annotations", []) or []) >= 8:
            export_height = max(export_height, 820)
    except Exception:
        pass

    return max(900, int(export_width)), max(520, int(export_height))


def _fig_to_tmp_png(fig, width: int = REPORT_EXPORT_WIDTH, height: int = REPORT_EXPORT_HEIGHT, scale: int = REPORT_EXPORT_SCALE) -> str:
    """Export the captured dashboard Plotly figure to a temporary PNG file."""
    import gc
    fig_for_export = _clone_dashboard_figure(fig)
    if not isinstance(fig_for_export, go.Figure):
        raise RuntimeError("PDF export only supports captured Plotly figures.")

    export_width, export_height = _export_dimensions_for_figure(fig_for_export, width, height)

    # On cloud, cap dimensions aggressively to avoid OOM in the Kaleido subprocess.
    if _IS_STREAMLIT_CLOUD:
        export_width = min(export_width, 900)
        export_height = min(export_height, 520)
        scale = 1

    export_attempts = [
        (export_width, export_height, scale),
        (export_width, export_height, 1),
        (max(800, int(export_width * 0.75)), max(450, int(export_height * 0.75)), 1),
    ]

    last_error = None
    for attempt_width, attempt_height, attempt_scale in export_attempts:
        try:
            img_bytes = pio.to_image(
                fig_for_export,
                format="png",
                width=attempt_width,
                height=attempt_height,
                scale=attempt_scale,
                validate=False,
            )
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".kaleido.png")
            tmp.write(img_bytes)
            tmp.close()
            # Free the raw bytes immediately to reduce peak memory.
            del img_bytes
            gc.collect()
            return tmp.name
        except Exception as exc:
            last_error = exc
            gc.collect()  # reclaim memory before retry

    raise RuntimeError(
        "Plotly/Kaleido image export failed. The PDF pipeline intentionally avoids "
        f"simplified Matplotlib redraws so figures remain faithful to the dashboard. Last error: {last_error}"
    )


def _safe_location_label(header: dict) -> str:
    if not header or not isinstance(header, dict):
        return "Unknown Location"
    loc = header.get("location", {}) if isinstance(header, dict) else {}
    if isinstance(loc, dict):
        city = None
        for key in ("city", "name", "stationname", "station"):
            val = loc.get(key)
            if _is_valid_label_value(val):
                city = val.strip()
                break
        country = loc.get("country") or ""
        if not _is_valid_label_value(country):
            country = ""
        if city:
            return f"{city}, {country}".strip(", ") if country else city
        return "Unknown Location"
    if isinstance(loc, str) and _is_valid_label_value(loc):
        return loc.strip()
    return "Unknown Location"


def _location_meta(header: dict) -> Dict[str, str]:
    """Return normalized location metadata for report cover/header."""
    loc = header.get("location", {}) if isinstance(header, dict) else {}
    city = "Unknown"
    country = "Unknown"
    lat = "--"
    lon = "--"
    tz = "--"
    elev = "--"
    wmo = "--"

    if isinstance(loc, dict):
        raw_city = loc.get("city") or loc.get("name")
        city = str(raw_city).strip() if _is_valid_label_value(raw_city) else city
        raw_country = loc.get("country")
        country = str(raw_country).strip() if _is_valid_label_value(raw_country) else country
        try:
            if loc.get("latitude") is not None:
                lat = f"{float(loc.get('latitude')):.3f}"
            if loc.get("longitude") is not None:
                lon = f"{float(loc.get('longitude')):.3f}"
            if loc.get("elevation_m") is not None:
                elev = f"{float(loc.get('elevation_m')):.0f} m"
        except Exception:
            pass
        if loc.get("timezone") is not None:
            tz = str(loc.get("timezone"))
        if loc.get("wmo") is not None:
            wmo = str(loc.get("wmo"))

    return {
        "city": city,
        "country": country,
        "lat": lat,
        "lon": lon,
        "tz": tz,
        "elev": elev,
        "wmo": wmo,
    }


def _normalized_pdf_figures(figs: dict) -> Dict[str, object]:
    """Normalize figure keys so naming whitespace/typos do not skip charts."""
    if not isinstance(figs, dict):
        return {}
    out: Dict[str, object] = {}
    for raw_key, fig in figs.items():
        key = str(raw_key).strip()
        if key:
            out[key] = fig
    return out


def _section_tag(title: str) -> str:
    low = title.lower()
    if "sun" in low or "solar" in low or "irradiance" in low or "insolation" in low or "cloud" in low:
        return "Solar & Sky Analysis"
    if "wind" in low:
        return "Wind Data"
    if "psych" in low:
        return "Psychrometrics"
    if "comfort" in low or "load" in low or "pmv" in low or "utci" in low or "mrt" in low:
        return "Thermal Comfort"
    if "humid" in low or "drybulb" in low or "temp" in low or "dew" in low or "degree day" in low:
        return "Temperature & Humidity"
    return "Overview Metadata"


def _add_manual_pdf_figure(key: str, fig: object) -> None:
    store = st.session_state.get("pdf_figures", {})
    store[str(key).strip()] = _clone_dashboard_figure(fig)
    st.session_state["pdf_figures"] = store


def _add_manual_pdf_caption(key: str, caption: str) -> None:
    if not caption:
        return
    store = st.session_state.get("pdf_captions", {})
    clean_key = str(key).strip()
    store[clean_key] = str(caption).strip()
    formatted = format_figure_title(clean_key)
    if formatted and formatted != clean_key:
        store[formatted] = str(caption).strip()
    st.session_state["pdf_captions"] = store


def _remove_manual_pdf_captions(keys: Iterable[str]) -> None:
    store = st.session_state.get("pdf_captions", {})
    if not isinstance(store, dict) or not store:
        return
    remove_norms = set()
    for key in keys:
        clean_key = str(key).strip()
        if not clean_key:
            continue
        remove_norms.add(_normalize_report_key(clean_key))
        formatted = format_figure_title(clean_key)
        if formatted:
            remove_norms.add(_normalize_report_key(formatted))
    if not remove_norms:
        return
    st.session_state["pdf_captions"] = {
        k: v for k, v in store.items()
        if _normalize_report_key(k) not in remove_norms
    }


def _merged_pdf_figures() -> Dict[str, object]:
    """Return collected Plotly figures, perfectly deduplicated."""
    manual_figs = _normalized_pdf_figures(st.session_state.get("pdf_figures", {}))
    auto_figs = _normalized_pdf_figures(st.session_state.get("pdf_figures_auto", {}))

    def _is_map_figure(fig_obj: object) -> bool:
        map_trace_types = {
            "scattermapbox", "choroplethmapbox", "densitymapbox", "scattergeo", "choropleth", "densitygeo"
        }
        for tr in getattr(fig_obj, "data", []) or []:
            if str(getattr(tr, "type", "")).lower() in map_trace_types:
                return True
        return False

    def _fallback_title(fig_obj: object) -> str:
        trace_types = [str(getattr(tr, "type", "")).lower() for tr in (getattr(fig_obj, "data", []) or [])]
        first_type = trace_types[0] if trace_types else ""
        if "heatmap" in first_type: return "Heatmap"
        if "histogram" in first_type: return "Histogram"
        if "bar" in first_type: return "Bar chart"
        if "scatter" in first_type: return "Scatter plot"
        if "surface" in first_type or "mesh3d" in first_type: return "3D chart"
        return f"{first_type.title()} chart" if first_type else "Chart"

    filtered: Dict[str, object] = {}
    seen_fingerprints = set()
    seen_report_norms = set()

    # Manual figs takes precedence (they provide better titles)
    for source in (manual_figs, auto_figs):
        for title, fig in source.items():
            if _is_map_figure(fig):
                continue
            
            # Smart deduplication by JSON fingerprint
            try:
                j = fig.to_json()
                if j in seen_fingerprints:
                    continue
                seen_fingerprints.add(j)
            except Exception:
                pass

            clean_title = str(title).strip()
            if clean_title.lower() in {"", "undefined", "none", "nan"} or clean_title.lower().startswith("visualization"):
                clean_title = _fallback_title(fig)

            report_norms = _report_equivalent_norms(clean_title)
            if report_norms and not report_norms.isdisjoint(seen_report_norms):
                continue
            seen_report_norms.update(report_norms)

            unique_title = clean_title
            suffix = 2
            while unique_title in filtered:
                unique_title = f"{clean_title} ({suffix})"
                suffix += 1
            filtered[unique_title] = fig

    return filtered


def _request_dashboard_pdf_build() -> None:
    """Request a dashboard-only PDF build on the next dashboard render pass."""
    st.session_state["pdf_dashboard_autobuild_pending"] = True
    st.session_state["pdf_download_bytes"] = None
    st.session_state["pdf_download_name"] = None
    st.session_state["pdf_download_error"] = None
    st.session_state["pdf_capture_origin_page"] = st.session_state.get("nav_page", DEFAULT_PAGE)

    # Fresh capture per export click.
    st.session_state["pdf_figures"] = {}
    st.session_state["pdf_figures_auto"] = {}
    st.session_state["pdf_captions"] = {}
    st.session_state["pdf_figure_fingerprints"] = set()


def _finalize_dashboard_pdf_if_pending(effective_page: str) -> None:
    """Build PDF once dashboard tabs are rendered in the normal app flow.

    The sidebar is rendered before dashboard figures are captured, so trigger one
    clean rerun after building the PDF to reveal the download button immediately.
    """
    if not st.session_state.get("pdf_dashboard_autobuild_pending", False):
        return
    if effective_page != "Dashboard":
        return

    figs = _merged_pdf_figures()
    if not figs:
        st.session_state["pdf_download_bytes"] = None
        st.session_state["pdf_download_name"] = None
        st.session_state["pdf_download_error"] = "No dashboard visualizations were captured."
        st.session_state["pdf_dashboard_autobuild_pending"] = False
        origin_page = st.session_state.get("pdf_capture_origin_page") or "Export"
        if origin_page not in ALLOWED_PAGES or origin_page == "Dashboard":
            origin_page = "Export"
        st.session_state["nav_page"] = origin_page
        st.session_state["active_page"] = "select_station" if origin_page == DEFAULT_PAGE else "dashboard"
        _rerun()
        return

    try:
        pdf_bytes = build_climate_pdf()
        loc_name = _safe_location_label(st.session_state.get("header") or {})
        safe_name = str(loc_name).replace(" ", "_").replace(",", "")

        st.session_state["pdf_download_bytes"] = pdf_bytes
        st.session_state["pdf_download_name"] = f"{safe_name}_Report.pdf"
        st.session_state["pdf_download_error"] = None
    except Exception as exc:
        st.session_state["pdf_download_bytes"] = None
        st.session_state["pdf_download_name"] = None
        st.session_state["pdf_download_error"] = f"PDF generation failed: {exc}"
    st.session_state["pdf_dashboard_autobuild_pending"] = False
    origin_page = st.session_state.get("pdf_capture_origin_page") or "Export"
    if origin_page not in ALLOWED_PAGES or origin_page == "Dashboard":
        origin_page = "Export"
    st.session_state["nav_page"] = origin_page
    st.session_state["active_page"] = "select_station" if origin_page == DEFAULT_PAGE else "dashboard"
    _rerun()


def _render_all_visualizations_silently() -> None:
    """Placeholder - rendering all pages from sidebar causes SessionInfo errors.
    
    Instead, users navigate naturally through tabs which auto-captures everything.
    This maintains Streamlit's session protocol integrity.
    """
    pass


REPORT_TAB_ORDER = [
    "Overview & Stats",
    "Comfort & Loads",
    "Temp & Humidity",
    "Solar Analysis",
    "Psychrometrics",
    "Wind",
    "Precipitation",
    "Raw Data",
]


REPORT_STRUCTURE = [
    {
        "tab": "Overview & Stats",
        "section": "Overview Metadata",
        "intro": "Site context and broad resource patterns are introduced before the report moves into thermal comfort, temperature, solar, psychrometric, and wind diagnostics.",
        "figures": [
            {"aliases": ["Annual Diurnal Resource Heatmap"], "title": "Annual Diurnal Resource Heatmap"},
        ],
    },
    {
        "tab": "Comfort & Loads",
        "section": "Thermal Comfort",
        "intro": "Comfort metrics indicate when conditions are naturally acceptable versus when active conditioning or adaptive strategies are required.",
        "figures": [
            {"aliases": ["Comfort Loads"], "title": "Comfort Load Profile"},
            {"aliases": ["di_index Annual Heatmap"], "title": "Discomfort Index by Hour and Day"},
            {"aliases": ["utci_index Annual Heatmap", "UTCI Heatmap"], "title": "UTCI by Hour and Day"},
            {"aliases": ["pmv_index Annual Heatmap"], "title": "PMV by Hour and Day"},
            {"aliases": ["Mean Radiant Temperature by Hour and Day", "MRT Heatmap", "mrt_heatmap"], "title": "Mean Radiant Temperature by Hour and Day"},
            {"aliases": ["Full Hourly Time Series — Dry-Bulb and Dew-Point Temperature", "DB DP Time Series", "hourly_timeseries_temperature"], "title": "Full Hourly Time Series — Dry-Bulb and Dew-Point Temperature"},
            {"aliases": ["Full Hourly Time Series — Relative Humidity", "RH Time Series", "hourly_timeseries_rh"], "title": "Full Hourly Time Series — Relative Humidity"},
            {"aliases": ["Seasonal Psychrometric Points", "Seasonal Psychrometric Points (4-panel)", "seasonal_psychrometric_points"], "title": "Seasonal Psychrometric Points"},
            {"aliases": ["Hourly Psychrometric Paths", "Hourly Psychrometric Paths (monthly)", "hourly_psychrometric_paths"], "title": "Hourly Psychrometric Paths"},
            {"aliases": ["Diurnal Thermal Comfort Frequency — Shading Scenario", "Shading Scenario Matrix", "diurnal_comfort_shading"], "title": "Diurnal Thermal Comfort Frequency — Shading Scenario"},
            {"aliases": ["Diurnal Thermal Comfort Frequency — Wind Scenario", "Wind Scenario Matrix", "diurnal_comfort_wind"], "title": "Diurnal Thermal Comfort Frequency — Wind Scenario"},
            {"aliases": ["Diurnal Thermal Comfort Frequency — Humidity Scenario", "Humidity Scenario Matrix", "diurnal_comfort_humidity"], "title": "Diurnal Thermal Comfort Frequency — Humidity Scenario"},
        ],
    },
    {
        "tab": "Temp & Humidity",
        "section": "Temperature & Humidity",
        "intro": "Thermo-hygrometric plots are sequenced from broad annual summaries to monthly summaries, hourly point clouds, and annual heatmaps.",
        "figures": [
            {"aliases": ["Annual Climate Statistics"], "title": "Annual Temperature and Humidity Statistics"},
            {"aliases": ["drybulb Monthly Bar"], "title": "Monthly Dry-Bulb Temperature"},
            {"aliases": ["drybulb Hourly Dot Plot"], "title": "Hourly Dry-Bulb Temperature Distribution"},
            {"aliases": ["drybulb Annual Heatmap", "Dry Bulb Heatmap", "Drybulb Temperature Matrix"], "title": "Dry-Bulb Temperature by Hour and Day"},
            {"aliases": ["relhum Monthly Bar", "relhum_monthly_bar"], "title": "Monthly Relative Humidity"},
            {"aliases": ["relhum Hourly Dot Plot", "relhum_hourly_dot_plot"], "title": "Hourly Relative Humidity Distribution"},
            {"aliases": ["relhum Annual Heatmap", "RH Heatmap", "relhum_annual_heatmap"], "title": "Relative Humidity by Hour and Day"},
        ],
    },
    {
        "tab": "Solar Analysis",
        "section": "Solar & Sky Analysis",
        "intro": "Solar-resource and cloud figures follow the same sequence as the dashboard: sun-path geometry first, then annual irradiance and sky-condition diagnostics.",
        "figures": [
            {"aliases": ["Sun Path 2D"], "title": "Sun Path Diagram (2D)"},
            {"aliases": ["Sun Path 3D"], "title": "Sun Path Diagram (3D)"},
            {"aliases": ["Sun Path Cartesian"], "title": "Sun Path (Cartesian Projection)"},
            {"aliases": ["Monthly Solar Insolation"], "title": "Monthly Solar Insolation"},
            {"aliases": ["DHI Irradiance Heatmap", "Diffuse Horizontal Irradiance Heatmap"], "title": "Diffuse Horizontal Irradiance by Hour and Day"},
            {"aliases": ["DNI Irradiance Heatmap", "Direct Normal Irradiance Heatmap", "Solar Annual Heatmap", "Irradiance Heatmap"], "title": "Direct Normal Irradiance by Hour and Day"},
            {"aliases": ["Cloud Coverage"], "title": "Monthly Cloud Coverage Frequency"},
            {"aliases": ["Cloud Coverage Scatter"], "title": "Cloud Coverage Scatter"},
            {"aliases": ["Cloud Coverage Heatmap"], "title": "Cloud Coverage by Hour and Day"},
            {"aliases": ["Longwave Horizontal Irradiance by Hour and Day", "Longwave Irradiance Heatmap", "longwave_irradiance_heatmap"], "title": "Longwave Horizontal Irradiance by Hour and Day"},
            {"aliases": ["Hourly Incoming Radiation", "Incoming Radiation Box Plot", "hourly_incoming_radiation_boxwhisker"], "title": "Hourly Incoming Radiation"},
            {"aliases": ["Monthly Solar Insolation on Inclined Surfaces", "Optimal Tilt Insolation", "inclined_surface_insolation"], "title": "Monthly Solar Insolation on Inclined Surfaces"},
        ],
    },
    {
        "tab": "Psychrometrics",
        "section": "Psychrometrics",
        "intro": "The psychrometric plot frames hourly outdoor states in temperature-moisture space for passive strategy and latent-load interpretation.",
        "figures": [
            {"aliases": ["Psychrometric Chart"], "title": "Psychrometric Chart"},
        ],
    },
    {
        "tab": "Wind",
        "section": "Wind Data",
        "intro": "Wind directionality and magnitude are shown as rendered in the dashboard for exposure, ventilation, and outdoor comfort review.",
        "figures": [
            {"aliases": ["Annual Wind Rose", "annual_wind_rose"], "title": "Annual Wind Rose"},
            {"aliases": ["Monthly Wind Speed", "monthly_wind_speed"], "title": "Monthly Mean Wind Speed"},
            {"aliases": ["Wind Speed Frequency Distribution", "wind_speed_frequency_distribution"], "title": "Wind Speed Frequency Distribution"},
            {"aliases": ["Wind Speed by Hour and Day", "Wind Speed Heatmap", "wind_speed_heatmap"], "title": "Wind Speed by Hour and Day"},
            {"aliases": ["Seasonal Wind Roses", "Seasonal Wind Roses (4-panel)", "seasonal_wind_roses"], "title": "Seasonal Wind Roses"},
            {"aliases": ["Diurnal Wind Roses", "Diurnal Wind Roses Matrix", "diurnal_wind_roses"], "title": "Diurnal Wind Roses"},
            {"aliases": ["Annual Directional Wind Power", "Wind Power 4-panel", "directional_wind_power"], "title": "Annual Directional Wind Power"},
        ],
    },
    {
        "tab": "Precipitation",
        "section": "Precipitation & Thermal Load",
        "intro": "Precipitation, snow-depth, and degree-day indicators translate hourly weather data into envelope, drainage, and seasonal load context.",
        "figures": [
            {"aliases": ["Monthly Precipitation", "monthly_precipitation"], "title": "Monthly Precipitation"},
            {"aliases": ["Snowfall Profile", "snowfall_profile"], "title": "Snowfall Profile"},
            {"aliases": ["Heating and Cooling Degree Days", "Degree Day Summary", "degree_day_summary"], "title": "Heating and Cooling Degree Days"},
        ],
    },
    {
        "tab": "Raw Data",
        "section": "Additional Captured Figures",
        "intro": "Additional dashboard figures captured during this Streamlit session are listed here in capture order.",
        "figures": [],
    },
]


def _normalize_report_key(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip().lower())
    return re.sub(r"_+", "_", s).strip("_")


def format_figure_title(raw_name: str) -> str:
    key = _normalize_report_key(raw_name)
    title_map = {
        "annual_diurnal_resource_heatmap": "Annual Diurnal Resource Heatmap",
        "annual_climate_statistics": "Annual Temperature and Humidity Statistics",
        "comfort_loads": "Comfort Load Profile",
        "relhum_monthly_bar": "Monthly Relative Humidity",
        "relhum_hourly_dot_plot": "Hourly Relative Humidity Distribution",
        "relhum_annual_heatmap": "Relative Humidity by Hour and Day",
        "drybulb_annual_heatmap": "Dry-Bulb Temperature by Hour and Day",
        "utci_index_annual_heatmap": "UTCI by Hour and Day",
        "di_index_annual_heatmap": "Discomfort Index by Hour and Day",
        "pmv_index_annual_heatmap": "PMV by Hour and Day",
        "annual_wind_rose": "Annual Wind Rose",
        "drybulb_monthly_bar": "Monthly Dry-Bulb Temperature",
        "drybulb_hourly_dot_plot": "Hourly Dry-Bulb Temperature Distribution",
        "monthly_solar_insolation": "Monthly Solar Insolation",
        "irradiance_heatmap": "Solar Irradiance by Hour and Day",
        "dhi_irradiance_heatmap": "Diffuse Horizontal Irradiance by Hour and Day",
        "dni_irradiance_heatmap": "Direct Normal Irradiance by Hour and Day",
        "cloud_coverage": "Monthly Cloud Coverage Frequency",
        "cloud_coverage_scatter": "Cloud Coverage Scatter",
        "cloud_coverage_heatmap": "Cloud Coverage by Hour and Day",
        "sun_path_2d": "Sun Path Diagram (2D)",
        "sun_path_3d": "Sun Path Diagram (3D)",
        "sun_path_cartesian": "Sun Path (Cartesian Projection)",
        "psychrometric_chart": "Psychrometric Chart",
        "monthly_wind_speed": "Monthly Mean Wind Speed",
        "wind_speed_frequency_distribution": "Wind Speed Frequency Distribution",
        "wind_speed_heatmap": "Wind Speed by Hour and Day",
        "monthly_precipitation": "Monthly Precipitation",
        "snowfall_profile": "Snowfall Profile",
        "heating_and_cooling_degree_days": "Heating and Cooling Degree Days",
        "degree_day_summary": "Heating and Cooling Degree Days",
        "longwave_horizontal_irradiance_by_hour_and_day": "Longwave Horizontal Irradiance by Hour and Day",
        "hourly_incoming_radiation": "Hourly Incoming Radiation",
        "monthly_solar_insolation_on_inclined_surfaces": "Monthly Solar Insolation on Inclined Surfaces",
        "mean_radiant_temperature_by_hour_and_day": "Mean Radiant Temperature by Hour and Day",
        "full_hourly_time_series_dry_bulb_and_dew_point_temperature": "Full Hourly Time Series — Dry-Bulb and Dew-Point Temperature",
        "full_hourly_time_series_relative_humidity": "Full Hourly Time Series — Relative Humidity",
        "seasonal_psychrometric_points": "Seasonal Psychrometric Points",
        "hourly_psychrometric_paths": "Hourly Psychrometric Paths",
        "diurnal_thermal_comfort_frequency_shading_scenario": "Diurnal Thermal Comfort Frequency — Shading Scenario",
        "diurnal_thermal_comfort_frequency_wind_scenario": "Diurnal Thermal Comfort Frequency — Wind Scenario",
        "diurnal_thermal_comfort_frequency_humidity_scenario": "Diurnal Thermal Comfort Frequency — Humidity Scenario",
        "wind_speed_by_hour_and_day": "Wind Speed by Hour and Day",
        "seasonal_wind_roses": "Seasonal Wind Roses",
        "diurnal_wind_roses": "Diurnal Wind Roses",
        "directional_wind_power": "Annual Directional Wind Power",
        "annual_directional_wind_power": "Annual Directional Wind Power",
    }
    if key in title_map:
        return title_map[key]

    cleaned = re.sub(r"[_\-]+", " ", str(raw_name)).strip()
    if re.fullmatch(r"(scatter\s+plot|plot)(\s*\[\s*\d+\s*\]|\s+\d+)?", cleaned, flags=re.IGNORECASE):
        return "Metric vs. Time - Exploratory Relationship"
    replacements = {
        r"\brelhum\b": "Relative Humidity",
        r"\brh\b": "Relative Humidity",
        r"\bdrybulb\b": "Dry-Bulb",
        r"\butci\b": "UTCI",
        r"\bpmv\b": "PMV",
        r"\bdi\b": "Discomfort Index",
        r"\bghi\b": "GHI",
        r"\bdni\b": "DNI",
        r"\bdhi\b": "DHI",
        r"\bmrt\b": "Mean Radiant Temperature",
    }
    for pattern, repl in replacements.items():
        cleaned = re.sub(pattern, repl, cleaned, flags=re.IGNORECASE)
    titled = cleaned.title()
    for acronym in ("UTCI", "PMV", "GHI", "DNI", "DHI"):
        titled = re.sub(rf"\b{acronym.title()}\b", acronym, titled)
    titled = titled.replace("Dry-Bulb", "Dry-Bulb")
    return titled


def _report_equivalent_norms(name: str) -> set[str]:
    """Return all normalized report aliases that identify the same planned figure."""
    base_norms = {_normalize_report_key(name)}
    formatted = format_figure_title(name)
    if formatted:
        base_norms.add(_normalize_report_key(formatted))

    for block in REPORT_STRUCTURE:
        for spec in block.get("figures", []):
            spec_norms = {
                _normalize_report_key(spec.get("title", "")),
                *(_normalize_report_key(alias) for alias in spec.get("aliases", [])),
            }
            spec_norms.discard("")
            if base_norms.intersection(spec_norms):
                return base_norms | spec_norms
    return {norm for norm in base_norms if norm}


def _resolve_report_sections(figs: Dict[str, object]) -> List[Dict[str, object]]:
    lookup = {_normalize_report_key(k): k for k in figs.keys()}
    used = set()
    sections: List[Dict[str, object]] = []

    for block in REPORT_STRUCTURE:
        items = []
        for spec in block.get("figures", []):
            aliases = spec.get("aliases", [])
            chosen = None
            for alias in aliases:
                nk = _normalize_report_key(alias)
                if nk in lookup and lookup[nk] not in used:
                    chosen = lookup[nk]
                    break
            if chosen:
                used.add(chosen)
                items.append({
                    "raw_key": chosen,
                    "title": spec.get("title") or format_figure_title(chosen),
                })

        if items or block.get("tab") == "Raw Data":
            sections.append({
                "tab": block["tab"],
                "section": block["section"],
                "intro": block["intro"],
                "items": items,
            })

    # Unmapped captured figures are appended to the explicit additional bucket,
    # not to the planned precipitation section.
    raw_section = next((s for s in sections if s.get("section") == "Additional Captured Figures"), None)
    if raw_section is None:
        raw_section = {"tab": "Raw Data", "section": "Additional Captured Figures", "intro": "Additional dashboard figures captured during this Streamlit session.", "items": []}
        sections.append(raw_section)
    for raw_key in figs.keys():
        if raw_key not in used:
            raw_section["items"].append({"raw_key": raw_key, "title": format_figure_title(raw_key)})

    # Hide truly empty sections except when every section is empty.
    nonempty = [s for s in sections if s.get("items")]
    return nonempty if nonempty else sections


def _figure_caption_text(clean_title: str, raw_key: str, section_name: str) -> str:
    low = _normalize_report_key(raw_key)
    title_key = _normalize_report_key(clean_title)
    caption_map = {
        "annual_diurnal_resource_heatmap": "A compact year-at-a-glance matrix for the main environmental resources. Read across the year and down the hour axis to find recurring daily windows that shape passive design, comfort exposure, and operational timing.",
        "comfort_loads": "Monthly comfort and load indicators are combined to show where overheating, heat stress, and comfort compliance concentrate. The secondary comfort-percent line should be read against the bar totals rather than as a separate climate variable.",
        "di_index_annual_heatmap": "The Discomfort Index heatmap classifies humid heat stress by day and hour. Use the legend categories to locate persistent muggy periods where dry-bulb temperature alone may understate occupant stress.",
        "utci_index_annual_heatmap": "UTCI combines air temperature, humidity, wind, and radiation into outdoor thermal stress categories. Strong bands indicate periods when outdoor exposure requires shade, wind management, or schedule adaptation.",
        "pmv_index_annual_heatmap": "PMV translates hourly thermal conditions into a thermal sensation scale. The neutral band indicates hours closest to standard comfort assumptions, while warm and cool classes identify conditioning pressure.",
        "annual_climate_statistics": "The paired temperature and humidity trends show annual timing, daily variability, and comfort-band relationships. Range bars emphasize volatility; smoothed lines reveal the seasonal signal.",
        "drybulb_monthly_bar": "Monthly dry-bulb bars summarize the selected statistic across the year, making seasonal peaks, troughs, and shoulder-period transitions visible at a glance.",
        "drybulb_hourly_dot_plot": "Hourly dry-bulb points are faceted by month to show daily spread within each season. Dense vertical clouds indicate high intra-day variability and broader control requirements.",
        "drybulb_annual_heatmap": "Dry-bulb temperature is mapped by day of year and hour of day. Horizontal warm or cool bands reveal recurring diurnal timing, while vertical shifts show seasonal progression.",
        "relhum_monthly_bar": "Monthly relative humidity bars summarize the site's moisture profile. Compare high-humidity seasons against dry shoulder months to anticipate latent loads and envelope condensation risk.",
        "relhum_hourly_dot_plot": "Hourly relative humidity points show how moisture conditions spread across the daily cycle in each month. Wide panels indicate unstable moisture conditions rather than a single monthly norm.",
        "relhum_annual_heatmap": "Relative humidity is mapped by day and hour to expose persistent moist and dry windows. This view is useful for latent-load screening and passive ventilation timing.",
        "sun_path_2d": "The 2D sun-path diagram shows solar altitude and azimuth over the selected seasonal window. It is most useful for orientation, shading geometry, and facade exposure checks.",
        "sun_path_3d": "The 3D sun-path view adds spatial context to solar position and selected environmental coloring. Use it to understand how high and low sun angles relate to massing and sky exposure.",
        "sun_path_cartesian": "The Cartesian sun-path projection converts azimuth and altitude into a design-friendly coordinate field. Analemmas, seasonal arcs, and hour labels support shading and access studies.",
        "monthly_solar_insolation": "Daily mean irradiance components are overlaid across the year. Compare GHI, DNI, and DHI to distinguish direct-sun potential from diffuse-sky contribution.",
        "irradiance_heatmap": "The irradiance heatmap maps solar intensity by day and hour. Bright, continuous bands indicate reliable solar availability; broken bands suggest seasonal or cloud-driven intermittency.",
        "dhi_irradiance_heatmap": "Diffuse horizontal irradiance is mapped by day and hour to show sky-diffuse daylight and solar availability. This is most useful where overcast or hazy conditions soften direct-sun exposure.",
        "dni_irradiance_heatmap": "Direct normal irradiance is mapped by day and hour to show beam-solar availability. Continuous high-value bands indicate stronger potential for direct gain, glare exposure, and concentrating solar systems.",
        "cloud_coverage": "Monthly sky-cover frequencies show the balance among clear, intermediate, and cloudy conditions. The stacked composition is more important than any single monthly total.",
        "cloud_coverage_scatter": "Hourly sky-cover points show how cloudiness varies within each month and hour. The smoothed overlay helps separate recurring daily structure from noisy weather events.",
        "cloud_coverage_heatmap": "Total sky cover is mapped by day and hour to reveal cloudy periods that may suppress solar gain, daylight availability, and passive heating potential.",
        "psychrometric_chart": "Hourly outdoor states are plotted in psychrometric space with comfort and strategy overlays. Dense clusters indicate dominant climate states; outlying arms show seasonal extremes.",
        "annual_wind_rose": "Wind-rose sectors show prevailing direction and speed-class frequency. Read the longest sectors first, then compare color distribution to understand whether wind is frequent, strong, or diffuse.",
        "monthly_wind_speed": "Monthly bars show mean wind speed by calendar month. Compare seasonal peaks against ventilation and exposure needs before relying on wind as a passive resource.",
        "wind_speed_frequency_distribution": "The histogram shows how often each wind-speed class occurs, with the fitted curve summarizing the annual distribution. A right-shifted tail indicates stronger exposure and greater outdoor comfort sensitivity.",
        "monthly_precipitation": "Monthly bars sum precipitation depth across the year. Peak wet months indicate drainage, facade wetting, and site-water management priorities.",
        "snowfall_profile": "The snow profile summarizes snow-depth behavior by month. Persistent winter depth points to roof-load, access, meltwater, and envelope durability concerns.",
        "heating_and_cooling_degree_days": "Degree days aggregate hourly departures from an 18 deg C base into monthly heating and cooling demand indicators. Compare HDD and CDD totals to understand whether envelope heat retention or heat rejection dominates.",
        "longwave_horizontal_irradiance_by_hour_and_day": "Heatmap of longwave (thermal) horizontal radiation by day-of-year and hour-of-day. This seasonal longwave pattern is highly relevant to nighttime radiative cooling and envelope heat loss at this location.",
        "hourly_incoming_radiation": "Monthly box-whisker chart showing DNI, DHI, GHI, and longwave radiation distributions. Use the interquartile spread and whiskers to compare direct, diffuse, global, and thermal sky conditions across seasons.",
        "monthly_solar_insolation_on_inclined_surfaces": "Overlay of total monthly insolation received on horizontal, vertical, and tilt-optimized surfaces. The optimal fixed tilt angle maximizes the annual total insolation received.",
        "mean_radiant_temperature_by_hour_and_day": "Heatmap of Mean Radiant Temperature (MRT) by day-of-year and hour-of-day, assuming full exposure to sun and wind. Peak MRT periods significantly increase heat stress and impact outdoor UTCI.",
        "full_hourly_time_series_dry_bulb_and_dew_point_temperature": "Full 8,760-point line chart comparing dry-bulb and dew-point temperatures. Persistent separation between the two traces highlights dry-air periods and evaporative cooling opportunity.",
        "full_hourly_time_series_relative_humidity": "Full 8,760-point line chart of relative humidity across the year. Reference bands help distinguish dry, moderate, and humid periods that affect comfort, condensation risk, and latent load.",
        "seasonal_psychrometric_points": "Four seasonal psychrometric scatter plots showing all hourly points in dry-bulb vs. humidity ratio space. Point clusters identify which seasons naturally fall inside the overlaid UTCI comfort zone.",
        "hourly_psychrometric_paths": "Twelve monthly mean 24-hour paths traced in psychrometric space with UTCI comfort contours overlaid. Paths reveal exactly which times of day naturally pass through the comfort zone.",
        "diurnal_thermal_comfort_frequency_shading_scenario": "Stacked bar matrix of UTCI comfort categories comparing shaded vs. unshaded conditions by season and time-of-day. Shading provides the greatest comfort benefit during peak sun exposure periods.",
        "diurnal_thermal_comfort_frequency_wind_scenario": "Stacked bar matrix comparing calm-wind vs. exposed-wind UTCI comfort frequency by season and time-of-day. Wind provides meaningful cooling benefits against specific heat stress periods.",
        "diurnal_thermal_comfort_frequency_humidity_scenario": "Stacked bar matrix comparing ambient humidity vs. neutralized humidity (RH=50%) UTCI comfort frequency. This assesses whether humidity control alone is a meaningful strategy for achieving comfort.",
        "wind_speed_by_hour_and_day": "Heatmap of wind speed (m/s) by day-of-year and hour-of-day. The seasonal and diurnal wind patterns mapped here directly dictate natural ventilation scheduling.",
        "seasonal_wind_roses": "Four seasonal wind rose polar plots showing speed bins and frequency by direction. The prevailing direction shifts across seasons indicate how to orient openings for optimal cross-ventilation.",
        "diurnal_wind_roses": "A 4x4 matrix of wind roses showing diurnal wind shifts across the four seasons. The diurnal wind shift pattern is critical for timing natural ventilation strategies.",
        "annual_directional_wind_power": "Four-panel plot showing mean wind speed, wind power density, frequency, and wind energy density by direction at 50m elevation. The most energetic wind direction guides structural and energy-harvesting design decisions.",
    }
    if low in caption_map:
        return caption_map[low]
    if title_key in caption_map:
        return caption_map[title_key]
    if "wind_rose" in low:
        return "Rose sectors encode wind direction frequency, while color or radius encodes intensity class. Read dominant sectors first, then compare spread to assess ventilation potential and exposure risk."
    if "annual_heatmap" in low or "heatmap" in low:
        return "Columns represent months (or day-of-year) and rows represent hour-of-day, so persistent hot/cool or humid/dry bands are easy to spot. This view is useful for locating recurring peak-load windows."
    if "monthly_bar" in low:
        return "Each bar summarizes the monthly central tendency for this metric. Compare shoulder months against seasonal extremes to identify transition periods and control opportunities."
    if "hourly_dot_plot" in low or "scatter" in low:
        return "Point clouds show intra-day spread and variability by month. Wider vertical spread indicates stronger hour-to-hour volatility, which matters for comfort stability and control design."
    if "psychrometric" in low:
        return "Each point is an hourly thermo-hygrometric state. Cluster position and density indicate when passive strategies are feasible and when dehumidification or sensible cooling dominates."
    if "sun_path" in low or "solar" in low or "irradiance" in low:
        return "This figure summarizes solar availability and geometry across time. Use it to evaluate orientation, shading depth, daylight potential, and photovoltaic opportunity."
    if "comfort" in low or "utci" in low or "pmv" in low or "discomfort" in low:
        return "This chart quantifies comfort-state frequency across the year and by time-of-day context. It helps target mitigation actions to the highest-risk intervals."
    return f"This figure belongs to the {section_name.lower()} section and provides supporting evidence for climate interpretation in the same sequence as the Streamlit dashboard."


def _metric_series_from_hourly(cdf: Optional[pd.DataFrame], col: Optional[str], kind: str = "") -> pd.Series:
    if cdf is None or cdf.empty or not col or col not in cdf.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(cdf[col], errors="coerce")
    if kind in {"solar", "wind", "precip", "snow"}:
        s = s.mask(s < 0)
    if kind == "solar":
        s = s.mask(s >= 9999)
    elif kind == "wind":
        s = s.mask(s >= 999)
    elif kind in {"precip", "snow"}:
        s = s.mask(s >= 900)
    elif kind == "rh":
        s = s.mask((s < 0) | (s > 100))
    elif kind == "temp":
        s = s.mask((s < -90) | (s > 80))
    return s.dropna()


def _hourly_wind_speed(cdf: Optional[pd.DataFrame]) -> pd.Series:
    col = get_metric_column(cdf, ["windspd", "wind_speed", "windspeed", "wspd", "ws"]) if cdf is not None else None
    return _metric_series_from_hourly(cdf, col, "wind")


def _hourly_ghi(cdf: Optional[pd.DataFrame]) -> pd.Series:
    col = get_metric_column(cdf, ["glohorrad", "global_hor", "global_horizontal", "ghi", "radiation", "solar", "irradiance"]) if cdf is not None else None
    return _metric_series_from_hourly(cdf, col, "solar")


def _wind_data_unavailable(wind: pd.Series) -> bool:
    return wind.empty


def _first_sentence(text: str) -> str:
    pieces = re.split(r"(?<=[.!?])\s+", str(text).strip(), maxsplit=1)
    return pieces[0].strip() if pieces and pieces[0].strip() else str(text).strip()


def _figure_interpretation_sentence(title: str, cdf: Optional[pd.DataFrame]) -> str:
    interp = _figure_interpretation_text(title, cdf).strip()
    if interp.lower().startswith("interpretation:"):
        interp = interp.split(":", 1)[1].strip()
    if interp:
        return _first_sentence(interp)
    return "For this location, use the computed pattern to size passive-design opportunities and flag HVAC risk periods before detailed simulation."


def _two_sentence_caption(clean_title: str, raw_key: str, section_name: str, cdf: Optional[pd.DataFrame]) -> str:
    dynamic_captions = st.session_state.get("pdf_captions", {})
    if dynamic_captions:
        candidate_norms = {
            _normalize_report_key(raw_key),
            _normalize_report_key(clean_title),
            _normalize_report_key(format_figure_title(raw_key)),
        }
        for key, caption in dynamic_captions.items():
            if _normalize_report_key(key) in candidate_norms and str(caption).strip():
                return str(caption).strip()

    read_sentence = _first_sentence(_figure_caption_text(clean_title, raw_key, section_name))
    interpretation = _figure_interpretation_sentence(raw_key or clean_title, cdf)
    if interpretation and not interpretation.endswith((".", "!", "?")):
        interpretation += "."
    return f"{read_sentence} {interpretation}".strip()


def _figure_interpretation_text(title: str, cdf: Optional[pd.DataFrame]) -> str:
    if cdf is None or cdf.empty:
        return ""

    try:
        t_col = get_metric_column(cdf, ["drybulb", "dry_bulb", "temp", "temperature"])
        rh_col = get_metric_column(cdf, ["relhum", "humidity", "rh"])
        w_col = get_metric_column(cdf, ["windspd", "wind_speed", "windspeed", "wspd", "ws"])
        s_col = get_metric_column(cdf, ["glohorrad", "global_hor", "global_horizontal", "ghi", "radiation", "solar", "irradiance"])
        cloud_col = get_metric_column(cdf, ["totskycvr", "sky_cover", "cloud"])

        low = _normalize_report_key(title)
        bits: List[str] = []

        wants_temp = any(token in low for token in ["temp", "drybulb", "dry_bulb", "annual_climate", "resource", "psych", "comfort", "utci", "pmv", "di_index"])
        wants_rh = any(token in low for token in ["humid", "relhum", "rh", "annual_climate", "resource", "psych", "comfort", "utci", "pmv", "di_index"])
        wants_wind = "wind" in low or "resource" in low or "utci" in low
        wants_solar = any(token in low for token in ["solar", "sun", "irradiance", "insolation", "resource"])
        wants_cloud = "cloud" in low

        if t_col and wants_temp:
            t = _metric_series_from_hourly(cdf, t_col, "temp")
            if not t.empty:
                bits.append(f"Annual dry-bulb spans {t.min():.1f} to {t.max():.1f} deg C with mean {t.mean():.1f} deg C")

        if rh_col and wants_rh:
            rh = _metric_series_from_hourly(cdf, rh_col, "rh")
            if not rh.empty:
                bits.append(f"mean relative humidity is {rh.mean():.0f}%")

        if w_col and wants_wind:
            w = _metric_series_from_hourly(cdf, w_col, "wind")
            if not w.empty:
                if _wind_data_unavailable(w):
                    bits.append("Wind speed values are zero or below threshold in this EPW file")
                else:
                    bits.append(f"wind speed averages {w.mean():.1f} m/s with 90th percentile {w.quantile(0.9):.1f} m/s")
            else:
                bits.append("Wind data not available in source file")

        if s_col and wants_solar:
            s = _metric_series_from_hourly(cdf, s_col, "solar")
            if not s.empty:
                bits.append(f"global irradiance mean is {s.mean():.0f} W/m2 with upper decile {s.quantile(0.9):.0f} W/m2")

        if cloud_col and wants_cloud:
            cld = pd.to_numeric(cdf[cloud_col], errors="coerce").dropna()
            if not cld.empty:
                bits.append(f"mean total sky cover is {cld.mean():.1f}/10")

        if bits:
            return "Interpretation: " + "; ".join(bits) + "."
    except Exception:
        pass

    return ""


def _climate_summary_lines(cdf: Optional[pd.DataFrame]) -> List[str]:
    if cdf is None or cdf.empty:
        return [
            "No hourly climate dataframe was available at export time.",
            "Section-level interpretations are therefore qualitative.",
        ]

    lines: List[str] = []
    try:
        t_col = get_metric_column(cdf, ["drybulb", "dry_bulb", "temp", "temperature"])
        rh_col = get_metric_column(cdf, ["relhum", "humidity", "rh"])
        dp_col = get_metric_column(cdf, ["dew", "dewpoint"])
        w_col = get_metric_column(cdf, ["windspd", "wind_speed", "windspeed", "wspd", "ws"])
        s_col = get_metric_column(cdf, ["glohorrad", "global_hor", "global_horizontal", "ghi", "radiation", "solar", "irradiance"])

        month_series = None
        hour_series = None
        if "month" in cdf.columns:
            month_series = pd.to_numeric(cdf["month"], errors="coerce")
        elif isinstance(cdf.index, pd.DatetimeIndex):
            month_series = pd.Series(cdf.index.month, index=cdf.index)

        if "hour" in cdf.columns:
            hour_series = pd.to_numeric(cdf["hour"], errors="coerce")
        elif isinstance(cdf.index, pd.DatetimeIndex):
            hour_series = pd.Series(cdf.index.hour, index=cdf.index)

        if t_col:
            t = pd.to_numeric(cdf[t_col], errors="coerce")
            t_valid = t.dropna()
            if not t_valid.empty:
                lines.append(f"Annual dry-bulb ranges from {t_valid.min():.1f} to {t_valid.max():.1f} deg C, with annual mean {t_valid.mean():.1f} deg C.")

                if month_series is not None:
                    by_month = t.groupby(month_series).mean().dropna()
                    if not by_month.empty:
                        cold_m = int(by_month.idxmin())
                        hot_m = int(by_month.idxmax())
                        lines.append(f"Monthly pattern: coldest mean month is {calendar.month_abbr[cold_m]} ({by_month.loc[cold_m]:.1f} deg C), warmest is {calendar.month_abbr[hot_m]} ({by_month.loc[hot_m]:.1f} deg C).")

                if hour_series is not None:
                    by_hour = t.groupby(hour_series).mean().dropna()
                    if not by_hour.empty:
                        hmin = int(by_hour.idxmin())
                        hmax = int(by_hour.idxmax())
                        lines.append(f"Diurnal pattern: mean minima occur near {hmin:02d}:00 and mean maxima near {hmax:02d}:00 local time.")

        if rh_col:
            rh = pd.to_numeric(cdf[rh_col], errors="coerce").dropna()
            if not rh.empty:
                lines.append(f"Humidity profile: annual mean relative humidity is {rh.mean():.0f}%, with broad variability indicating distinct moist and dry windows.")

        if dp_col:
            dp = pd.to_numeric(cdf[dp_col], errors="coerce").dropna()
            if not dp.empty:
                lines.append(f"Dew-point behavior spans {dp.min():.1f} to {dp.max():.1f} deg C, which is useful for condensation and latent load checks.")

        if w_col:
            w = _metric_series_from_hourly(cdf, w_col, "wind")
            if not w.empty:
                if _wind_data_unavailable(w):
                    lines.append("Wind speed values are zero or below threshold in this EPW file.")
                else:
                    lines.append(f"Wind regime: annual mean speed is {w.mean():.1f} m/s, with 90th percentile {w.quantile(0.9):.1f} m/s.")
            else:
                lines.append("Wind data not available in source file.")

        if s_col:
            s = _metric_series_from_hourly(cdf, s_col, "solar")
            if not s.empty:
                lines.append(f"Solar resource: annual mean global horizontal irradiance is {s.mean():.0f} W/m2 and upper decile is {s.quantile(0.9):.0f} W/m2.")

        if lines:
            lines.append("These annual, seasonal, monthly, and diurnal indicators should be interpreted with the figure set in each section for design-level decisions.")
    except Exception:
        pass

    return lines or ["Insufficient clean data was available to compute narrative climate summary statistics."]


def _build_additional_pdf_figures(cdf: Optional[pd.DataFrame]) -> Dict[str, object]:
    """Build required report figures directly from hourly rows when dashboard captures are absent."""
    extra: Dict[str, object] = {}
    if cdf is None or cdf.empty:
        return extra

    try:
        df = cdf.copy()
        
        comfort_pkg = st.session_state.get("comfort_pkg", {})
        if "mrt" in comfort_pkg:
            df["mrt"] = comfort_pkg["mrt"]
        if "utci" in comfort_pkg:
            df["utci"] = comfort_pkg["utci"]
            
        if "month" in df.columns:
            df["__month"] = pd.to_numeric(df["month"], errors="coerce")
        elif isinstance(df.index, pd.DatetimeIndex):
            df["__month"] = df.index.month
        else:
            df["__month"] = np.nan

        if "hour" in df.columns:
            df["__hour"] = pd.to_numeric(df["hour"], errors="coerce")
        elif isinstance(df.index, pd.DatetimeIndex):
            df["__hour"] = df.index.hour
        else:
            df["__hour"] = np.nan

        def _season_from_month(m):
            if pd.isna(m):
                return np.nan
            m = int(m)
            if m in (12, 1, 2):
                return "Winter"
            if m in (3, 4, 5):
                return "Spring"
            if m in (6, 7, 8):
                return "Summer"
            return "Autumn"

        def _tod_from_hour(h):
            if pd.isna(h):
                return np.nan
            h = int(h)
            if 0 <= h < 6:
                return "Night"
            if 6 <= h < 12:
                return "Morning"
            if 12 <= h < 18:
                return "Afternoon"
            return "Evening"

        df["__season"] = df["__month"].apply(_season_from_month)
        df["__tod"] = df["__hour"].apply(_tod_from_hour)

        t_col = get_metric_column(df, ["drybulb", "dry_bulb", "temp", "temperature"])
        rh_col = get_metric_column(df, ["relhum", "humidity", "rh"])
        dp_col = get_metric_column(df, ["dew", "dewpoint"])
        mrt_col = get_metric_column(df, ["mrt", "mean_radiant"])
        utci_col = get_metric_column(df, ["utci"]) 
        w_col = get_metric_column(df, ["wind_speed", "windspeed", "windspd", "wspd", "ws"])
        wd_col = get_metric_column(df, ["wind_direction", "winddir", "wind_dir", "wdir", "wd"])
        ghi_col = get_metric_column(df, ["glohorrad", "global_hor", "global_horizontal", "ghi", "radiation", "solar", "irradiance"])
        dhi_col = get_metric_column(df, ["difhorrad", "dhi", "diffuse_horizontal"])
        dni_col = get_metric_column(df, ["dirnorrad", "dni", "direct_normal"])
        lw_col = get_metric_column(df, ["horirsky", "longwave", "ir_hoz", "downwelling_longwave"])
        precip_col = _precip_depth_column(df)
        precip_wtr_col = _precipitable_water_column(df)
        snow_col = get_metric_column(df, ["snowdepth", "snow_depth", "snowfall", "snow"])

        if isinstance(df.index, pd.DatetimeIndex):
            df["__doy"] = df.index.dayofyear
        elif {"month", "day"}.issubset(df.columns):
            year_src = df["year"] if "year" in df.columns else pd.Series(2021, index=df.index)
            date_guess = pd.to_datetime(
                dict(
                    year=pd.to_numeric(year_src, errors="coerce").fillna(2021).astype(int),
                    month=pd.to_numeric(df["month"], errors="coerce"),
                    day=pd.to_numeric(df["day"], errors="coerce"),
                ),
                errors="coerce",
            )
            df["__doy"] = date_guess.dt.dayofyear
        else:
            df["__doy"] = np.nan

        month_names = [calendar.month_abbr[m] for m in range(1, 13)]

        def _clean_col(col: Optional[str], kind: str = "") -> pd.Series:
            return _metric_series_from_hourly(df, col, kind)

        def _month_group(col: Optional[str], kind: str = "", agg: str = "mean") -> pd.DataFrame:
            if not col:
                return pd.DataFrame(columns=["month", "label", "value"])
            local = pd.DataFrame({
                "month": pd.to_numeric(df["__month"], errors="coerce"),
                "value": _clean_col(col, kind),
            }).dropna()
            if local.empty:
                return pd.DataFrame(columns=["month", "label", "value"])
            if agg == "sum":
                grouped = local.groupby("month")["value"].sum()
            elif agg == "max":
                grouped = local.groupby("month")["value"].max()
            else:
                grouped = local.groupby("month")["value"].mean()
            out = grouped.reindex(range(1, 13)).reset_index()
            out.columns = ["month", "value"]
            out["label"] = out["month"].astype(int).map(lambda m: calendar.month_abbr[m])
            return out

        def _placeholder_fig(title: str, message: str) -> go.Figure:
            fig = go.Figure()
            fig.add_annotation(
                x=0.5, y=0.5, xref="paper", yref="paper",
                text=message, showarrow=False,
                font=dict(size=22, color=CHART_DARK_TEXT),
                align="center",
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig.update_layout(title=title, height=560)
            return fig

        # PDF export must not depend on which Streamlit dashboard tabs the user
        # happened to visit. Build the advanced publication figures directly here.
        station_name = _safe_location_label(st.session_state.get("header", {}))
        epw_metadata = _location_meta(st.session_state.get("header", {}))
        adv_df = _prepare_advanced_figure_df(df)
        _remove_manual_pdf_captions([
            "Hourly Timeseries Temperature",
            "Hourly Timeseries Relative Humidity",
            "Longwave Irradiance Heatmap",
            "Hourly Incoming Radiation Boxwhisker",
            "Inclined Surface Insolation",
        ])

        def _store_advanced_fig(title: str, builder, *args) -> None:
            try:
                fig, caption = builder(*args)
                if fig is not None:
                    extra[title] = fig
                    if caption:
                        _add_manual_pdf_caption(title, caption)
            except Exception as exc:
                extra[title] = _placeholder_fig(title, f"Figure build failed: {exc}")

        if not adv_df.empty:
            _store_advanced_fig("Longwave Horizontal Irradiance by Hour and Day", build_fig_a_longwave_heatmap, adv_df, station_name)
            _store_advanced_fig("Hourly Incoming Radiation", build_fig_b_hourly_incoming_radiation_boxwhisker, adv_df, station_name)
            _store_advanced_fig("Monthly Solar Insolation on Inclined Surfaces", build_fig_c_inclined_surface_insolation, adv_df, station_name, epw_metadata)
            _store_advanced_fig("MRT Heatmap", build_fig_d_mrt_heatmap, adv_df, station_name)
            _store_advanced_fig("Full Hourly Time Series - Dry-Bulb and Dew-Point Temperature", build_fig_e_hourly_timeseries_temperature, adv_df, station_name)
            _store_advanced_fig("Full Hourly Time Series - Relative Humidity", build_fig_f_hourly_timeseries_rh, adv_df, station_name)

            if {"drybulb_C", "rh_pct"}.issubset(adv_df.columns):
                tdb_arr = pd.to_numeric(adv_df["drybulb_C"], errors="coerce").to_numpy(float)
                rh_arr = pd.to_numeric(adv_df["rh_pct"], errors="coerce").clip(0, 100).to_numpy(float)
                wind_arr = (
                    pd.to_numeric(adv_df["wind_speed_ms"], errors="coerce").fillna(1.5).clip(lower=0.1).to_numpy(float)
                    if "wind_speed_ms" in adv_df.columns else np.full(len(adv_df), 1.5)
                )
                if "ghi_Wm2" in adv_df.columns:
                    ghi_arr = pd.to_numeric(adv_df["ghi_Wm2"], errors="coerce").fillna(0).clip(lower=0).to_numpy(float)
                    tr_arr = tdb_arr + (0.7 / 0.95) * 0.308 * ghi_arr / (4 * 5.67e-8 * np.power(tdb_arr + 273.15, 3))
                    tr_arr = np.clip(tr_arr, tdb_arr, tdb_arr + 60.0)
                else:
                    tr_arr = tdb_arr.copy()
                try:
                    utci_dict = compute_all_utci_scenarios(tdb_arr, tr_arr, wind_arr, rh_arr)
                    _store_advanced_fig("Seasonal Psychrometric Points", build_fig_g_seasonal_psychrometric, adv_df, utci_dict["baseline"], station_name)
                    _store_advanced_fig("Hourly Psychrometric Paths", build_fig_h_hourly_psychrometric_paths, adv_df, station_name)
                    _store_advanced_fig("Diurnal Thermal Comfort Frequency - Shading Scenario", build_fig_i_diurnal_comfort_shading, adv_df, utci_dict, station_name)
                    _store_advanced_fig("Diurnal Thermal Comfort Frequency - Wind Scenario", build_fig_j_diurnal_comfort_wind, adv_df, utci_dict, station_name)
                    _store_advanced_fig("Diurnal Thermal Comfort Frequency - Humidity Scenario", build_fig_k_diurnal_comfort_humidity, adv_df, utci_dict, station_name)
                except Exception as exc:
                    msg = f"UTCI scenario build failed: {exc}"
                    extra["Seasonal Psychrometric Points"] = _placeholder_fig("Seasonal Psychrometric Points", msg)
                    extra["Hourly Psychrometric Paths"] = _placeholder_fig("Hourly Psychrometric Paths", msg)
                    extra["Diurnal Thermal Comfort Frequency - Shading Scenario"] = _placeholder_fig("Diurnal Thermal Comfort Frequency - Shading Scenario", msg)
                    extra["Diurnal Thermal Comfort Frequency - Wind Scenario"] = _placeholder_fig("Diurnal Thermal Comfort Frequency - Wind Scenario", msg)
                    extra["Diurnal Thermal Comfort Frequency - Humidity Scenario"] = _placeholder_fig("Diurnal Thermal Comfort Frequency - Humidity Scenario", msg)

            wind_col_adv = next((c for c in ["wind_speed_ms", "windspd", "windspd_ms"] if c in adv_df.columns), None)
            wind_ok = False
            if wind_col_adv:
                wind_adv = pd.to_numeric(adv_df[wind_col_adv], errors="coerce")
                wind_ok = not wind_adv.dropna().empty
            if wind_ok:
                _store_advanced_fig("wind_speed_heatmap", build_fig_l, adv_df, station_name)
                _store_advanced_fig("seasonal_wind_roses", build_fig_m, adv_df, station_name)
                _store_advanced_fig("diurnal_wind_roses", build_fig_n, adv_df, station_name)
                _store_advanced_fig("directional_wind_power", build_fig_o, adv_df, station_name)
            else:
                msg = "Wind data not available."
                extra["wind_speed_heatmap"] = _placeholder_fig("Wind Speed by Hour and Day", msg)
                extra["seasonal_wind_roses"] = _placeholder_fig("Seasonal Wind Roses", msg)
                extra["diurnal_wind_roses"] = _placeholder_fig("Diurnal Wind Roses", msg)
                extra["directional_wind_power"] = _placeholder_fig("Annual Directional Wind Power", msg)

        def _annual_day_hour_heatmap(col: Optional[str], kind: str, title: str, colorscale: str, unit: str, store_key: Optional[str] = None) -> None:
            if not col:
                return
            local = pd.DataFrame({
                "hour": pd.to_numeric(df["__hour"], errors="coerce"),
                "doy": pd.to_numeric(df["__doy"], errors="coerce"),
                "value": _clean_col(col, kind),
            }).dropna()
            if local.empty:
                return
            piv = local.pivot_table(index="hour", columns="doy", values="value", aggfunc="mean")
            piv = piv.reindex(index=range(24), columns=range(1, 367))
            fig = go.Figure(
                go.Heatmap(
                    z=piv.values,
                    x=list(range(1, 367)),
                    y=list(range(24)),
                    colorscale=colorscale,
                    colorbar=dict(title=unit),
                )
            )
            fig.update_layout(title=title, xaxis_title="Day of year", yaxis_title="Hour of day", height=620)
            extra[store_key or title] = fig

        def _hourly_distribution(col: Optional[str], kind: str, title: str, y_title: str, color: str, store_key: Optional[str] = None) -> None:
            if not col:
                return
            local = pd.DataFrame({
                "month": pd.to_numeric(df["__month"], errors="coerce"),
                "hour": pd.to_numeric(df["__hour"], errors="coerce"),
                "value": _clean_col(col, kind),
            }).dropna()
            if local.empty:
                return
            local["month_name"] = local["month"].astype(int).map(lambda m: calendar.month_abbr[m])
            fig = px.scatter(
                local,
                x="hour",
                y="value",
                facet_col="month_name",
                facet_col_wrap=4,
                category_orders={"month_name": month_names},
                opacity=0.38,
                title=title,
            )
            fig.update_traces(marker=dict(size=3, color=color))
            fig.update_layout(xaxis_title="Hour of day", yaxis_title=y_title, height=780, showlegend=False)
            extra[store_key or title] = fig

        # Required temperature and humidity report figures.
        temp = _clean_col(t_col, "temp")
        rh = _clean_col(rh_col, "rh")
        if not temp.empty or not rh.empty:
            stat_rows = []
            if not temp.empty:
                stat_rows.append(("Dry-bulb temperature (deg C)", float(temp.min()), float(temp.mean()), float(temp.max())))
            if not rh.empty:
                stat_rows.append(("Relative humidity (%)", float(rh.min()), float(rh.mean()), float(rh.max())))
            if stat_rows:
                stat_df = pd.DataFrame(stat_rows, columns=["Metric", "Min", "Mean", "Max"])
                fig_stats = go.Figure()
                fig_stats.add_trace(
                    go.Bar(
                        y=stat_df["Metric"],
                        x=stat_df["Max"] - stat_df["Min"],
                        base=stat_df["Min"],
                        orientation="h",
                        name="Annual range",
                        marker_color="#2dd4bf",
                        opacity=0.52,
                    )
                )
                fig_stats.add_trace(
                    go.Scatter(
                        x=stat_df["Mean"],
                        y=stat_df["Metric"],
                        mode="markers+text",
                        text=[f"mean {v:.1f}" for v in stat_df["Mean"]],
                        textposition="top center",
                        name="Mean",
                        marker=dict(size=12, color="#f59e0b", line=dict(color="#111827", width=1)),
                    )
                )
                fig_stats.update_layout(
                    title="Annual Temperature and Humidity Statistics",
                    xaxis_title="Observed value",
                    yaxis_title="",
                    height=420,
                    barmode="overlay",
                )
                extra["Annual Climate Statistics"] = fig_stats

        if not temp.empty and t_col:
            m_temp = _month_group(t_col, "temp", "mean")
            if not m_temp.empty:
                fig = px.bar(m_temp, x="label", y="value", title="Monthly Dry-Bulb Temperature", labels={"label": "Month", "value": "Mean dry-bulb (deg C)"})
                fig.update_traces(marker_color="#f97316")
                fig.update_layout(height=460)
                extra["drybulb Monthly Bar"] = fig
            _hourly_distribution(t_col, "temp", "Hourly Dry-Bulb Temperature Distribution", "Dry-bulb (deg C)", "#fb923c", "drybulb Hourly Dot Plot")
            _annual_day_hour_heatmap(t_col, "temp", "Dry-Bulb Temperature by Hour and Day", "RdYlBu_r", "deg C", "drybulb Annual Heatmap")

        if not rh.empty and rh_col:
            m_rh = _month_group(rh_col, "rh", "mean")
            if not m_rh.empty:
                fig = px.bar(m_rh, x="label", y="value", title="Monthly Relative Humidity", labels={"label": "Month", "value": "Mean RH (%)"})
                fig.update_traces(marker_color="#38bdf8")
                fig.update_layout(height=460, yaxis_range=[0, 100])
                extra["relhum Monthly Bar"] = fig
            _hourly_distribution(rh_col, "rh", "Hourly Relative Humidity Distribution", "Relative humidity (%)", "#60a5fa", "relhum Hourly Dot Plot")
            _annual_day_hour_heatmap(rh_col, "rh", "Relative Humidity by Hour and Day", "Blues", "%", "relhum Annual Heatmap")

        # Required psychrometric chart: full hourly point cloud plus comfort/strategy overlays.
        if t_col and rh_col and not temp.empty and not rh.empty:
            ps = pd.DataFrame({
                "temp": _clean_col(t_col, "temp"),
                "rh": _clean_col(rh_col, "rh"),
                "month": pd.to_numeric(df["__month"], errors="coerce"),
            }).dropna()
            if not ps.empty:
                pressure_kpa = 101.325
                p_col = get_metric_column(df, ["atmos_pressure", "pressure", "barometric"])
                if p_col:
                    p_med = pd.to_numeric(df[p_col], errors="coerce").dropna().median()
                    if pd.notna(p_med) and p_med > 20000:
                        pressure_kpa = float(p_med) / 1000.0
                sat_kpa = 0.61078 * np.exp((17.2694 * ps["temp"]) / (ps["temp"] + 237.3))
                vapor_kpa = (ps["rh"].clip(0, 100) / 100.0) * sat_kpa
                ps["humidity_ratio"] = (0.621945 * vapor_kpa / (pressure_kpa - vapor_kpa)).clip(lower=0, upper=0.035) * 1000.0

                fig_psy = go.Figure()
                overlays = [
                    ("Solar gain", [10, 20, 20, 10], [3, 3, 9, 9], "#facc15"),
                    ("Thermal mass", [20, 34, 34, 20], [3, 3, 10, 10], "#a78bfa"),
                    ("Natural ventilation", [18, 30, 30, 18], [4, 4, 14, 14], "#34d399"),
                    ("Evaporative cooling", [28, 40, 40, 28], [4, 4, 12, 12], "#60a5fa"),
                    ("ASHRAE comfort zone", [20, 27, 27, 20], [4, 4, 12, 12], "#f8fafc"),
                ]
                for name, xs, ys, color in overlays:
                    fig_psy.add_trace(
                        go.Scatter(
                            x=xs + [xs[0]],
                            y=ys + [ys[0]],
                            mode="lines",
                            fill="toself",
                            name=name,
                            line=dict(color=color, width=1.4),
                            fillcolor=color,
                            opacity=0.16 if name != "ASHRAE comfort zone" else 0.24,
                        )
                    )
                fig_psy.add_trace(
                    go.Scatter(
                        x=ps["temp"],
                        y=ps["humidity_ratio"],
                        mode="markers",
                        name="Hourly outdoor state",
                        marker=dict(
                            size=3,
                            color=ps["month"],
                            colorscale="Turbo",
                            opacity=0.38,
                            colorbar=dict(title="Month"),
                        ),
                    )
                )
                fig_psy.update_layout(
                    title="Psychrometric Chart",
                    xaxis_title="Dry-bulb temperature (deg C)",
                    yaxis_title="Humidity ratio (g/kg dry air)",
                    height=720,
                    legend=dict(orientation="h", y=-0.18),
                )
                extra["Psychrometric Chart"] = fig_psy

        # Required wind figures, with explicit all-zero placeholder.
        wind = _clean_col(w_col, "wind")
        if not w_col:
            msg = "Wind data not available in source file."
            extra["Annual Wind Rose"] = _placeholder_fig("Annual Wind Rose", msg)
            extra["Monthly Wind Speed"] = _placeholder_fig("Monthly Mean Wind Speed", msg)
            extra["Wind Speed Frequency Distribution"] = _placeholder_fig("Wind Speed Frequency Distribution", msg)
        elif _wind_data_unavailable(wind):
            msg = "Wind speed values are zero or below threshold in this EPW file."
            extra["Annual Wind Rose"] = _placeholder_fig("Annual Wind Rose", msg)
            extra["Monthly Wind Speed"] = _placeholder_fig("Monthly Mean Wind Speed", msg)
            extra["Wind Speed Frequency Distribution"] = _placeholder_fig("Wind Speed Frequency Distribution", msg)
        elif w_col and not wind.empty:
            m_wind = _month_group(w_col, "wind", "mean")
            if not m_wind.empty:
                fig = px.bar(m_wind, x="label", y="value", title="Monthly Mean Wind Speed", labels={"label": "Month", "value": "Wind speed (m/s)"})
                fig.update_traces(marker_color="#22c55e")
                fig.update_layout(height=460)
                extra["Monthly Wind Speed"] = fig

            fig_hist = go.Figure()
            
            # Calm winds (<0.5 m/s)
            calm_wind = wind[wind < 0.5]
            calm_pct = (len(calm_wind) / len(wind)) * 100 if len(wind) > 0 else 0
            
            fig_hist.add_trace(go.Histogram(
                x=wind, nbinsx=28, histnorm="probability density", 
                name="Observed hours", marker_color="#38bdf8", opacity=0.68
            ))
            
            # Add a vertical line/span for calm winds if meaningful
            if calm_pct > 0:
                fig_hist.add_vrect(
                    x0=0, x1=0.5,
                    fillcolor="red", opacity=0.2,
                    layer="below", line_width=0,
                    annotation_text=f"Calm ({calm_pct:.1f}%)", 
                    annotation_position="top right"
                )

            if wind.mean() > 0 and wind.std() > 0:
                try:
                    # Use scipy to fit weibull min
                    params = stats.weibull_min.fit(wind[wind > 0], floc=0)
                    k = params[0]
                    lam = params[2]
                    
                    x_vals = np.linspace(0, max(float(wind.quantile(0.995)), float(wind.max()), 1.0), 160)
                    y_vals = stats.weibull_min.pdf(x_vals, *params)
                    fig_hist.add_trace(go.Scatter(
                        x=x_vals, y=y_vals, mode="lines", 
                        name=f"Weibull fit (k={k:.2f}, λ={lam:.2f})", 
                        line=dict(color="#f59e0b", width=3)
                    ))
                    fig_hist.add_annotation(
                        text=f"Weibull k={k:.2f}<br>lambda={lam:.2f} m/s",
                        x=0.98,
                        y=0.92,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        align="right",
                        bgcolor="rgba(255,255,255,0.85)",
                        bordercolor="#f59e0b",
                        font=dict(size=11, color="#111827"),
                    )
                except Exception:
                    pass
            fig_hist.update_layout(title="Wind Speed Frequency Distribution", xaxis_title="Wind speed (m/s)", yaxis_title="Probability density", height=520, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            extra["Wind Speed Frequency Distribution"] = fig_hist

        # Required precipitation, snow, and degree-day figures.
        precip = _clean_col(precip_col, "precip")
        precip_has_data = precip_col and not precip.empty and float(precip.max()) > 0
        if precip_has_data:
            m_precip = _month_group(precip_col, "precip", "sum")
            fig = px.bar(m_precip, x="label", y="value", title="Monthly Precipitation", labels={"label": "Month", "value": "Precipitation depth (mm)"})
            fig.update_traces(marker_color="#38bdf8")
            fig.update_layout(height=460)
            extra["Monthly Precipitation"] = fig
            # Computed caption with actual precipitation values.
            if not m_precip.empty and "value" in m_precip.columns:
                total_annual_mm = float(m_precip["value"].sum())
                wettest_idx = m_precip["value"].idxmax()
                wettest_month = str(m_precip.loc[wettest_idx, "label"]) if pd.notna(wettest_idx) else "[unavailable]"
                driest_idx = m_precip[m_precip["value"] > 0]["value"].idxmin() if (m_precip["value"] > 0).any() else None
                driest_month = str(m_precip.loc[driest_idx, "label"]) if driest_idx is not None else "[unavailable]"
                cap_precip = safe_format_caption(
                    "Monthly bars sum liquid precipitation depth across the year from the EPW liqprecipdepth field. "
                    "For {station}, total annual precipitation is {total_mm:.0f} mm, peaking in {wettest} and "
                    "lowest in {driest} - use this distribution to size roof drainage, assess facade water "
                    "management risk, and identify dry-season passive cooling windows.",
                    dict(station=station_name, total_mm=total_annual_mm, wettest=wettest_month, driest=driest_month))
                _add_manual_pdf_caption("Monthly Precipitation", cap_precip)
        else:
            precip_wtr = _clean_col(precip_wtr_col, "precip")
            if precip_wtr_col and not precip_wtr.empty and float(precip_wtr.max()) > 0:
                m_wtr = _month_group(precip_wtr_col, "precip", "mean")
                fig = px.bar(
                    m_wtr,
                    x="label",
                    y="value",
                    title="Monthly Precipitable Water",
                    labels={"label": "Month", "value": "Mean precipitable water (mm)"},
                )
                fig.update_traces(marker_color="#38bdf8")
                fig.update_layout(height=460, margin=dict(t=90))
                fig.add_annotation(
                    text="<i>Precipitation depth unavailable — showing atmospheric precipitable water, not rainfall.</i>",
                    x=0.5,
                    y=1.08,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=10, color="#94a3b8"),
                )
                extra["Monthly Precipitation"] = fig
            else:
                msg = (
                    "Liquid precipitation depth is not populated in this EPW file. Many TMY EPWs leave precipitation depth as all zero/missing values."
                    if precip_col
                    else "Precipitation data not available in source file."
                )
                extra["Monthly Precipitation"] = _placeholder_fig("Monthly Precipitation", msg)

        snow = _clean_col(snow_col, "snow")
        if snow_col and not snow.empty:
            m_snow = _month_group(snow_col, "snow", "max")
            fig = px.bar(m_snow, x="label", y="value", title="Snowfall Profile", labels={"label": "Month", "value": "Peak snow depth"})
            fig.update_traces(marker_color="#e0f2fe")
            fig.update_layout(height=460)
            extra["Snowfall Profile"] = fig
        else:
            extra["Snowfall Profile"] = _placeholder_fig("Snowfall Profile", "Snow-depth data not available in source file.")

        if t_col and not temp.empty:
            dd = pd.DataFrame({
                "month": pd.to_numeric(df["__month"], errors="coerce"),
                "temp": _clean_col(t_col, "temp"),
            }).dropna()
            if not dd.empty:
                dd["HDD18"] = (18.0 - dd["temp"]).clip(lower=0) / 24.0
                dd["CDD18"] = (dd["temp"] - 18.0).clip(lower=0) / 24.0
                
                dd["date"] = dd.index.date if hasattr(dd.index, "date") else dd.index
                daily_mean = dd.groupby("date")["temp"].mean()
                daily_gdd = (daily_mean - 5.0).clip(lower=0)
                
                if hasattr(dd.index, "date"):
                    daily_month = pd.to_datetime(daily_mean.index).month
                    gdd_bym = daily_gdd.groupby(daily_month).sum().reindex(range(1, 13)).fillna(0)
                else:
                    gdd_bym = pd.Series(0, index=range(1, 13))
                    
                bym = dd.groupby("month")[["HDD18", "CDD18"]].sum().reindex(range(1, 13)).fillna(0)
                bym["GDD5"] = gdd_bym.values
                
                bym["Month"] = [calendar.month_abbr[m] for m in range(1, 13)]
                annual_hdd = float(bym["HDD18"].sum())
                annual_cdd = float(bym["CDD18"].sum())
                annual_gdd = float(bym["GDD5"].sum())
                
                fig_dd = make_subplots(
                    rows=1,
                    cols=2,
                    subplot_titles=("Heating and Cooling Degree Days (Base 18°C)", "Growing Degree Days (Base 5°C)"),
                )
                fig_dd.add_trace(go.Bar(x=bym["Month"], y=bym["HDD18"], name="HDD18", marker_color="#60a5fa"), row=1, col=1)
                fig_dd.add_trace(go.Bar(x=bym["Month"], y=bym["CDD18"], name="CDD18", marker_color="#fb923c"), row=1, col=1)
                fig_dd.add_trace(go.Bar(x=bym["Month"], y=bym["GDD5"], name="GDD5", marker_color="#437a22"), row=1, col=2)
                fig_dd.add_annotation(
                    text=f"Annual GDD: {annual_gdd:.0f} °C·days",
                    x=0.98,
                    y=0.95,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    align="right",
                    font=dict(size=11),
                    bgcolor="white",
                    bordercolor="gray",
                )
                fig_dd.update_layout(title="Heating and Cooling Degree Days", barmode="group", height=520)
                fig_dd.update_yaxes(title_text="Degree days", row=1, col=1)
                fig_dd.update_yaxes(title_text="Degree days", row=1, col=2)
                extra["Heating and Cooling Degree Days"] = fig_dd
                # BUG 6: Computed caption with actual HDD, CDD, GDD values
                climate_zone = "heating-dominated" if annual_hdd > 3 * annual_cdd else ("cooling-dominated" if annual_cdd > 3 * annual_hdd else "mixed-load")
                cap_dd = safe_format_caption(
                    "Degree days aggregate hourly departures from an 18 deg C base into monthly heating and cooling "
                    "demand indicators, with growing degree days (GDD base 5 deg C) added for agricultural and "
                    "phenological context. For {station}, annual HDD18 is {hdd:.0f}, CDD18 is {cdd:.0f}, and "
                    "GDD5 is {gdd:.0f} deg C-days - placing this climate in the {zone} "
                    "zone for early-stage mechanical system sizing.",
                    dict(station=station_name, hdd=annual_hdd, cdd=annual_cdd, gdd=annual_gdd, zone=climate_zone))
                _add_manual_pdf_caption("Heating and Cooling Degree Days", cap_dd)

        def _add_month_hour_heatmap(value_col: Optional[str], title: str, colorscale: str = "Viridis"):
            if not value_col:
                return
            local = df[["__month", "__hour", value_col]].copy()
            local[value_col] = pd.to_numeric(local[value_col], errors="coerce")
            local = local.dropna(subset=["__month", "__hour", value_col])
            if local.empty:
                return
            piv = local.pivot_table(index="__hour", columns="__month", values=value_col, aggfunc="mean")
            if piv.empty:
                return
            heatmap_kwargs = {}
            if colorscale == SOLAR_COLORSCALE:
                max_val = pd.to_numeric(local[value_col], errors="coerce").max()
                zmax = float(max_val) if pd.notna(max_val) and float(max_val) > 0 else 1.0
                heatmap_kwargs = {"zmin": 0, "zmax": zmax}
            fig = go.Figure(
                data=go.Heatmap(
                    z=piv.values,
                    x=[calendar.month_abbr[int(m)] for m in piv.columns],
                    y=[int(h) for h in piv.index],
                    colorscale=colorscale,
                    colorbar=dict(title=value_col),
                    **heatmap_kwargs,
                )
            )
            fig.update_layout(title=title, xaxis_title="Month", yaxis_title="Hour of Day")
            extra[title] = fig

        # Heatmaps requested
        _add_month_hour_heatmap(t_col, "Dry Bulb Heatmap", "Turbo")
        _add_month_hour_heatmap(dp_col, "Dew Point Heatmap", "Cividis")
        _add_month_hour_heatmap(rh_col, "RH Heatmap", "Blues")
        _add_month_hour_heatmap(mrt_col, "MRT Heatmap", "RdYlBu_r")
        _add_month_hour_heatmap(utci_col, "UTCI Heatmap", "RdYlBu_r")
        
        # Fig D: UTCI Annual Time Series
        if utci_col and t_col:
            utci_ts = pd.DataFrame({
                "utci": df[utci_col], 
                "temp": df[t_col], 
                "date": df.index.date if isinstance(df.index, pd.DatetimeIndex) else df.index
            }).dropna()
            
            if not utci_ts.empty:
                utci_daily_max = utci_ts.groupby("date")["utci"].max()
                utci_daily_min = utci_ts.groupby("date")["utci"].min()
                temp_daily = utci_ts.groupby("date")["temp"].mean()
                
                fig_utci_ts = go.Figure()
                fig_utci_ts.add_trace(go.Scatter(x=temp_daily.index, y=temp_daily.values, mode="lines", name="Mean Dry Bulb", line=dict(color="#fb923c", width=1)))
                fig_utci_ts.add_trace(go.Scatter(x=utci_daily_max.index, y=utci_daily_max.values, mode="lines", name="Max UTCI", line=dict(color="#ef4444", width=2)))
                fig_utci_ts.add_trace(go.Scatter(x=utci_daily_min.index, y=utci_daily_min.values, mode="lines", name="Min UTCI", line=dict(color="#3b82f6", width=2)))
                
                fig_utci_ts.update_layout(
                    title="UTCI Annual Time Series",
                    yaxis_title="Temperature (°C)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                extra["UTCI Annual Time Series"] = fig_utci_ts
            else:
                extra["UTCI Annual Time Series"] = _placeholder_fig("UTCI Annual Time Series", "Valid UTCI/Temperature data not available.")
        else:
            extra["UTCI Annual Time Series"] = _placeholder_fig("UTCI Annual Time Series", "UTCI or Dry Bulb Temperature not available.")
        
        # Solar heatmaps
        _add_month_hour_heatmap(dhi_col, "Diffuse Horizontal Irradiance by Hour and Day", SOLAR_COLORSCALE)
        _add_month_hour_heatmap(dni_col, "Direct Normal Irradiance by Hour and Day", SOLAR_COLORSCALE)
        
        # Fig A: Longwave
        if lw_col:
            _add_month_hour_heatmap(lw_col, "Longwave Horizontal Irradiance by Hour and Day", "Magma")
        else:
            extra["Longwave Horizontal Irradiance by Hour and Day"] = _placeholder_fig("Longwave Horizontal Irradiance by Hour and Day", "Longwave radiation data not available.")
            
        # Fig B: Hourly Incoming Radiation Box Plot
        if ghi_col and dhi_col and dni_col:
            melted = []
            for col, name in [(ghi_col, "GHI"), (dni_col, "DNI"), (dhi_col, "DHI")]:
                temp_df = df[[col, "__month"]].copy()
                temp_df["Radiation Type"] = name
                temp_df.rename(columns={col: "Irradiance (W/m²)"}, inplace=True)
                melted.append(temp_df)
            if lw_col:
                temp_df = df[[lw_col, "__month"]].copy()
                temp_df["Radiation Type"] = "Longwave"
                temp_df.rename(columns={lw_col: "Irradiance (W/m²)"}, inplace=True)
                melted.append(temp_df)
            
            box_df = pd.concat(melted).dropna()
            box_df["Month"] = box_df["__month"].astype(int).apply(lambda m: calendar.month_abbr[m])
            fig_box = px.box(box_df, x="Month", y="Irradiance (W/m²)", color="Radiation Type", title="Hourly Incoming Radiation")
            fig_box.update_layout(height=600, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            extra["Hourly Incoming Radiation"] = fig_box
        else:
            extra["Hourly Incoming Radiation"] = _placeholder_fig("Hourly Incoming Radiation", "Required solar components missing.")

        # Fig C: Monthly Solar Insolation on Inclined Surfaces
        if ghi_col and dhi_col and dni_col:
            header = st.session_state.get("header", {})
            loc = header.get("location", {})
            lat = float(loc.get("latitude") or 45.0)
            lon = float(loc.get("longitude") or 0.0)
            
            optimal_tilt_az = 0 if lat < 0 else 180
            optimal_tilt_abs = abs(lat) * 0.87
            
            insolation_fig = go.Figure()
            if PVLIB_AVAILABLE:
                try:
                    tz_hours = float(loc.get("timezone") or 0.0)
                    from datetime import timezone, timedelta
                    tzinfo = timezone(timedelta(hours=tz_hours))
                    times = df.index
                    if getattr(times, 'tz', None) is None:
                        times = times.tz_localize(tzinfo)
                    
                    solpos = pvlib.solarposition.get_solarposition(times, lat, lon)
                    solpos = _drop_datetime_timezone(solpos)
                    surfaces = {
                        "Horizontal": (0, 180),
                        f"Vertical {'North' if lat < 0 else 'South'}": (90, optimal_tilt_az),
                        f"Vertical {'South' if lat < 0 else 'North'}": (90, 180 if optimal_tilt_az == 0 else 0),
                        "Vertical East": (90, 90),
                        "Vertical West": (90, 270),
                        f"Optimal Tilt ({optimal_tilt_abs:.0f}°)": (optimal_tilt_abs, optimal_tilt_az)
                    }
                    
                    month_names = [calendar.month_abbr[m] for m in range(1, 13)]
                    for name, (tilt, az) in surfaces.items():
                        irrad = pvlib.irradiance.get_total_irradiance(
                            surface_tilt=tilt,
                            surface_azimuth=az,
                            solar_zenith=pd.to_numeric(solpos['apparent_zenith'], errors="coerce").to_numpy(float),
                            solar_azimuth=pd.to_numeric(solpos['azimuth'], errors="coerce").to_numpy(float),
                            dni=pd.to_numeric(df[dni_col], errors="coerce").fillna(0).to_numpy(float),
                            ghi=pd.to_numeric(df[ghi_col], errors="coerce").fillna(0).to_numpy(float),
                            dhi=pd.to_numeric(df[dhi_col], errors="coerce").fillna(0).to_numpy(float)
                        )
                        poa = pd.Series(np.asarray(irrad['poa_global'], dtype=float), index=df.index).fillna(0)
                        monthly_insolation = poa.groupby(df["__month"]).sum() / 1000.0
                        monthly_insolation = monthly_insolation.reindex(range(1, 13)).fillna(0)
                        insolation_fig.add_trace(go.Bar(x=month_names, y=monthly_insolation, name=name))
                    
                    insolation_fig.update_layout(
                        title="Monthly Solar Insolation on Inclined Surfaces (kWh/m²)",
                        barmode="group",
                        yaxis_title="Insolation (kWh/m²)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    extra["Monthly Solar Insolation on Inclined Surfaces"] = insolation_fig
                except Exception as e:
                    extra["Monthly Solar Insolation on Inclined Surfaces"] = _placeholder_fig("Monthly Solar Insolation on Inclined Surfaces", f"Error calculating insolation: {e}")
            else:
                extra["Monthly Solar Insolation on Inclined Surfaces"] = _placeholder_fig("Monthly Solar Insolation on Inclined Surfaces", "pvlib required for inclined surface calculations.")
        else:
            extra["Monthly Solar Insolation on Inclined Surfaces"] = _placeholder_fig("Monthly Solar Insolation on Inclined Surfaces", "Required solar components missing.")

        # Degree days summary
        if t_col and "__month" in df.columns:
            tt = pd.to_numeric(df[t_col], errors="coerce")
            mm = pd.to_numeric(df["__month"], errors="coerce")
            dtmp = pd.DataFrame({"m": mm, "t": tt}).dropna()
            if not dtmp.empty:
                dtmp["hdd18"] = (18.0 - dtmp["t"]).clip(lower=0)
                dtmp["cdd18"] = (dtmp["t"] - 18.0).clip(lower=0)
                bym = dtmp.groupby("m")[["hdd18", "cdd18"]].sum().reset_index()
                bym["month"] = bym["m"].astype(int).apply(lambda m: calendar.month_abbr[m])
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Bar(x=bym["month"], y=bym["hdd18"], name="HDD18"))
                fig_dd.add_trace(go.Bar(x=bym["month"], y=bym["cdd18"], name="CDD18"))
                fig_dd.update_layout(title="Degree Day Summary", barmode="group", yaxis_title="Degree-hours")
                extra["Degree Day Summary"] = fig_dd

        # Psychrometric annual/seasonal/monthly and hourly paths
        if t_col and rh_col:
            ps = df[[t_col, rh_col, "__season", "__month", "__hour"]].copy()
            ps[t_col] = pd.to_numeric(ps[t_col], errors="coerce")
            ps[rh_col] = pd.to_numeric(ps[rh_col], errors="coerce")
            ps = ps.dropna(subset=[t_col, rh_col])
            if not ps.empty:
                sampled = ps.sample(n=min(len(ps), 4000), random_state=7) if len(ps) > 4000 else ps
                fig_pa = px.scatter(sampled, x=t_col, y=rh_col, opacity=0.35, title="Annual Psychrometric Analysis")
                fig_pa.update_layout(xaxis_title="Dry Bulb (deg C)", yaxis_title="Relative Humidity (%)")
                extra["Annual Psychrometric Analysis"] = fig_pa

                if sampled["__season"].notna().any():
                    fig_ps = px.scatter(sampled, x=t_col, y=rh_col, color="__season", opacity=0.35, title="Seasonal Psychrometric Analysis")
                    fig_ps.update_layout(xaxis_title="Dry Bulb (deg C)", yaxis_title="Relative Humidity (%)", legend_title="Season")
                    extra["Seasonal Psychrometric Analysis"] = fig_ps

                if sampled["__month"].notna().any():
                    sampled = sampled.copy()
                    sampled["__month_name"] = sampled["__month"].astype(int).map(lambda m: calendar.month_abbr[m])
                    fig_pm = px.scatter(sampled, x=t_col, y=rh_col, color="__month_name", opacity=0.28, title="Monthly Psychrometric Analysis")
                    fig_pm.update_layout(xaxis_title="Dry Bulb (deg C)", yaxis_title="Relative Humidity (%)", legend_title="Month")
                    extra["Monthly Psychrometric Analysis"] = fig_pm

                # 12-month lines for Hourly Psychrometric Paths
                hp = ps.dropna(subset=["__month", "__hour"]).groupby(["__month", "__hour"])[[t_col, rh_col]].mean().reset_index()
                if not hp.empty:
                    fig_hp = go.Figure()
                    
                    # Add simple Comfort Zone polygon (T: 20-26°C, RH: 20-80%)
                    fig_hp.add_shape(type="rect", x0=20, y0=20, x1=26, y1=80, line=dict(color="rgba(16, 185, 129, 0.5)", width=2), fillcolor="rgba(16, 185, 129, 0.1)")
                    fig_hp.add_annotation(x=23, y=82, text="Comfort Zone", showarrow=False, font=dict(color="#10b981"))
                    
                    for month in range(1, 13):
                        m_data = hp[hp["__month"] == month].sort_values("__hour")
                        if not m_data.empty:
                            m_data = pd.concat([m_data, m_data.iloc[[0]]]) # Close the path
                            fig_hp.add_trace(go.Scatter(
                                x=m_data[t_col], 
                                y=m_data[rh_col], 
                                mode="lines+markers", 
                                name=calendar.month_abbr[month],
                                line=dict(width=2),
                                marker=dict(size=6)
                            ))
                            
                    fig_hp.update_layout(
                        title="Hourly Psychrometric Paths", 
                        xaxis_title="Dry Bulb (°C)", 
                        yaxis_title="Relative Humidity (%)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    extra["Hourly Psychrometric Paths"] = fig_hp
                    
        # Comfort Scenarios (G, H, I)
        import metrics.comfort_energy as ce
        
        def _build_comfort_scenario_chart(df_scenario, title):
            # Compute UTCI
            if "glohorrad" not in df_scenario.columns and ghi_col:
                df_scenario["glohorrad"] = df_scenario[ghi_col]
            try:
                utci_scenario = ce.compute_utci_approx(df_scenario, temp_col=t_col, rh_col=rh_col, wind_col=w_col)
            except Exception as e:
                return _placeholder_fig(title, f"Could not compute scenario UTCI: {e}")
                
            cf_scen = df_scenario[["__season", "__tod"]].copy()
            cf_scen["utci"] = utci_scenario
            cf_scen = cf_scen.dropna(subset=["utci", "__season", "__tod"])
            if cf_scen.empty:
                return _placeholder_fig(title, "Scenario resulted in no valid UTCI values.")
                
            c_val = cf_scen["utci"]
            cf_scen["class"] = np.select(
                [c_val < 9, (c_val >= 9) & (c_val <= 26), c_val > 26],
                ["Cold Stress", "No thermal stress", "Heat Stress"],
                default="Other",
            )
            grp_scen = cf_scen.groupby(["__season", "__tod", "class"]).size().reset_index(name="count")
            totals_scen = grp_scen.groupby(["__season", "__tod"])["count"].transform("sum")
            grp_scen["pct"] = (grp_scen["count"] / totals_scen * 100.0).round(1)
            grp_scen["season_tod"] = grp_scen["__season"] + " - " + grp_scen["__tod"]
            fig_scen = px.bar(
                grp_scen,
                x="season_tod",
                y="pct",
                color="class",
                title=title,
                color_discrete_map={"Cold Stress": "#3b82f6", "No thermal stress": "#10b981", "Heat Stress": "#ef4444", "Other": "#888"}
            )
            fig_scen.update_layout(xaxis_title="Season and Time of Day", yaxis_title="Frequency (%)", barmode="stack", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            return fig_scen

        if t_col and rh_col and w_col and "__season" in df.columns and "__tod" in df.columns:
            # G. Shading Scenario (glohorrad = 0 -> MRT = Tdb)
            df_shade = df.copy()
            if ghi_col: df_shade[ghi_col] = 0
            df_shade["glohorrad"] = 0
            extra["Diurnal Thermal Comfort Frequency — Shading Scenario"] = _build_comfort_scenario_chart(df_shade, "Diurnal Thermal Comfort Frequency — Shading Scenario")
            
            # H. Wind Scenario (windspd = 0.5)
            df_wind = df.copy()
            df_wind[w_col] = 0.5
            extra["Diurnal Thermal Comfort Frequency — Wind Scenario"] = _build_comfort_scenario_chart(df_wind, "Diurnal Thermal Comfort Frequency — Wind Scenario")
            
            # I. Humidity Scenario (relhum = 50)
            df_hum = df.copy()
            df_hum[rh_col] = 50
            extra["Diurnal Thermal Comfort Frequency — Humidity Scenario"] = _build_comfort_scenario_chart(df_hum, "Diurnal Thermal Comfort Frequency — Humidity Scenario")
        else:
            extra["Diurnal Thermal Comfort Frequency — Shading Scenario"] = _placeholder_fig("Diurnal Thermal Comfort Frequency — Shading Scenario", "Required variables missing.")
            extra["Diurnal Thermal Comfort Frequency — Wind Scenario"] = _placeholder_fig("Diurnal Thermal Comfort Frequency — Wind Scenario", "Required variables missing.")
            extra["Diurnal Thermal Comfort Frequency — Humidity Scenario"] = _placeholder_fig("Diurnal Thermal Comfort Frequency — Humidity Scenario", "Required variables missing.")

        # Wind family: roses, matrices, distributions, profiles, directional energy density
        if w_col and not _wind_data_unavailable(_clean_col(w_col, "wind")):
            ws = pd.to_numeric(df[w_col], errors="coerce")
            wdf = df.copy()
            wdf[w_col] = ws
            wdf = wdf.dropna(subset=[w_col])
            if not wdf.empty:
                wind_data = wdf[w_col].dropna()
                fig_hist = go.Figure()
                
                calm_pct = (len(wind_data[wind_data < 0.5]) / len(wind_data)) * 100 if len(wind_data) > 0 else 0
                
                fig_hist.add_trace(go.Histogram(
                    x=wind_data, nbinsx=24, histnorm="probability density", 
                    name="Observed hours", marker_color="#38bdf8", opacity=0.68
                ))
                
                if calm_pct > 0:
                    fig_hist.add_vrect(
                        x0=0, x1=0.5, fillcolor="red", opacity=0.2, layer="below", line_width=0,
                        annotation_text=f"Calm ({calm_pct:.1f}%)", annotation_position="top right"
                    )

                if wind_data.mean() > 0 and wind_data.std() > 0:
                    try:
                        params = stats.weibull_min.fit(wind_data[wind_data > 0], floc=0)
                        k = params[0]
                        lam = params[2]
                        x_vals = np.linspace(0, max(float(wind_data.quantile(0.995)), float(wind_data.max()), 1.0), 160)
                        y_vals = stats.weibull_min.pdf(x_vals, *params)
                        fig_hist.add_trace(go.Scatter(
                            x=x_vals, y=y_vals, mode="lines", 
                            name=f"Weibull fit (k={k:.2f}, λ={lam:.2f})", 
                            line=dict(color="#f59e0b", width=3)
                        ))
                        fig_hist.add_annotation(
                            text=f"Weibull k={k:.2f}<br>lambda={lam:.2f} m/s",
                            x=0.98,
                            y=0.92,
                            xref="paper",
                            yref="paper",
                            showarrow=False,
                            align="right",
                            bgcolor="rgba(255,255,255,0.85)",
                            bordercolor="#f59e0b",
                            font=dict(size=11, color="#111827"),
                        )
                    except Exception:
                        pass
                        
                fig_hist.update_layout(
                    title="Wind Speed Frequency Distribution", 
                    xaxis_title="Wind Speed (m/s)", 
                    yaxis_title="Probability density",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                extra["Wind Speed Frequency Distribution"] = fig_hist

                hpw = wdf.dropna(subset=["__hour"]).groupby("__hour")[w_col].mean().reset_index().sort_values("__hour")
                if not hpw.empty:
                    fig_hpw = px.line(hpw, x="__hour", y=w_col, markers=True, title="Annual Hourly Wind Speed Profile")
                    fig_hpw.update_layout(xaxis_title="Hour of Day", yaxis_title="Wind Speed (m/s)")
                    extra["Annual Hourly Wind Speed Profile"] = fig_hpw

                shp = wdf.dropna(subset=["__hour", "__season"]).groupby(["__season", "__hour"])[w_col].mean().reset_index()
                if not shp.empty:
                    fig_shp = px.line(shp, x="__hour", y=w_col, color="__season", markers=False, title="Seasonal Hourly Wind Speed Profiles")
                    fig_shp.update_layout(xaxis_title="Hour of Day", yaxis_title="Wind Speed (m/s)", legend_title="Season")
                    extra["Seasonal Hourly Wind Speed Profiles"] = fig_shp

                mtx = wdf.dropna(subset=["__month", "__hour"]).pivot_table(index="__hour", columns="__month", values=w_col, aggfunc="mean")
                if not mtx.empty:
                    fig_mtx = go.Figure(
                        data=go.Heatmap(
                            z=mtx.values,
                            x=[calendar.month_abbr[int(m)] for m in mtx.columns],
                            y=[int(h) for h in mtx.index],
                            colorscale="YlGnBu",
                            colorbar=dict(title="m/s"),
                        )
                    )
                    fig_mtx.update_layout(title="Wind Speed by Hour and Day", xaxis_title="Month", yaxis_title="Hour")
                    extra["Wind Speed by Hour and Day"] = fig_mtx

                if wd_col:
                    wdf_dir = wdf.copy()
                    wd = pd.to_numeric(wdf_dir[wd_col], errors="coerce")
                    if wd.isna().all():
                        wd = wdf_dir[wd_col].astype(str).str.strip().str.upper().map(_COMPASS_TO_DEG)
                    wdf_dir[wd_col] = wd
                    wdf_dir = wdf_dir.dropna(subset=[wd_col])

                    if not wdf_dir.empty:
                        try:
                            fig_wr = create_wind_rose(wdf_dir)
                            if fig_wr is not None:
                                fig_wr.update_layout(title="Annual Wind Rose")
                                extra["Annual Wind Rose"] = fig_wr
                        except Exception:
                            pass

                        try:
                            beaufort_bins = [0, 0.5, 1.5, 3.3, 5.5, 7.9, 10.7, 13.8, 17.1, 20.7, 24.4, 28.4, 32.6, np.inf]
                            beaufort_labels = ["Calm", "Light air", "Light breeze", "Gentle breeze", "Moderate breeze", "Fresh breeze", "Strong breeze", "Near gale", "Gale", "Strong gale", "Storm", "Violent storm", "Hurricane"]
                            dir_bins = np.arange(0, 361, 22.5)
                            dir_labels = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
                            season_fig = make_subplots(
                                rows=2, cols=2,
                                specs=[[{"type": "polar"}, {"type": "polar"}], [{"type": "polar"}, {"type": "polar"}]],
                                subplot_titles=["Winter", "Spring", "Summer", "Autumn"],
                                vertical_spacing=0.08,
                                horizontal_spacing=0.06,
                            )
                            colors = px.colors.sequential.Viridis[:len(beaufort_labels)]
                            for s_idx, season in enumerate(["Winter", "Spring", "Summer", "Autumn"]):
                                subset = wdf_dir[wdf_dir["__season"] == season].copy()
                                if subset.empty:
                                    continue
                                subset["dir_cat"] = pd.cut(subset[wd_col] % 360, bins=dir_bins, labels=dir_labels, include_lowest=True, ordered=False)
                                subset["speed_cat"] = pd.cut(pd.to_numeric(subset[w_col], errors="coerce"), bins=beaufort_bins, labels=beaufort_labels, include_lowest=True, ordered=False)
                                rose = subset.groupby(["dir_cat", "speed_cat"], observed=True).size().unstack(fill_value=0).reindex(index=dir_labels, columns=beaufort_labels, fill_value=0)
                                total = rose.sum().sum()
                                if total == 0:
                                    continue
                                pct = rose / total * 100
                                row = (s_idx // 2) + 1
                                col = (s_idx % 2) + 1
                                for b_idx, speed_cat in enumerate(beaufort_labels):
                                    if pct[speed_cat].sum() <= 0:
                                        continue
                                    season_fig.add_trace(
                                        go.Barpolar(
                                            r=pct[speed_cat],
                                            theta=dir_labels,
                                            name=speed_cat,
                                            marker_color=colors[b_idx % len(colors)],
                                            marker_line_width=0,
                                            showlegend=(s_idx == 0),
                                            hovertemplate=f"Season: {season}<br>Dir: %{{theta}}<br>Speed: {speed_cat}<br>Pct: %{{r:.1f}}%<extra></extra>",
                                        ),
                                        row=row,
                                        col=col,
                                    )
                            season_fig.update_layout(height=900, margin=dict(l=50, r=50, t=100, b=50), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                            season_fig.update_annotations(yshift=10)
                            for num in range(1, 5):
                                polar_id = f"polar{num}" if num > 1 else "polar"
                                if polar_id in season_fig.layout:
                                    season_fig.layout[polar_id].radialaxis.tickfont.size = 7
                                    season_fig.layout[polar_id].angularaxis.direction = "clockwise"
                                    season_fig.layout[polar_id].angularaxis.rotation = 90
                            extra["Seasonal Wind Roses"] = season_fig
                        except Exception as e:
                            extra["Seasonal Wind Roses"] = _placeholder_fig("Seasonal Wind Roses", f"Seasonal wind rose rendering failed: {e}")

                        for season in ["Winter", "Spring", "Summer", "Autumn"]:
                            wsx = wdf_dir[wdf_dir["__season"] == season]
                            if len(wsx) < 20:
                                continue
                            try:
                                fig_swr = create_wind_rose(wsx)
                                if fig_swr is not None:
                                    fig_swr.update_layout(title=f"{season} Wind Rose")
                                    extra[f"{season} Wind Rose"] = fig_swr
                            except Exception:
                                pass

                        for tod in ["Night", "Morning", "Afternoon", "Evening"]:
                            wtx = wdf_dir[wdf_dir["__tod"] == tod]
                            if len(wtx) < 20:
                                continue
                            try:
                                fig_twr = create_wind_rose(wtx)
                                if fig_twr is not None:
                                    fig_twr.update_layout(title=f"{tod} Wind Rose")
                                    extra[f"{tod} Wind Rose"] = fig_twr
                            except Exception:
                                pass
                                
                        # 4x4 Diurnal Wind Rose Matrix
                        try:
                            max_spd = wdf_dir["Speed_mph"].max() if "Speed_mph" in wdf_dir else (pd.to_numeric(wdf_dir[w_col], errors="coerce") * 2.23694).max()
                            if pd.isna(max_spd) or max_spd <= 0: max_spd = 10
                            speed_bins = np.linspace(0, max_spd, 7)
                            speed_bins[-1] = max(speed_bins[-1], max_spd + 1e-9)
                            speed_labels = [f"{speed_bins[i]:.1f}-{speed_bins[i+1]:.1f}" for i in range(6)]
                            dir_bins = np.arange(0, 361, 22.5)
                            dir_labels = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
                            colors = px.colors.diverging.RdYlBu[::-1][:6]
                            
                            fig_matrix = make_subplots(
                                rows=4, cols=4,
                                specs=[[{"type": "polar"}] * 4] * 4,
                                subplot_titles=[f"{s} · {t}" for s in ["Winter", "Spring", "Summer", "Autumn"] for t in ["Night", "Morning", "Afternoon", "Evening"]],
                                vertical_spacing=0.05,
                                horizontal_spacing=0.03
                            )
                            
                            wdf_dir["Speed_mph"] = pd.to_numeric(wdf_dir[w_col], errors="coerce") * 2.23694
                            for i, season in enumerate(["Winter", "Spring", "Summer", "Autumn"]):
                                for j, tod in enumerate(["Night", "Morning", "Afternoon", "Evening"]):
                                    subset = wdf_dir[(wdf_dir["__season"] == season) & (wdf_dir["__tod"] == tod)].copy()
                                    if len(subset) < 10:
                                        continue
                                        
                                    subset["dir_cat"] = pd.cut(subset[wd_col], bins=dir_bins, labels=dir_labels, include_lowest=True, ordered=False)
                                    subset["speed_cat"] = pd.cut(subset["Speed_mph"], bins=speed_bins, labels=speed_labels, include_lowest=True, ordered=False)
                                    
                                    wind_data = subset.groupby(["dir_cat", "speed_cat"], observed=True).size().unstack(fill_value=0).reindex(index=dir_labels, columns=speed_labels, fill_value=0)
                                    total = wind_data.sum().sum()
                                    if total == 0: continue
                                    wind_pct = wind_data / total * 100
                                    
                                    for k, speed_cat in enumerate(speed_labels):
                                        show_leg = (i == 0 and j == 0)
                                        fig_matrix.add_trace(
                                            go.Barpolar(
                                                r=wind_pct[speed_cat].reindex(dir_labels, fill_value=0),
                                                theta=dir_labels,
                                                name=f"{speed_cat} mph",
                                                marker_color=colors[k],
                                                marker_line_width=0,
                                                showlegend=show_leg,
                                                hovertemplate=f"Season: {season}<br>ToD: {tod}<br>Dir: %{{theta}}<br>Speed: {speed_cat} mph<br>Pct: %{{r:.1f}}%<extra></extra>"
                                            ),
                                            row=i+1, col=j+1
                                        )
                                        
                            fig_matrix.update_layout(
                                height=1400,
                                width=1400,
                                margin=dict(l=60, r=60, t=120, b=60),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            fig_matrix.update_annotations(font_size=8, yshift=8)
                            for num in range(1, 17):
                                polar_id = f"polar{num}" if num > 1 else "polar"
                                if polar_id in fig_matrix.layout:
                                    fig_matrix.layout[polar_id].radialaxis.showticklabels = False
                                    fig_matrix.layout[polar_id].radialaxis.ticks = ""
                                    fig_matrix.layout[polar_id].angularaxis.direction = "clockwise"
                                    fig_matrix.layout[polar_id].angularaxis.rotation = 90
                                    fig_matrix.layout[polar_id].angularaxis.tickfont.size = 8
                                
                            extra["Diurnal Wind Roses"] = fig_matrix
                        except Exception as e:
                            extra["Diurnal Wind Roses"] = _placeholder_fig("Diurnal Wind Roses Matrix", f"Matrix rendering failed: {e}")

                        bins = np.arange(0, 361, 30)
                        labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins) - 1)]
                        wpd = wdf_dir[[wd_col, w_col]].copy()
                        wpd["dir_bin"] = pd.cut(wpd[wd_col] % 360, bins=bins, labels=labels, include_lowest=True)
                        wpd["power"] = np.power(pd.to_numeric(wpd[w_col], errors="coerce"), 3)
                        grp = wpd.groupby("dir_bin", observed=True)["power"].mean().dropna()
                        if not grp.empty:
                            fig_wpd = go.Figure(
                                data=go.Barpolar(
                                    theta=list(grp.index.astype(str)),
                                    r=list(grp.values),
                                    marker_color=list(grp.values),
                                    marker_colorscale="Viridis",
                                )
                            )
                            fig_wpd.update_layout(title=None, margin=dict(t=90))
                            extra["Annual Directional Wind Power"] = fig_wpd
    except Exception:
        return extra

    for alias_key, title_key in {
        "wind_speed_heatmap": "Wind Speed by Hour and Day",
        "seasonal_wind_roses": "Seasonal Wind Roses",
        "diurnal_wind_roses": "Diurnal Wind Roses",
        "directional_wind_power": "Annual Directional Wind Power",
    }.items():
        if alias_key in extra and title_key in extra:
            extra.pop(title_key, None)

    # Keep advanced additions bounded so PDF export remains stable on Cloud.
    preferred_order = [
        "Annual Climate Statistics",
        "drybulb Monthly Bar", "drybulb Hourly Dot Plot", "drybulb Annual Heatmap",
        "relhum Monthly Bar", "relhum Hourly Dot Plot", "relhum Annual Heatmap",
        "Longwave Horizontal Irradiance by Hour and Day",
        "Hourly Incoming Radiation",
        "Monthly Solar Insolation on Inclined Surfaces",
        "MRT Heatmap",
        "Full Hourly Time Series - Dry-Bulb and Dew-Point Temperature",
        "Full Hourly Time Series - Relative Humidity",
        "Psychrometric Chart",
        "Seasonal Psychrometric Points",
        "Hourly Psychrometric Paths",
        "Diurnal Thermal Comfort Frequency - Shading Scenario",
        "Diurnal Thermal Comfort Frequency - Wind Scenario",
        "Diurnal Thermal Comfort Frequency - Humidity Scenario",
        "Annual Wind Rose", "Monthly Wind Speed", "Wind Speed Frequency Distribution",
        "wind_speed_heatmap",
        "seasonal_wind_roses",
        "diurnal_wind_roses",
        "directional_wind_power",
        "Monthly Precipitation", "Snowfall Profile", "Heating and Cooling Degree Days",
    ]
    max_extra_figs = len(preferred_order)
    ordered = [name for name in preferred_order if name in extra]
    for k in extra.keys():
        if k not in ordered:
            ordered.append(k)
    trimmed = {k: extra[k] for k in ordered[:max_extra_figs]}
    return trimmed


def _get_extra_figures():
    cdf = st.session_state.get("cdf")
    if cdf is None or cdf.empty:
        return {}
    
    file_id = id(cdf)
    if st.session_state.get("_extra_figs_id") != file_id:
        with st.spinner("Generating advanced analytical figures (this may take a moment)..."):
            st.session_state["_extra_figs"] = _build_additional_pdf_figures(cdf)
            st.session_state["_extra_figs_id"] = file_id
    return st.session_state.get("_extra_figs", {})


def _captured_figure_by_alias(*aliases: str) -> Optional[go.Figure]:
    alias_norms = {_normalize_report_key(alias) for alias in aliases if str(alias).strip()}
    for store_name in ("pdf_figures", "pdf_figures_auto"):
        store = st.session_state.get(store_name, {}) or {}
        for key, fig in store.items():
            key_norms = {
                _normalize_report_key(key),
                _normalize_report_key(format_figure_title(str(key))),
            }
            if alias_norms.intersection(key_norms):
                try:
                    return _clone_dashboard_figure(fig)
                except Exception:
                    return fig
    return None


def _render_captured_context_figure(label: str, aliases: List[str], source_note: str) -> None:
    fig = _captured_figure_by_alias(label, *aliases)
    if fig is None:
        st.info(f"{label} is not captured yet. Visit its primary dashboard tab or generate the PDF report once to populate this cross-reference.")
        return
    st.caption(source_note)
    _st_plotly_chart(fig, use_container_width=True, key=f"context_{_normalize_report_key(label)}")


def _month_series_for_cdf(cdf: pd.DataFrame, index: pd.Index) -> pd.Series:
    if isinstance(index, pd.DatetimeIndex):
        return pd.Series(index.month, index=index)
    if "month" in cdf.columns:
        return pd.to_numeric(cdf["month"], errors="coerce").reindex(index)
    return pd.Series(np.nan, index=index)


def _drop_datetime_timezone(obj):
    try:
        if getattr(obj.index, "tz", None) is not None:
            obj = obj.copy()
            obj.index = obj.index.tz_localize(None)
    except Exception:
        try:
            obj.index = obj.index.tz_convert(None)
        except Exception:
            pass
    return obj


def _column_by_normalized_name(df: pd.DataFrame, aliases: Iterable[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    norm_to_col = {
        re.sub(r"[^a-z0-9]+", "", str(col).lower()): col
        for col in df.columns
    }
    for alias in aliases:
        col = norm_to_col.get(re.sub(r"[^a-z0-9]+", "", str(alias).lower()))
        if col:
            return col
    return None


def _precip_depth_column(df: pd.DataFrame) -> Optional[str]:
    return _column_by_normalized_name(
        df,
        [
            "liq_precip_depth",
            "liqprecipdepth",
            "liquid_precipitation_depth",
            "liquidprecipitationdepth",
            "precip_depth",
            "precipdepth",
            "precipitation",
            "rainfall",
            "rain",
            "precip",
        ],
    )


def _precipitable_water_column(df: pd.DataFrame) -> Optional[str]:
    return _column_by_normalized_name(df, ["precip_wtr", "precipwtr", "precipitable_water"])


def _monthly_precipitation_dashboard_fig(cdf: Optional[pd.DataFrame]) -> go.Figure:
    if cdf is None or cdf.empty:
        return placeholder_figure("Precipitation data not available in source file.")

    precip_col = _precip_depth_column(cdf)
    precip = _metric_series_from_hourly(cdf, precip_col, "precip")
    plot_title = "Monthly Precipitation"
    y_title = "Precipitation depth (mm)"
    grouped_agg = "sum"

    if precip.empty or float(precip.max()) <= 0:
        fallback_col = _precipitable_water_column(cdf)
        fallback = _metric_series_from_hourly(cdf, fallback_col, "precip")
        if fallback.empty or float(fallback.max()) <= 0:
            return placeholder_figure(
                "Liquid precipitation depth is not populated in this EPW file. "
                "Many TMY EPWs leave precipitation depth as all zero/missing values."
            )
        precip = fallback
        plot_title = "Monthly Precipitable Water"
        y_title = "Mean precipitable water (mm)"
        grouped_agg = "mean"

    month = _month_series_for_cdf(cdf, precip.index)
    local = pd.DataFrame({"month": month, "value": precip}).dropna()
    grouped_src = local.groupby("month")["value"]
    grouped = (
        grouped_src.sum()
        if grouped_agg == "sum"
        else grouped_src.mean()
    ).reindex(range(1, 13), fill_value=0)
    labels = [calendar.month_abbr[m] for m in range(1, 13)]
    fig = px.bar(x=labels, y=grouped.values, labels={"x": "Month", "y": y_title}, title=plot_title)
    fig.update_traces(marker_color="#38bdf8")
    fig.update_layout(height=420, margin=dict(t=90))
    if grouped_agg == "mean":
        fig.add_annotation(
            text="<i>Precipitation depth unavailable — showing atmospheric precipitable water, not rainfall.</i>",
            x=0.5,
            y=1.08,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=10, color="#94a3b8"),
        )
    return fig


def _snowfall_profile_dashboard_fig(cdf: Optional[pd.DataFrame]) -> go.Figure:
    if cdf is None or cdf.empty:
        return placeholder_figure("Snow-depth data not available in source file.")
    snow_col = get_metric_column(cdf, ["snowdepth", "snow_depth", "snowfall", "snow"])
    snow = _metric_series_from_hourly(cdf, snow_col, "snow")
    if not snow_col or snow.empty:
        return placeholder_figure("Snow-depth data not available in source file.")
    month = _month_series_for_cdf(cdf, snow.index)
    local = pd.DataFrame({"month": month, "value": snow}).dropna()
    grouped = local.groupby("month")["value"].max().reindex(range(1, 13), fill_value=0)
    labels = [calendar.month_abbr[m] for m in range(1, 13)]
    fig = px.bar(x=labels, y=grouped.values, labels={"x": "Month", "y": "Peak snow depth"}, title="Snowfall Profile")
    fig.update_traces(marker_color="#e0f2fe")
    fig.update_layout(height=420)
    return fig


def _degree_day_dashboard_fig(cdf: Optional[pd.DataFrame]) -> go.Figure:
    if cdf is None or cdf.empty:
        return placeholder_figure("Dry-bulb temperature is required for degree-day calculations.")
    t_col = get_metric_column(cdf, ["drybulb", "dry_bulb", "temp", "temperature"])
    temp = _metric_series_from_hourly(cdf, t_col, "temp")
    if not t_col or temp.empty:
        return placeholder_figure("Dry-bulb temperature is required for degree-day calculations.")
    if isinstance(temp.index, pd.DatetimeIndex):
        daily = temp.resample("D").mean().dropna()
        month = pd.Series(daily.index.month, index=daily.index)
        hdd = (18.0 - daily).clip(lower=0).groupby(month).sum()
        cdd = (daily - 18.0).clip(lower=0).groupby(month).sum()
        gdd = (daily - 5.0).clip(lower=0).groupby(month).sum()
    else:
        month = _month_series_for_cdf(cdf, temp.index)
        hourly = pd.DataFrame({"month": month, "temp": temp}).dropna()
        hdd = (18.0 - hourly["temp"]).clip(lower=0).groupby(hourly["month"]).sum() / 24.0
        cdd = (hourly["temp"] - 18.0).clip(lower=0).groupby(hourly["month"]).sum() / 24.0
        gdd = (hourly["temp"] - 5.0).clip(lower=0).groupby(hourly["month"]).sum() / 24.0
    labels = [calendar.month_abbr[m] for m in range(1, 13)]
    fig = go.Figure()
    for name, values, color in [
        ("HDD18", hdd, "#60a5fa"),
        ("CDD18", cdd, "#fb923c"),
        ("GDD5", gdd, "#22c55e"),
    ]:
        fig.add_bar(name=name, x=labels, y=values.reindex(range(1, 13), fill_value=0).values, marker_color=color)
    fig.update_layout(title="Heating and Cooling Degree Days", barmode="group", height=460, yaxis_title="Degree days")
    return fig


def _monthly_wind_speed_dashboard_fig(cdf: Optional[pd.DataFrame]) -> go.Figure:
    if cdf is None or cdf.empty:
        return placeholder_figure("Wind data not available in source file.")
    wind_col = get_metric_column(cdf, WIND_SPEED_ALIASES)
    if not wind_col:
        return placeholder_figure("Wind data not available in source file.")
    wind = _metric_series_from_hourly(cdf, wind_col, "wind")
    if _wind_data_unavailable(wind):
        return placeholder_figure("Wind speed values are zero or below threshold in this EPW file.")
    month = _month_series_for_cdf(cdf, wind.index)
    local = pd.DataFrame({"month": month, "value": wind}).dropna()
    grouped = local.groupby("month")["value"].mean().reindex(range(1, 13), fill_value=0)
    labels = [calendar.month_abbr[m] for m in range(1, 13)]
    fig = px.bar(x=labels, y=grouped.values, labels={"x": "Month", "y": "Wind speed (m/s)"}, title="Monthly Mean Wind Speed")
    fig.update_traces(marker_color="#22c55e")
    fig.update_layout(height=420)
    return fig


def _wind_speed_frequency_dashboard_fig(cdf: Optional[pd.DataFrame]) -> go.Figure:
    if cdf is None or cdf.empty:
        return placeholder_figure("Wind data not available in source file.")
    wind_col = get_metric_column(cdf, WIND_SPEED_ALIASES)
    if not wind_col:
        return placeholder_figure("Wind data not available in source file.")
    wind = _metric_series_from_hourly(cdf, wind_col, "wind")
    if _wind_data_unavailable(wind):
        return placeholder_figure("Wind speed values are zero or below threshold in this EPW file.")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=wind, nbinsx=28, histnorm="probability density", name="Observed hours", marker_color="#38bdf8", opacity=0.68))
    positive = wind[wind > 0]
    if len(positive) > 20 and float(positive.std()) > 0:
        try:
            params = stats.weibull_min.fit(positive, floc=0)
            k = params[0]
            lam = params[2]
            x_vals = np.linspace(0, max(float(wind.quantile(0.995)), float(wind.max()), 1.0), 160)
            y_vals = stats.weibull_min.pdf(x_vals, *params)
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name=f"Weibull fit (k={k:.2f}, lambda={lam:.2f})", line=dict(color="#f59e0b", width=3)))
        except Exception:
            pass
    fig.update_layout(title="Wind Speed Frequency Distribution", xaxis_title="Wind speed (m/s)", yaxis_title="Probability density", height=460, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def render_precipitation_thermal_load_page() -> None:
    cdf = st.session_state.get("cdf")
    fig_precip = _monthly_precipitation_dashboard_fig(cdf)
    _st_plotly_chart(fig_precip, use_container_width=True, key="precip_context_monthly_precip")
    _add_manual_pdf_figure("Monthly Precipitation", fig_precip)
    fig_snow = _snowfall_profile_dashboard_fig(cdf)
    _st_plotly_chart(fig_snow, use_container_width=True, key="precip_context_snowfall")
    _add_manual_pdf_figure("Snowfall Profile", fig_snow)

    # Precipitation Heatmap
    precip_col = _precip_depth_column(cdf)
    if precip_col:
        _render_heatmap(cdf, precip_col, "Precipitation Annual Heatmap (Hour x Day)", "mm", "Blues")

def _koppen_classification(cdf: Optional[pd.DataFrame], header: Optional[dict] = None) -> Dict[str, str]:
    header = header or {}
    estimated = "estimated"
    try:
        if cdf is None or cdf.empty:
            raise ValueError("no dataframe")
        t_col = get_metric_column(cdf, ["drybulb", "dry_bulb", "temp", "temperature"])
        p_col = _precip_depth_column(cdf)
        if not t_col:
            raise ValueError("no temperature")
        if "month" in cdf.columns:
            months = pd.to_numeric(cdf["month"], errors="coerce")
        elif isinstance(cdf.index, pd.DatetimeIndex):
            months = pd.Series(cdf.index.month, index=cdf.index)
        else:
            raise ValueError("no months")
        temp = _metric_series_from_hourly(cdf, t_col, "temp")
        t_month = pd.DataFrame({"m": months, "t": temp}).dropna().groupby("m")["t"].mean()
        if t_month.empty:
            raise ValueError("no monthly temp")
        coldest = float(t_month.min())
        warmest = float(t_month.max())
        annual_precip = None
        if p_col:
            precip = _metric_series_from_hourly(cdf, p_col, "precip")
            p_month = pd.DataFrame({"m": months, "p": precip}).dropna().groupby("m")["p"].sum()
            if not p_month.empty:
                annual_precip = float(p_month.sum())
                estimated = "computed from hourly EPW temperature and precipitation"

        if annual_precip is not None and annual_precip < 350 and warmest > 10:
            code, name = "BSk", "Cold semi-arid climate"
            implication = "Prioritize solar control, durable water-wise site strategies, and night-flush opportunities where diurnal cooling is available."
        elif warmest < 10:
            code, name = "ET", "Tundra climate"
            implication = "Envelope heat retention, snow management, and cold-weather detailing dominate passive design decisions."
        elif coldest <= 0 and warmest > 10:
            hot_summer = "a" if warmest >= 22 else "b"
            code, name = f"Df{hot_summer}", "Humid continental climate"
            implication = "Design must balance winter heat retention with summer solar control and shoulder-season ventilation."
        elif 0 < coldest < 18 and warmest > 10:
            hot_summer = "a" if warmest >= 22 else "b"
            code, name = f"Cf{hot_summer}", "Temperate humid climate"
            implication = "Moisture control, adaptive shading, and mixed-mode ventilation windows are primary design levers."
        elif coldest >= 18:
            code, name = "Aw", "Tropical savanna climate"
            implication = "Shading, ventilation, latent-load control, and rain protection generally outweigh heating strategies."
        else:
            code, name = "Cfb", "Temperate oceanic climate"
            implication = "Moderate temperatures support mixed-mode operation, but humidity and seasonal solar access still need careful control."
    except Exception:
        loc = (header or {}).get("location", {}) if isinstance(header, dict) else {}
        lat = loc.get("lat") if isinstance(loc, dict) else None
        try:
            lat_abs = abs(float(lat))
        except Exception:
            lat_abs = 35.0
        code, name = ("Dfb", "Humid continental climate") if lat_abs > 40 else ("Cfa", "Humid subtropical climate")
        implication = "Classification is estimated from geographic context; verify against monthly temperature and precipitation before publication."
        estimated = "estimated from geographic context"

    return {"code": code, "name": name, "implication": implication, "basis": estimated}


def build_design_strategy_summary_page(pdf: ClimateReportPDF, cdf: Optional[pd.DataFrame]) -> None:
    pdf.current_section = "Design Strategy Summary"
    pdf.add_page()
    margin = 16
    content_w = pdf.w - 2 * margin
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*PDF_INK)
    pdf.set_xy(margin, 28)
    pdf.cell(content_w, 8, "Design Strategy Summary", ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*PDF_MUTED)
    pdf.set_xy(margin, 41)
    pdf.multi_cell(content_w, 5.2, _pdf_safe_text("Computed implications from the hourly weather profile, placed after psychrometrics to connect climate state-space patterns to architectural decisions."))

    t_col = get_metric_column(cdf, ["drybulb", "dry_bulb", "temp", "temperature"]) if cdf is not None else None
    rh_col = get_metric_column(cdf, ["relhum", "humidity", "rh"]) if cdf is not None else None
    temp = _metric_series_from_hourly(cdf, t_col, "temp")
    rh = _metric_series_from_hourly(cdf, rh_col, "rh")

    rows = []
    if not temp.empty:
        passive_heating = float((temp.between(8, 18)).mean() * 100)
        ventilation = temp.between(18, 26)
        if not rh.empty:
            ventilation = ventilation & rh.reindex(temp.index).between(30, 70).fillna(False)
        vent_hours = int(ventilation.sum())
        hot_months = []
        if isinstance(cdf.index, pd.DatetimeIndex):
            hot_by_month = temp[temp > 30].groupby(temp[temp > 30].index.month).size()
            hot_months = [calendar.month_abbr[int(m)] for m, v in hot_by_month.items() if v >= 24]
        hdd = float((18.0 - temp).clip(lower=0).sum() / 24.0)
        cdd = float((temp - 18.0).clip(lower=0).sum() / 24.0)
        ratio = hdd / max(cdd, 1.0)
        if ratio >= 3:
            priority = "High insulation priority; heating demand dominates cooling."
        elif ratio <= 0.7:
            priority = "Cooling and solar-control priority; heat rejection dominates."
        else:
            priority = "Balanced envelope: pair insulation with solar control and mixed-mode controls."
        implication = (
            f"With {hdd:.0f} HDD18 and {cdd:.0f} CDD18, this location should be treated as "
            f"{'heating-led' if ratio >= 3 else 'cooling-led' if ratio <= 0.7 else 'mixed-load'} in early design."
        )
        rows = [
            ("Passive Heating Potential", f"{passive_heating:.0f}% of hours fall in the cool-but-usable passive heating band."),
            ("Natural Ventilation Windows", f"{vent_hours:,} hours meet the temperature/humidity screening window."),
            ("Overheating Risk Months", ", ".join(hot_months) if hot_months else "No month has at least 24 hours above 30 deg C."),
            ("Insulation Priority", priority),
            ("Key Design Implication", implication),
        ]
    else:
        rows = [("Data availability", "Dry-bulb temperature was unavailable, so design strategy metrics could not be computed.")]

    y = 62
    for label, value in rows:
        pdf.set_fill_color(*PDF_SOFT_BG)
        pdf.set_draw_color(*PDF_FAINT)
        pdf.rect(margin, y, content_w, 20, "F")
        pdf.rect(margin, y, content_w, 20)
        pdf.set_xy(margin + 6, y + 4)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*PDF_ACCENT)
        pdf.cell(content_w - 12, 4, _pdf_safe_text(label.upper()), ln=1)
        pdf.set_xy(margin + 6, y + 10)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*PDF_INK)
        pdf.multi_cell(content_w - 12, 4.8, _pdf_safe_text(value))
        y += 25


def build_data_quality_notes_page(pdf: ClimateReportPDF, cdf: Optional[pd.DataFrame], header: dict, source: str, generated_on: str) -> None:
    pdf.current_section = "Data Quality & Source Notes"
    pdf.add_page()
    margin = 16
    content_w = pdf.w - 2 * margin
    loc = _location_meta(header)
    rows = len(cdf) if cdf is not None else 0
    cols = list(cdf.columns) if cdf is not None else []
    period = str(header.get("data_periods") or header.get("period") or "EPW annual weather period")
    if cdf is not None and isinstance(cdf.index, pd.DatetimeIndex) and len(cdf.index):
        period = f"{cdf.index.min().strftime('%Y-%m-%d %H:%M')} to {cdf.index.max().strftime('%Y-%m-%d %H:%M')}"
    missing_cols = []
    zero_cols = []
    if cdf is not None and not cdf.empty:
        for col in cols:
            s = pd.to_numeric(cdf[col], errors="coerce")
            if s.notna().sum() == 0:
                missing_cols.append(col)
            elif s.notna().sum() > 24 and bool((s.dropna().abs() <= 1e-9).all()):
                zero_cols.append(col)
    notes = st.session_state.get("_last_epw_notes") or []
    gap_fill = "; ".join(str(n) for n in notes) if notes else "No explicit gap-fill notes were recorded beyond standard numeric cleaning."
    qa_rows = [
        ("Source file and format", f"{source}; EPW/ZIP weather source parsed into hourly dataframe."),
        ("Data period of record", period),
        ("Rows and fields", f"{rows:,} hourly rows; {len(cols)} fields."),
        ("Missing-value columns", ", ".join(missing_cols[:18]) if missing_cols else "No fully missing numeric columns detected."),
        ("Zero-value columns", ", ".join(zero_cols[:18]) if zero_cols else "No all-zero numeric columns detected."),
        ("Gap-fill methods", gap_fill),
        ("WMO station ID", str(loc.get("wmo", "--"))),
        ("Coordinate accuracy note", f"Latitude {loc.get('lat', '--')}, longitude {loc.get('lon', '--')}; coordinates are inherited from the source EPW metadata/station index."),
        ("Report generation credit", f"Generated {generated_on} by Climate Analysis Pro from the Streamlit climate analysis workspace."),
    ]

    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*PDF_INK)
    pdf.set_xy(margin, 28)
    pdf.cell(content_w, 8, "Data Quality & Source Notes", ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*PDF_MUTED)
    pdf.set_xy(margin, 41)
    pdf.multi_cell(content_w, 5.2, _pdf_safe_text("Appendix notes documenting source, completeness, and assumptions used by this generated databook."))

    y = 60
    for label, value in qa_rows:
        if y > pdf.h - 32:
            pdf.add_page()
            y = 24
        pdf.set_xy(margin, y)
        pdf.set_font("Helvetica", "B", 8.8)
        pdf.set_text_color(*PDF_ACCENT)
        pdf.cell(content_w, 4, _pdf_safe_text(label.upper()), ln=1)
        pdf.set_xy(margin + 3, y + 5)
        pdf.set_font("Helvetica", "", 9.3)
        pdf.set_text_color(*PDF_INK)
        pdf.multi_cell(content_w - 6, 4.5, _pdf_safe_text(value))
        y = pdf.get_y() + 5


def _pdf_safe_text(t: str) -> str:
    s = str(t)
    replacements = {
        "â€”": "-", "â€“": "-", "—": "-", "–": "-",
        "â€œ": '"', "â€": '"', "“": '"', "”": '"',
        "â€˜": "'", "â€™": "'", "‘": "'", "’": "'",
        "Â°": " deg ", "°": " deg ",
        "Â²": "2", "²": "2",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s.encode("latin-1", "replace").decode("latin-1")


def _pdf_section_label(section_name: str) -> str:
    label = str(section_name or "").strip()
    if label.lower() in {"undefined", "none", "null"}:
        return ""
    if label == "Precipitation & Thermal Load":
        return "Precipitation Thermal Load"
    return label


def build_cover_page(pdf: ClimateReportPDF, location_label: str, source: str, loc_meta: Dict[str, str], generated_on: str) -> None:
    pdf.current_section = "Cover"
    pdf.add_page()
    pdf.set_fill_color(255, 255, 255)
    pdf.rect(0, 0, pdf.w, pdf.h, "F")

    large_page = _pdf_is_large_page(pdf)
    landscape_page = pdf.w > pdf.h
    if landscape_page and not large_page:
        x0 = 24
        accent_x = 16
        top_y = 34
        title_y = 49
        copy_y = 82
        location_y = 108
        meta_y = 55
        note_y = pdf.h - 28
        content_w = min(pdf.w * 0.52, 150)
        meta_x = x0 + content_w + 18
        meta_w = max(86, pdf.w - meta_x - x0)
    else:
        x0 = 42 if large_page else 28
        accent_x = 26 if large_page else 18
        top_y = 56 if large_page else 42
        title_y = 74 if large_page else 58
        copy_y = 122 if large_page else 88
        location_y = 152 if large_page else 112
        meta_y = 218 if large_page else 148
        note_y = pdf.h - (54 if large_page else 53)
        content_w = min(pdf.w - (2 * x0), 300 if large_page else 158)
        meta_x = x0
        meta_w = min(content_w, 300 if large_page else 154)

    # Full-height teal stripe with a white separator to keep the cover content readable.
    pdf.set_fill_color(*PDF_ACCENT)
    stripe_w = 16 if not large_page else 22
    pdf.rect(0, 0, stripe_w, pdf.h, "F")
    pdf.set_fill_color(255, 255, 255)
    pdf.rect(stripe_w, 0, 2, pdf.h, "F")
    pdf.set_draw_color(*PDF_ACCENT)
    pdf.set_line_width(0.25)
    pdf.line(x0 - 2, pdf.h - (84 if large_page else 65), min(pdf.w - x0, x0 + content_w), pdf.h - (84 if large_page else 65))

    include_branding = bool(st.session_state.get("export_include_branding", True))
    white_label = bool(st.session_state.get("export_white_label", False))
    report_title = str(st.session_state.get("export_report_title") or "Climate Analysis Report").strip()
    if white_label:
        eyebrow = "TECHNICAL CLIMATE REPORT"
    elif include_branding:
        eyebrow = "ARCHITECTURAL AND ENVIRONMENTAL DATABOOK"
    else:
        eyebrow = "CLIMATE INTELLIGENCE REPORT"

    pdf.set_font("Helvetica", "B", 13 if large_page else 10)
    pdf.set_text_color(*PDF_ACCENT)
    pdf.set_xy(x0, top_y)
    pdf.cell(content_w, 5, _pdf_safe_text(eyebrow), ln=1)

    pdf.set_text_color(*PDF_INK)
    pdf.set_font("Helvetica", "B", 48 if large_page else 36)
    pdf.set_xy(x0, title_y)
    pdf.multi_cell(content_w, 15 if large_page else 12, _pdf_safe_text(report_title.replace(" Report", "\nReport", 1)))

    pdf.set_font("Helvetica", "", 15 if large_page else 12)
    pdf.set_text_color(*PDF_MUTED)
    pdf.set_xy(x0, copy_y)
    desc = (
        "Technical climate analysis generated from the loaded weather dataset."
        if white_label
        else "A dashboard-faithful technical report generated from the climate analysis workspace."
    )
    pdf.multi_cell(content_w, 7.5 if large_page else 6, _pdf_safe_text(desc))

    pdf.set_font("Helvetica", "B", 21 if large_page else 16)
    pdf.set_text_color(*PDF_INK)
    pdf.set_xy(x0, location_y)
    pdf.multi_cell(content_w, 10 if large_page else 8, _pdf_safe_text(location_label))

    y_pos = meta_y
    meta_items = [
        ("Station", f"{loc_meta.get('city', '--')}, {loc_meta.get('country', '--')}"),
        ("Weather Source", source[:88]),
        ("WMO Station", loc_meta.get("wmo", "--")),
        ("Coordinates", f"{loc_meta.get('lat', '--')} deg N, {loc_meta.get('lon', '--')} deg E"),
        ("Elevation", loc_meta.get("elev", "--")),
        ("Timezone", loc_meta.get("tz", "--")),
        ("Generated", generated_on),
    ]

    pdf.set_fill_color(*PDF_SOFT_BG)
    meta_h = 78 if large_page else 67
    pdf.rect(meta_x, y_pos - 7, meta_w, meta_h, "F")
    pdf.set_draw_color(*PDF_FAINT)
    pdf.rect(meta_x, y_pos - 7, meta_w, meta_h)
    for label, val in meta_items:
        pdf.set_xy(meta_x + 8, y_pos)
        pdf.set_font("Helvetica", "B", 9 if large_page else 8)
        pdf.set_text_color(*PDF_MUTED)
        pdf.cell(54 if large_page else 39, 6, _pdf_safe_text(label.upper()), ln=0)
        pdf.set_font("Helvetica", "", 12.5 if large_page else 11)
        pdf.set_text_color(*PDF_INK)
        pdf.cell(meta_w - (76 if large_page else 56), 6, _pdf_safe_text(str(val)), ln=1)
        y_pos += 9.6 if large_page else 8.5

    pdf.set_font("Helvetica", "", 10 if large_page else 8.5)
    pdf.set_text_color(*PDF_MUTED)
    pdf.set_xy(x0, note_y)
    pdf.multi_cell(content_w, 5.5, "Figures are exported from the same styled Plotly objects rendered in the dashboard, preserving the app's chart theme and visual ordering.")


def _estimate_toc_pages(pdf: ClimateReportPDF, sections: List[Dict[str, object]], figure_count: int) -> int:
    """Mirror the fixed-height ToC renderer so downstream page numbers stay aligned."""
    toc_top = 28
    line_h = 5.1
    y = toc_top + 8 + 5 + 4
    pages = 1

    def ensure_space() -> None:
        nonlocal y, pages
        if y > pdf.h - 22:
            pages += 1
            y = toc_top

    ensure_space()
    y += line_h + 1
    for _section in sections:
        ensure_space()
        y += line_h

    y += 3
    ensure_space()
    y += line_h + 1
    for _ in range(figure_count):
        ensure_space()
        y += line_h
    return pages


def build_toc(pdf: ClimateReportPDF, sections: List[Dict[str, object]], figure_rows: List[Tuple[int, str, str, int]], section_page_map: Dict[str, int]) -> None:
    """Render the table of contents below the running header on every ToC page."""
    toc_top = 28
    line_h = 5.1
    margin = 16
    content_w = pdf.w - (2 * margin)
    page_w = 28
    label_w = content_w - page_w

    pdf.current_section = "Contents"
    pdf.add_page()
    pdf.set_xy(margin, toc_top)

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*PDF_INK)
    pdf.cell(content_w, 8, "Contents", ln=1)
    pdf.set_x(margin)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*PDF_MUTED)
    pdf.cell(content_w, 5, "Sections and figures follow the same order as the Streamlit dashboard tabs.", ln=1)
    y = pdf.get_y() + 4

    def ensure_space() -> None:
        nonlocal y
        if y > pdf.h - 22:
            pdf.add_page()
            pdf.set_xy(margin, toc_top)
            y = toc_top

    def render_block_heading(label: str) -> None:
        nonlocal y
        ensure_space()
        pdf.set_xy(margin, y)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*PDF_ACCENT)
        pdf.cell(content_w, line_h, label, ln=1)
        y = pdf.get_y() + 1

    def render_toc_line(label: str, page_no, bold: bool = False) -> None:
        nonlocal y
        ensure_space()
        pdf.set_xy(margin, y)
        pdf.set_font("Helvetica", "B" if bold else "", 9 if bold else 8.5)
        pdf.set_text_color(*PDF_INK if bold else PDF_MUTED)
        clean_label = _pdf_safe_text(str(label))
        if len(clean_label) > 90:
            clean_label = clean_label[:87] + "..."
        pdf.cell(label_w, line_h, clean_label, ln=0)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(*PDF_MUTED)
        pdf.set_x(margin + label_w)
        pdf.cell(page_w, line_h, f"p. {page_no}", ln=1, align="R")
        y = pdf.get_y()

    render_block_heading("Sections")
    for section in sections:
        section_name = str(section.get("section", ""))
        render_toc_line(section_name, section_page_map.get(section_name, "-"), bold=True)

    y += 3
    render_block_heading("Figures")
    for fnum, _section, clean_title, page_no in figure_rows:
        render_toc_line(f"Figure {fnum}. {clean_title}", page_no, bold=False)


def build_section_page(pdf: ClimateReportPDF, tab_name: str, section_name: str, intro: str, figure_count: int) -> None:
    pdf.current_section = _pdf_section_label(section_name)
    pdf.add_page()
    margin = 16
    content_w = pdf.w - (2 * margin)
    # Teal accent bar below header
    pdf.set_fill_color(*PDF_ACCENT)
    pdf.rect(margin, 20, content_w, 8, "F")
    # Section label in bar
    pdf.set_xy(margin + 4, 21)
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(content_w - 8, 5, _pdf_safe_text(f"DASHBOARD TAB  \u2014  {tab_name.upper()}"), ln=1)
    # Keep teal vertical left bar
    pdf.set_draw_color(*PDF_ACCENT)
    pdf.set_line_width(0.9)
    pdf.line(18, 30, 18, 140)
    # Section title - large, left-aligned
    pdf.set_xy(30, 34)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(*PDF_INK)
    pdf.multi_cell(content_w - 30, 11, _pdf_safe_text(section_name))
    # Section description
    pdf.set_xy(30, pdf.get_y() + 3)
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(*PDF_MUTED)
    pdf.multi_cell(content_w - 30, 5.5, _pdf_safe_text(intro))
    # Thin rule
    pdf.set_draw_color(*PDF_RULE)
    pdf.set_line_width(0.2)
    rule_y = pdf.get_y() + 6
    pdf.line(30, rule_y, pdf.w - margin, rule_y)
    # Figure count tag
    pdf.set_xy(30, rule_y + 4)
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_text_color(*PDF_ACCENT)
    pdf.cell(content_w - 30, 5, _pdf_safe_text(f"{figure_count} figure{'s' if figure_count != 1 else ''} captured from this dashboard section"), ln=1)


def _png_dimensions(path: str) -> Tuple[int, int]:
    try:
        with open(path, "rb") as f:
            header = f.read(24)
        if header[:8] == b"\x89PNG\r\n\x1a\n":
            return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")
    except Exception:
        pass
    return REPORT_EXPORT_WIDTH, REPORT_EXPORT_HEIGHT


def _fit_dimensions(src_w: int, src_h: int, max_w: float, max_h: float) -> Tuple[float, float]:
    if src_w <= 0 or src_h <= 0:
        return max_w, max_h
    ratio = min(max_w / float(src_w), max_h / float(src_h))
    return src_w * ratio, src_h * ratio


def render_figure_page(
    pdf: ClimateReportPDF,
    fig_no: int,
    section_name: str,
    clean_title: str,
    raw_key: str,
    fig: object,
    cdf: Optional[pd.DataFrame],
    temp_images: List[str],
) -> None:
    display_section = _pdf_section_label(section_name)
    pdf.current_section = display_section
    pdf.add_page()
    large_page = _pdf_is_large_page(pdf)
    landscape_page = pdf.w > pdf.h
    margin = 22 if large_page else (18 if landscape_page else 16)
    content_w = pdf.w - (2 * margin)
    section_y = 30 if not landscape_page else 26
    title_y = 40 if large_page else (36 if landscape_page else 40)

    # Teal figure label tag
    tag_y = section_y - 1
    pdf.set_fill_color(*PDF_HIGHLIGHT)
    pdf.rect(margin, tag_y, content_w, 7, "F")
    pdf.set_xy(margin + 3, tag_y + 0.5)
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_text_color(*PDF_ACCENT)
    tag_text = f"{display_section.upper()}   -   FIGURE {fig_no}" if display_section else f"FIGURE {fig_no}"
    pdf.cell(content_w - 6, 5.5, _pdf_safe_text(tag_text), ln=1)
    # Figure title
    pdf.set_font("Helvetica", "B", 21 if large_page else 15)
    pdf.set_text_color(*PDF_INK)
    pdf.set_xy(margin, title_y)
    pdf.multi_cell(content_w, 8.6 if large_page else 6.8, _pdf_safe_text(f"Figure {fig_no}. {clean_title}"))

    export_err = ""
    img_path = None
    try:
        if large_page:
            export_w, export_h, export_scale = 2200, 1320, 1
        elif landscape_page:
            export_w, export_h, export_scale = 1600, 960, 1
        else:
            export_w, export_h, export_scale = REPORT_EXPORT_WIDTH, REPORT_EXPORT_HEIGHT, REPORT_EXPORT_SCALE
        img_path = _fig_to_tmp_png(fig, width=export_w, height=export_h, scale=export_scale)
        temp_images.append(img_path)
    except Exception as exc:
        export_err = str(exc)

    image_box_x = margin
    image_box_y = 68 if large_page else (54 if landscape_page else 60)
    image_box_w = content_w
    caption_reserved = 74 if large_page else (46 if landscape_page else 102)
    image_box_h = max(92 if landscape_page else 132, pdf.h - image_box_y - caption_reserved)
    pdf.set_fill_color(*PDF_SOFT_BG)
    pdf.set_draw_color(*PDF_FAINT)
    pdf.rect(image_box_x, image_box_y, image_box_w, image_box_h, "F")
    pdf.rect(image_box_x, image_box_y, image_box_w, image_box_h)

    if img_path:
        src_w, src_h = _png_dimensions(img_path)
        pad = 10 if large_page else (7 if landscape_page else 8)
        draw_w, draw_h = _fit_dimensions(src_w, src_h, image_box_w - pad, image_box_h - pad)
        draw_x = image_box_x + (image_box_w - draw_w) / 2
        draw_y = image_box_y + (image_box_h - draw_h) / 2
        pdf.image(img_path, x=draw_x, y=draw_y, w=draw_w, h=draw_h)
    else:
        pdf.set_xy(image_box_x + 4, image_box_y + image_box_h / 2)
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(*PDF_MUTED)
        pdf.cell(image_box_w - 8, 6, "Visualization rendering unavailable.", ln=1)
        if export_err:
            pdf.set_xy(image_box_x + 4, image_box_y + image_box_h / 2 + 8)
            pdf.set_font("Helvetica", "", 7)
            pdf.set_text_color(*PDF_MUTED)
            pdf.multi_cell(image_box_w - 8, 4, _pdf_safe_text(export_err[:240]))

    caption_y = image_box_y + image_box_h + 9
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*PDF_ACCENT)
    pdf.set_xy(margin, caption_y)
    pdf.cell(content_w, 4, "CAPTION", ln=1)
    pdf.set_xy(margin, caption_y + 5)
    pdf.set_font("Helvetica", "", 10.5 if large_page else (8.4 if landscape_page else 9.2))
    pdf.set_text_color(*PDF_INK)
    caption = _two_sentence_caption(clean_title, raw_key, section_name, cdf)
    pdf.multi_cell(content_w, 5.3 if large_page else (4.2 if landscape_page else 4.8), _pdf_safe_text(caption))


def build_climate_pdf() -> bytes:
    header = st.session_state.get("header", {})
    cdf = st.session_state.get("cdf")
    location_label = _pdf_safe_text(_safe_location_label(header))

    loc_meta = {k: _pdf_safe_text(v) for k, v in _location_meta(header).items()}
    source = _pdf_safe_text(st.session_state.get("source_label", "EPW File"))
    figs = _merged_pdf_figures()
    derived_figs = _build_additional_pdf_figures(cdf)
    existing_norms = set()
    existing_fingerprints = set()
    for existing_key, existing_fig in figs.items():
        existing_norms.update(_report_equivalent_norms(existing_key))
        try:
            existing_fingerprints.add(existing_fig.to_json())
        except Exception:
            pass
    for key, fig in derived_figs.items():
        key_norms = _report_equivalent_norms(key)
        try:
            fig_fingerprint = fig.to_json()
        except Exception:
            fig_fingerprint = ""
        if key_norms.isdisjoint(existing_norms) and (not fig_fingerprint or fig_fingerprint not in existing_fingerprints):
            figs[key] = _clone_dashboard_figure(fig)
            existing_norms.update(key_norms)
            if fig_fingerprint:
                existing_fingerprints.add(fig_fingerprint)
    generated_on = _pdf_safe_text(datetime.date.today().strftime("%B %d, %Y"))

    pdf = ClimateReportPDF(location_label=location_label, source_label=source, generated_on=generated_on)
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)

    sections = _resolve_report_sections(figs)

    figure_count = sum(len(section.get("items", [])) for section in sections)
    toc_pages = _estimate_toc_pages(pdf, sections, figure_count)
    figure_rows: List[Tuple[int, str, str, int]] = []
    section_page_map: Dict[str, int] = {}
    next_page = 1 + toc_pages + 1  # cover + ToC pages + climate summary page
    fig_no = 1
    for section in sections:
        section_name = str(section.get("section", ""))
        items = section.get("items", [])
        next_page += 1
        section_page_map[section_name] = next_page
        for item in items:
            next_page += 1
            figure_rows.append((fig_no, section_name, str(item.get("title", "Figure")), next_page))
            fig_no += 1
        if section_name == "Psychrometrics":
            next_page += 1

    build_cover_page(pdf, location_label, source, loc_meta, generated_on)
    build_toc(pdf, sections, figure_rows, section_page_map)

    # Climate summary page
    pdf.current_section = "Climate Summary"
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*PDF_INK)
    pdf.set_xy(16, 28)
    pdf.cell(178, 9, "Climate Summary", ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*PDF_MUTED)
    pdf.set_xy(16, 41)
    pdf.multi_cell(168, 5.5, _pdf_safe_text("Annual, monthly, and diurnal indicators derived from the loaded hourly dataset. Use these notes as orientation; the figure pages carry the detailed evidence."))
    pdf.set_draw_color(*PDF_RULE)
    pdf.set_line_width(0.22)
    pdf.line(16, 58, 194, 58)

    y = 68
    for line in _climate_summary_lines(cdf):
        if y > pdf.h - 34:
            pdf.add_page()
            y = 20
        pdf.set_fill_color(*PDF_SOFT_BG)
        pdf.set_draw_color(*PDF_FAINT)
        pdf.rect(16, y - 1, 178, 14, "F")
        pdf.rect(16, y - 1, 178, 14)
        pdf.set_xy(22, y + 2)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*PDF_INK)
        pdf.multi_cell(166, 4.7, _pdf_safe_text(line))
        y = max(pdf.get_y() + 3.0, y + 17)

    koppen = _koppen_classification(cdf, header)
    if y > pdf.h - 52:
        pdf.add_page()
        y = 24
    pdf.set_fill_color(236, 253, 245)
    pdf.set_draw_color(*PDF_FAINT)
    pdf.rect(16, y, min(178, pdf.w - 32), 34, "F")
    pdf.rect(16, y, min(178, pdf.w - 32), 34)
    pdf.set_xy(22, y + 4)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*PDF_ACCENT)
    pdf.cell(166, 4, "CLIMATE CLASSIFICATION", ln=1)
    pdf.set_xy(22, y + 10)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*PDF_INK)
    pdf.cell(166, 5, _pdf_safe_text(f"{koppen['code']} - {koppen['name']} ({koppen['basis']})"), ln=1)
    pdf.set_xy(22, y + 18)
    pdf.set_font("Helvetica", "", 9.2)
    pdf.set_text_color(*PDF_INK)
    pdf.multi_cell(166, 4.5, _pdf_safe_text(koppen["implication"]))

    # Section + figure pages
    import gc
    temp_images: List[str] = []
    fig_no = 1
    # On cloud, cap total figures to avoid cumulative OOM.
    _MAX_CLOUD_FIGURES = 25
    _total_figures_rendered = 0
    for section in sections:
        tab_name = str(section.get("tab", ""))
        section_name = str(section.get("section", ""))
        intro = str(section.get("intro", ""))
        items = section.get("items", [])
        build_section_page(pdf, tab_name, section_name, intro, len(items))

        for item in items:
            if _IS_STREAMLIT_CLOUD and _total_figures_rendered >= _MAX_CLOUD_FIGURES:
                break
            raw_key = str(item.get("raw_key", ""))
            clean_title = str(item.get("title", format_figure_title(raw_key)))
            fig = figs.get(raw_key)
            render_figure_page(pdf, fig_no, section_name, clean_title, raw_key, fig, cdf, temp_images)
            fig_no += 1
            _total_figures_rendered += 1
            # Reclaim memory between figure exports to stay under Streamlit Cloud limits.
            gc.collect()
        if section_name == "Psychrometrics":
            build_design_strategy_summary_page(pdf, cdf)

    # Appendix source notes, then definitions.
    build_data_quality_notes_page(pdf, cdf, header, source, generated_on)

    # Appendix (definitions)
    glossary_items = [
        ("UTCI", "Universal Thermal Climate Index; an equivalent temperature index combining air temperature, radiation, humidity, and wind to estimate outdoor thermal stress."),
        ("Mean Radiant Temperature (MRT)", "The uniform temperature of an imaginary enclosure in which the radiant heat transfer from the human body is equal to the radiant heat transfer in the actual non-uniform enclosure."),
        ("Psychrometric Chart", "A state-space chart linking dry-bulb temperature and moisture content, used to evaluate comfort zones and passive strategy windows."),
        ("Degree Days", "Aggregated temperature departures from a base setpoint used to approximate seasonal heating and cooling demand."),
        ("Beaufort Scale", "A categorical description of wind strength based on typical observed effects at the surface (No. 0-12, corresponding to specific m/s ranges)."),
        ("Wind Power Density", "Available wind energy flux per unit swept area, proportional to air density and the cube of wind speed."),
        ("Wind Power Classes", "A classification system (Class 1-7) representing the wind resource potential, based on wind power density ranges (W/m²) at specific elevations."),
        ("Weibull Distribution", "A continuous probability distribution often used to model wind speed frequencies. Formula: ƒ(x;λ,k) = (k/λ)(x/λ)^(k-1) exp(-(x/λ)^k)."),
        ("TMY / CWEC Data Disclaimer", "Typical Meteorological Year (TMY) and Canadian Weather for Energy Calculations (CWEC) datasets represent typical long-term conditions rather than extreme historical events. They are intended for energy simulation and general climate analysis, not for designing against extreme localized weather events.")
    ]
    pdf.current_section = "Glossary"
    glossary_margin = 16
    glossary_top = 28
    glossary_w = pdf.w - (2 * glossary_margin)

    def _add_glossary_page(with_heading: bool = False) -> float:
        pdf.add_page()
        pdf.set_xy(glossary_margin, glossary_top)
        if not with_heading:
            return float(glossary_top)
        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(*PDF_INK)
        pdf.cell(glossary_w, 8, "Glossary", ln=1)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*PDF_MUTED)
        pdf.set_xy(glossary_margin, 40)
        pdf.cell(glossary_w, 5, "Terms and definitions used throughout the report.", ln=1)
        pdf.set_draw_color(*PDF_RULE)
        pdf.set_line_width(0.22)
        pdf.line(glossary_margin, 55, pdf.w - glossary_margin, 55)
        return 66.0

    def _estimated_glossary_height(term: str, definition: str) -> float:
        usable_chars = 96
        lines = max(1, math.ceil(len(_pdf_safe_text(definition)) / usable_chars))
        return 6 + (lines * 5) + 5

    def _render_glossary_entry(term: str, definition: str, y_pos: float) -> float:
        pdf.set_xy(glossary_margin, y_pos)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*PDF_INK)
        pdf.cell(glossary_w, 5, _pdf_safe_text(term), ln=1)
        pdf.set_xy(glossary_margin + 4, y_pos + 6)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*PDF_MUTED)
        pdf.multi_cell(glossary_w - 8, 4.8, _pdf_safe_text(definition))
        return pdf.get_y() + 4

    y = _add_glossary_page(with_heading=True)
    disclaimer_item = glossary_items[-1]
    for term, definition in glossary_items[:-1]:
        if y + _estimated_glossary_height(term, definition) > pdf.h - 22:
            y = _add_glossary_page(with_heading=False)
        y = _render_glossary_entry(term, definition, y)

    if y + 40 > pdf.h - 22:
        y = _add_glossary_page(with_heading=False)
    y = _render_glossary_entry(disclaimer_item[0], disclaimer_item[1], y)

    out = pdf.output(dest="S")

    # Ensure temp chart images are cleaned up.
    for p in temp_images:
        try:
            os.remove(p)
        except Exception:
            pass

    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    return str(out).encode("latin-1", errors="replace")



def _ui_escape(value: object) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def _nav_jump_button(label: str, page: str, *, key: str, disabled: bool = False) -> None:
    if st.button(label, key=key, use_container_width=True, disabled=disabled):
        st.session_state["nav_page"] = page
        st.session_state["active_page"] = "select_station" if page == DEFAULT_PAGE else "dashboard"
        _rerun()


def render_overview_page():
    cdf = st.session_state.get("cdf")
    header = st.session_state.get("header") or {}
    if cdf is None:
        st.info("Load an EPW file to open the climate intelligence overview.")
        return

    loc = header.get("location", {}) if isinstance(header, dict) else {}
    location_label = _safe_location_label(header)
    source_label = st.session_state.get("source_label") or "EPW weather file"
    focus_threshold = float(st.session_state.get("custom_overheat_threshold", 30))

    try:
        tmin, tmax = cdf.index.min(), cdf.index.max()
        data_window = f"{tmin:%b %d, %Y} to {tmax:%b %d, %Y}"
    except Exception:
        data_window = "EPW annual weather period"

    rows = len(cdf)
    coverage = float(cdf.notna().mean().mean() * 100) if rows else 0.0
    missing_total = int(cdf.isna().sum().sum()) if rows else 0

    st.markdown(
        f"""
        <section class="cc-hero-panel">
            <div>
                <p class="cc-eyebrow">Climate intelligence workspace</p>
                <h1>{_ui_escape(location_label)}</h1>
                <p class="cc-hero-copy">
                    A guided EPW analysis flow for climate-responsive design, comfort analytics,
                    solar strategy, wind assessment, and presentation-ready reporting.
                </p>
            </div>
            <div class="cc-hero-meta">
                <span>{_ui_escape(source_label)}</span>
                <span>{_ui_escape(data_window)}</span>
                <span>{rows:,} hourly records</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(6)

    if "drybulb" in cdf:
        temp = pd.to_numeric(cdf["drybulb"], errors="coerce")
        metric_cols[0].metric("Mean dry bulb", format_temperature(temp.mean()))
        hot_hours = int((temp > focus_threshold).sum())
        metric_cols[1].metric(f"Hours > {format_temperature(focus_threshold, digits=0)}", f"{hot_hours:,} h")
    else:
        metric_cols[0].metric("Mean dry bulb", "--")
        metric_cols[1].metric("Hot hours", "--")

    if "relhum" in cdf:
        rh = pd.to_numeric(cdf["relhum"], errors="coerce")
        metric_cols[2].metric("Mean humidity", f"{rh.mean():.0f} %")
    else:
        metric_cols[2].metric("Mean humidity", "--")

    if "windspd" in cdf:
        wind = pd.to_numeric(cdf["windspd"], errors="coerce")
        metric_cols[3].metric("Mean wind", f"{wind.mean():.1f} m/s")
    else:
        metric_cols[3].metric("Mean wind", "--")

    if "glohorrad" in cdf:
        ghi = pd.to_numeric(cdf["glohorrad"], errors="coerce").clip(lower=0)
        metric_cols[4].metric("Annual GHI", f"{ghi.sum() / 1000:.0f} kWh/m2")
    else:
        metric_cols[4].metric("Annual GHI", "--")

    if "drybulb" in cdf:
        comfort_temp = pd.to_numeric(cdf["drybulb"], errors="coerce")
        comfort_share = ((comfort_temp >= 18) & (comfort_temp <= 26)).mean() * 100
        metric_cols[5].metric("18-26 C hours", f"{comfort_share:.0f} %")
    else:
        metric_cols[5].metric("18-26 C hours", "--")

    st.markdown("<div class='section-gap-lg'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <section class="cc-panel">
            <div class="cc-panel-head">
                <h3>Site And File Summary</h3>
                <p>Use this as the orientation layer before opening the Dashboard tabs.</p>
            </div>
            <div class="cc-summary-grid">
                <div><span>Location</span><strong>{_ui_escape(location_label)}</strong></div>
                <div><span>Latitude</span><strong>{_ui_escape(loc.get('latitude', '--'))}</strong></div>
                <div><span>Longitude</span><strong>{_ui_escape(loc.get('longitude', '--'))}</strong></div>
                <div><span>Elevation</span><strong>{_ui_escape(loc.get('elevation_m', '--'))} m</strong></div>
                <div><span>Timezone</span><strong>{_ui_escape(loc.get('timezone', '--'))}</strong></div>
                <div><span>WMO</span><strong>{_ui_escape(loc.get('wmo', '--'))}</strong></div>
                <div><span>Coverage</span><strong>{coverage:.1f} %</strong></div>
                <div><span>Missing values</span><strong>{missing_total:,}</strong></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-gap-lg'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <section class="cc-panel">
            <div class="cc-panel-head">
                <h3>Workspace Sections</h3>
                <p>Use the sidebar to move between the map, dashboard tabs, forecasts, live-data tools, and exports.</p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    flow_cols = st.columns(4)
    flow_items = [
        ("Dashboard", "Climate, comfort, solar, psychrometric, wind, and raw-data tabs"),
        ("Predictions", "Short-term forecast and future climate scenarios"),
        ("Live Data", "EPW comparison and sensor workflows"),
        ("Export", "PDF report plus chart-level SVG, HTML, and CSV outputs"),
    ]
    for col, (title, desc) in zip(flow_cols, flow_items):
        col.markdown(
            f"""
            <div class="cc-mini-card">
                <strong>{_ui_escape(title)}</strong>
                <span>{_ui_escape(desc)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_climate_page():
    st.markdown(
        """
        <section class="cc-page-intro">
            <p class="cc-eyebrow">Climate</p>
            <h1>Temperature, Humidity, And Diurnal Patterns</h1>
            <p>Use the trend views for seasonal behavior, then move into full-width heatmaps
            when you need the annual hourly fingerprint.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    trend_tab, temperature_tab, heatmap_tab, humidity_tab = st.tabs(
        ["Trends", "Temperature", "Heatmaps", "Humidity"]
    )
    with trend_tab:
        render_trends_page()
    with temperature_tab:
        render_temperature_page()
    with heatmap_tab:
        render_heatmap_page()
    with humidity_tab:
        render_humidity_page()
def build_fig_d_mrt_heatmap(df, station_name):
    df = _prepare_advanced_figure_df(df)
    missing = [c for c in ["drybulb_C", "ghi_Wm2"] if c not in df.columns]
    if missing:
        return placeholder_figure("Dry-bulb and solar radiation are required for MRT analysis."), ""

    sigma = 5.67e-8
    drybulb = pd.to_numeric(df['drybulb_C'], errors="coerce")
    ghi = pd.to_numeric(df['ghi_Wm2'], errors="coerce").clip(lower=0).fillna(0)
    mrt = (drybulb + 273.15 + 0.25 * (ghi / sigma).clip(lower=0) ** 0.25) - 273.15
    mat = _advanced_day_hour_matrix(df, mrt)
    if mat.empty:
        return placeholder_figure("Not enough hourly data to build MRT heatmap."), ""
    fig = go.Figure(data=go.Heatmap(
        z=mat.values,
        x=mat.columns,
        y=mat.index,
        colorscale='RdBu_r',
        colorbar=dict(title='°C')
    ))
    fig.update_layout(
        title="Mean Radiant Temperature by Hour and Day",
        xaxis_title="Day of Year",
        yaxis_title="Hour of Day",
        yaxis=dict(autorange="reversed"),
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_mrt = mrt.groupby(df.index.month).mean() if isinstance(df.index, pd.DatetimeIndex) else pd.Series(dtype=float)
    peak_mrt_month_idx = monthly_mrt.idxmax() if not monthly_mrt.empty else np.nan
    peak_mrt_month = months[peak_mrt_month_idx - 1] if pd.notna(peak_mrt_month_idx) else "[data unavailable]"
    
    hourly_mrt = mrt.groupby(df.index.hour).mean() if isinstance(df.index, pd.DatetimeIndex) else pd.Series(dtype=float)
    hourly_peak = hourly_mrt.idxmax() if not hourly_mrt.empty else np.nan
    peak_mrt_hour_start = int(hourly_peak) - 1 if pd.notna(hourly_peak) else 0
    peak_mrt_hour_end = peak_mrt_hour_start + 2
    
    mrt_delta = round((mrt.max() - drybulb.max()), 1)
    
    caption_template = "Mean radiant temperature is mapped by day and hour assuming full outdoor unshaded exposure to sun and wind, combining air temperature and incoming solar radiation into a single radiant environment index. For {station}, peak MRT in {peak_mrt_month} between {peak_mrt_hour_start}:00–{peak_mrt_hour_end}:00 exceeds air temperature by up to {mrt_delta}°C, meaning outdoor UTCI heat stress is significantly understated by dry-bulb temperature alone during afternoon hours."
    caption = safe_format_caption(caption_template, {
        "station": station_name,
        "peak_mrt_month": peak_mrt_month,
        "peak_mrt_hour_start": peak_mrt_hour_start,
        "peak_mrt_hour_end": peak_mrt_hour_end,
        "mrt_delta": mrt_delta
    })
    return fig, caption

def build_fig_e_hourly_timeseries_temperature(df, station_name):
    df = _prepare_advanced_figure_df(df)
    missing = [c for c in ["drybulb_C", "dewpoint_C"] if c not in df.columns]
    if missing:
        return placeholder_figure("Dry-bulb and dew-point data are required for temperature time-series analysis."), ""

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['drybulb_C'], name="Dry-Bulb Temperature", line=dict(color='blue', width=0.8), opacity=0.85))
    fig.add_trace(go.Scatter(x=df.index, y=df['dewpoint_C'], name="Dew-Point Temperature", line=dict(color='green', width=0.8), opacity=0.7))
    if not df.empty:
        fig.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[18.0, 18.0], name="Heating Base (18°C)", line=dict(color='gray', dash='dot', width=1.5), mode='lines'))
    
    fig.update_layout(
        title="Full Hourly Time Series — Dry-Bulb and Dew-Point Temperature",
        yaxis_title="Temperature (°C)",
        height=400,
        margin=dict(l=40, r=40, t=40, b=60),
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
    )
    fig.update_xaxes(
        tickformat="%b",
        dtick="M1",
    )
    
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    tdb_min = df['drybulb_C'].min().round(1)
    tdb_max = df['drybulb_C'].max().round(1)
    tdp_min = df['dewpoint_C'].min().round(1)
    tdp_max = df['dewpoint_C'].max().round(1)
    spread = df['drybulb_C'] - df['dewpoint_C']
    driest_month_idx = spread.groupby(df.index.month).mean().idxmax() if isinstance(df.index, pd.DatetimeIndex) else np.nan
    driest_month = months[driest_month_idx - 1] if pd.notna(driest_month_idx) else "[data unavailable]"
    
    caption_template = "The full hourly time series of dry-bulb (blue) and dew-point (green) temperature plots every hour of the typical meteorological year in sequence, making persistent cold and warm spells visible as continuous bands. For {station}, dry-bulb spans {tdb_min}–{tdb_max}°C and dew-point spans {tdp_min}–{tdp_max}°C, with the widest separation in {driest_month} — the strongest evaporative cooling opportunity window of the year."
    caption = safe_format_caption(caption_template, {
        "station": station_name,
        "tdb_min": tdb_min, "tdb_max": tdb_max,
        "tdp_min": tdp_min, "tdp_max": tdp_max,
        "driest_month": driest_month
    })
    return fig, caption

def build_fig_f_hourly_timeseries_rh(df, station_name):
    df = _prepare_advanced_figure_df(df)
    if "rh_pct" not in df.columns:
        return placeholder_figure("Relative humidity data is required for RH time-series analysis."), ""

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['rh_pct'], fill='tozeroy', fillcolor='rgba(0,0,255,0.12)', mode='lines', name='Relative Humidity', line=dict(color='blue', width=0.8)))
    if not df.empty:
        fig.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[30.0, 30.0], name="Dry Limit (30%)", line=dict(color='orange', dash='dot', width=1.5), mode='lines'))
        fig.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[70.0, 70.0], name="Humid Limit (70%)", line=dict(color='steelblue', dash='dot', width=1.5), mode='lines'))
    
    fig.update_layout(
        title="Full Hourly Time Series — Relative Humidity",
        yaxis_title="Relative Humidity (%)",
        yaxis=dict(range=[0, 100]),
        height=400,
        margin=dict(l=40, r=40, t=40, b=60),
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
    )
    fig.update_xaxes(
        tickformat="%b",
        dtick="M1",
    )
    
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pct_humid = round((df['rh_pct'] > 70).mean() * 100, 1)
    pct_dry = round((df['rh_pct'] < 30).mean() * 100, 1)
    
    monthly_mean_rh = df['rh_pct'].groupby(df.index.month).mean() if isinstance(df.index, pd.DatetimeIndex) else pd.Series(dtype=float)
    humid_list = [months[m-1] for m in monthly_mean_rh[monthly_mean_rh > 65].index]
    dry_list = [months[m-1] for m in monthly_mean_rh[monthly_mean_rh < 40].index]
    
    humid_months = ", ".join(humid_list) if humid_list else "none"
    dry_months = ", ".join(dry_list) if dry_list else "none"
    
    caption_template = "The full hourly relative humidity time series plots all 8,760 hours of the typical meteorological year with reference bands at 30% (dry threshold) and 70% (humid threshold). For {station}, {pct_humid}% of annual hours exceed 70% RH — concentrated in {humid_months} — while {pct_dry}% fall below 30% RH in {dry_months}, directly sizing the envelope vapour barrier strategy and mechanical dehumidification load."
    caption = safe_format_caption(caption_template, {
        "station": station_name,
        "pct_humid": pct_humid,
        "pct_dry": pct_dry,
        "humid_months": humid_months,
        "dry_months": dry_months
    })
    return fig, caption

def render_comfort_page():
    st.markdown(
        """
        <section class="cc-page-intro">
            <p class="cc-eyebrow">Comfort</p>
            <h1>Thermal Comfort And Heat Stress</h1>
            <p>Compare discomfort index, outdoor stress, and PMV behavior with advanced controls
            collapsed until they are needed.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    di_tab, utci_tab, pmv_tab = st.tabs(["Discomfort Index", "UTCI", "PMV"])
    with di_tab:
        render_di_page()
    with utci_tab:
        render_utci_page()
    with pmv_tab:
        render_pmv_page()
        
    st.markdown("---")
    st.markdown("### Advanced Comfort & Loads Diagnostics")
    st.caption("Figures generated for publication-standard reporting.")
    
    cdf = st.session_state.get("cdf")
    if cdf is not None and not cdf.empty:
        df_cwec = _prepare_advanced_figure_df(cdf)
        station_name = _safe_location_label(st.session_state.get("header", {}))
        
        # Fig D
        fig_d, cap_d = build_fig_d_mrt_heatmap(df_cwec, station_name)
        _st_plotly_chart(fig_d, use_container_width=True)
        st.caption(cap_d)
        _add_manual_pdf_figure("mrt_heatmap", fig_d)
        
        # Fig E
        fig_e, cap_e = build_fig_e_hourly_timeseries_temperature(df_cwec, station_name)
        _st_plotly_chart(fig_e, use_container_width=True)
        st.caption(cap_e)
        _add_manual_pdf_figure("hourly_timeseries_temperature", fig_e)
        
        # Fig F
        fig_f, cap_f = build_fig_f_hourly_timeseries_rh(df_cwec, station_name)
        _st_plotly_chart(fig_f, use_container_width=True)
        st.caption(cap_f)
        _add_manual_pdf_figure("hourly_timeseries_rh", fig_f)

def render_raw_data_workspace_page():
    cdf = st.session_state.get("cdf")
    st.markdown(
        """
        <section class="cc-page-intro">
            <p class="cc-eyebrow">Data</p>
            <h1>Raw EPW Data And Quality</h1>
            <p>Confirm coverage, inspect source values, and export filtered records behind the charts.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if cdf is not None:
        null_ct = cdf.isna().sum()
        cov_pct = ((1 - null_ct / len(cdf)) * 100).round(1)
        cov_df = (
            pd.DataFrame({"Coverage %": cov_pct, "Missing": null_ct})
            .sort_values("Coverage %", ascending=True)
            .head(12)
        )
        st.markdown(
            """
            <section class="cc-panel">
                <div class="cc-panel-head">
                    <h3>Data Completeness</h3>
                    <p>Lowest-coverage fields are shown first so gaps are visible before analysis.</p>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(cov_df, use_container_width=True)
        st.divider()
        st.markdown("### 💧 Precipitation & Snow")
        st.caption("Understand rainfall and snowfall volume and seasonal distribution.")
        render_precipitation_thermal_load_page()
        st.divider()
    render_raw_data_page()


def render_export_page():
    cdf = st.session_state.get("cdf")
    header = st.session_state.get("header")
    has_data = cdf is not None and header is not None
    captured_figures = _merged_pdf_figures()

    st.markdown(
        """
        <section class="cc-page-intro">
            <p class="cc-eyebrow">Export</p>
            <h1>Reporting Center</h1>
            <p>Generate a complete PDF report, review captured figures, and keep chart-level SVG,
            HTML, and CSV exports tied to the analysis sections where they belong.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown(
            """
            <section class="cc-panel cc-export-panel">
                <div class="cc-panel-head">
                    <h3>Full Climate Report</h3>
                    <p>Captures the complete internal dashboard once, then returns here with a download.</p>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )
        st.text_input("Report title", value=st.session_state.get("export_report_title", "Climate Analysis Report"), key="export_report_title")
        page_size_options = ["A4 Landscape", "A4 Portrait", "A3 Landscape", "A2 Landscape"]
        st.selectbox(
            "PDF page size",
            options=page_size_options,
            index=page_size_options.index(_pdf_page_choice()),
            key="export_pdf_page_size",
            help="A4 landscape keeps report text readable while giving charts a wider frame.",
        )
        st.toggle("Include research branding", value=st.session_state.get("export_include_branding", True), key="export_include_branding")
        st.toggle("Presentation / white-label mode", value=st.session_state.get("export_white_label", False), key="export_white_label")

        export_options_sig = "|".join([
            str(st.session_state.get("export_report_title", "")),
            str(st.session_state.get("export_pdf_page_size", "")),
            str(st.session_state.get("export_include_branding", True)),
            str(st.session_state.get("export_white_label", False)),
        ])
        if st.session_state.get("_export_options_sig") != export_options_sig:
            st.session_state["_export_options_sig"] = export_options_sig
            st.session_state["pdf_download_bytes"] = None
            st.session_state["pdf_download_name"] = None
            st.session_state["pdf_download_error"] = None

        if st.button("Generate Full PDF Report", type="primary", use_container_width=True, disabled=not has_data):
            try:
                with st.spinner("Preparing PDF report..."):
                    pdf_bytes = build_climate_pdf()
                loc_name = _safe_location_label(st.session_state.get("header") or {})
                safe_name = str(loc_name).replace(" ", "_").replace(",", "")
                st.session_state["pdf_download_bytes"] = pdf_bytes
                st.session_state["pdf_download_name"] = f"{safe_name}_Report.pdf"
                st.session_state["pdf_download_error"] = None
                st.session_state["pdf_dashboard_autobuild_pending"] = False
            except Exception as exc:
                st.session_state["pdf_download_bytes"] = None
                st.session_state["pdf_download_name"] = None
                st.session_state["pdf_download_error"] = f"PDF generation failed: {exc}"

        pdf_error = st.session_state.get("pdf_download_error")
        pdf_bytes_ready = st.session_state.get("pdf_download_bytes")
        pdf_name_ready = st.session_state.get("pdf_download_name")
        if pdf_bytes_ready and pdf_name_ready:
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes_ready,
                file_name=pdf_name_ready,
                mime="application/pdf",
                use_container_width=True,
            )
            st.success(f"PDF ready with {len(captured_figures)} visualization(s).")
        elif pdf_error:
            st.error(pdf_error)
        elif has_data:
            st.caption("Generate the report to capture the complete chart set.")
        else:
            st.info("Load a weather file before exporting reports.")

    with right:
        st.markdown(
            f"""
            <section class="cc-panel">
                <div class="cc-panel-head">
                    <h3>Chart Export Status</h3>
                    <p>{len(captured_figures)} Plotly figure(s) currently captured for reporting.</p>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )
        if captured_figures:
            for title in list(captured_figures.keys())[:10]:
                st.caption(f"- {format_figure_title(title)}")
            if len(captured_figures) > 10:
                st.caption(f"- {len(captured_figures) - 10} more captured figure(s)")
        else:
            st.caption("Visit analysis sections or generate a full report to capture figures.")
        debug_missing_cols = st.session_state.get("debug_missing_cols", {})
        if debug_missing_cols:
            with st.expander("Column alias debug", expanded=False):
                for alias_group, columns in debug_missing_cols.items():
                    st.caption(f"{alias_group}: get_metric_column could not match the current aliases.")
                    st.code("\n".join(map(str, columns)), language="text")

        st.markdown(
            """
            <div class="cc-export-note">
                <strong>Per-chart exports</strong>
                <span>SVG, HTML, and CSV actions stay inside each chart panel so exported artifacts remain tied to their context.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ========== MAIN TABS WITH IMPROVED ORGANIZATION ==========
def render_dashboard_page():
    cdf = st.session_state.get("cdf")
    header = st.session_state.get("header")
    comfort_pkg = st.session_state.get("comfort_pkg")
    effective_page = st.session_state.get("nav_page")
    if cdf is None: return

    capture_mode = bool(
        st.session_state.get("pdf_dashboard_autobuild_pending")
        and st.session_state.get("pdf_capture_origin_page") == "Export"
    )
    if capture_mode:
        st.markdown(
            """
            <section class="cc-pdf-capture-screen">
                <p class="cc-eyebrow">Export</p>
                <h1>Building PDF report</h1>
                <p>Capturing dashboard figures and compiling the report. You will return to Export automatically.</p>
            </section>
            """,
            unsafe_allow_html=True,
        )

    # Dashboard Page
    import plotly.express as px

    loc = header["location"]
    pdf_capture_mode = bool(st.session_state.get("pdf_dashboard_autobuild_pending"))
    st.session_state.setdefault("dashboard_section_nav", REPORT_TAB_ORDER[0])
    if st.session_state.get("dashboard_section_nav") not in REPORT_TAB_ORDER:
        st.session_state["dashboard_section_nav"] = REPORT_TAB_ORDER[0]
    if pdf_capture_mode:
        dashboard_section = REPORT_TAB_ORDER[0]
        st.caption("PDF capture mode: rendering all dashboard sections once for the report.")
    else:
        dashboard_section = st.radio(
            "Dashboard view",
            REPORT_TAB_ORDER,
            key="dashboard_section_nav",
            horizontal=True,
            label_visibility="collapsed",
        )

    def _render_dashboard_section(section_name: str) -> bool:
        return pdf_capture_mode or dashboard_section == section_name

    if _render_dashboard_section("Overview & Stats"):
        st.markdown("### 📊 Climate Overview")
        st.caption("Get a high-level sense of the site's climate, from its coordinates to the typical temperature, humidity, wind, and solar character.")
        st.markdown(f"## 📍 {loc.get('city')}, {loc.get('state_province')} — {loc.get('country')}")

        c1, c2, c3, c4, c5 = st.columns(5)

        def _fmt(val, f):
            try:
                return f(float(val))
            except Exception:
                return str(val)

        c1.metric("🌐 Latitude", _fmt(loc.get("latitude"), lambda v: f"{v:.5f}°"))
        c2.metric("🌐 Longitude", _fmt(loc.get("longitude"), lambda v: f"{v:.5f}°"))
        c3.metric("🕐 TZ (hrs from UTC)", _fmt(loc.get("timezone"), lambda v: f"{v:+.1f}"))
        c4.metric("⛰️ Elevation (m)", _fmt(loc.get("elevation_m"), lambda v: f"{v:.1f}"))
        c5.metric("🏷️ WMO", str(loc.get("wmo")))

        st.markdown("### 📈 Annual Climate Statistics")
        c1, c2, c3, c4 = st.columns(4)
        if "drybulb" in cdf:
            c1.metric("🌡️ Avg Temperature", format_temperature(cdf['drybulb'].mean()))
        if "relhum" in cdf:
            c2.metric("💧 Avg Humidity", f"{cdf['relhum'].mean():.0f} %")
        if "windspd" in cdf:
            c3.metric("💨 Avg Wind Speed", f"{cdf['windspd'].mean():.1f} m/s")
        if "glohorrad" in cdf:
            c4.metric("☀️ Avg Solar Rad", f"{cdf['glohorrad'].mean():.0f} W/m²")

        # Seasonal breakdown (Winter, Spring, Summer, Fall)
        season_months = {
            "Winter (Dec-Feb)": [12, 1, 2],
            "Spring (Mar-May)": [3, 4, 5],
            "Summer (Jun-Aug)": [6, 7, 8],
            "Fall (Sep-Nov)": [9, 10, 11],
        }

        def _season_mean(series: pd.Series, months: List[int]):
            if series is None or series.empty:
                return np.nan
            mask = series.index.month.isin(months)
            if not mask.any():
                return np.nan
            return float(series.loc[mask].mean())

        seasonal_rows = []
        for season_label, months in season_months.items():
            seasonal_rows.append({
                "Season": season_label,
                "Avg Temp (°C)": _season_mean(cdf.get("drybulb"), months),
                "Avg Humidity (%)": _season_mean(cdf.get("relhum"), months),
                "Avg Wind (m/s)": _season_mean(cdf.get("windspd"), months),
                "Avg Solar (W/m²)": _season_mean(cdf.get("glohorrad"), months),
            })

        seasonal_df = pd.DataFrame(seasonal_rows).set_index("Season")
        seasonal_df = seasonal_df.map(lambda v: "—" if pd.isna(v) else (f"{v:.1f}" if isinstance(v, float) else v))
        st.markdown("#### Seasonal snapshot")
        st.table(seasonal_df)

        try:
            tmin, tmax = cdf.index.min(), cdf.index.max()
            n_hours = len(cdf)
            st.caption(
                f"Data window: **{tmin:%b %d, %Y} — {tmax:%b %d, %Y}**  ·  Records: **{n_hours:,}** hours"
            )
        except Exception as e:
            st.warning(f"Data window failed: {e}")

        def _month_name(m: int) -> str:
            try:
                return pd.Timestamp(2001, int(m), 1).strftime("%B")
            except Exception:
                return f"Month {m}"

        highlights: List[str] = []
        if "drybulb" in cdf and not cdf["drybulb"].dropna().empty:
            temp_series = cdf["drybulb"].dropna()
            monthly_means = temp_series.groupby(temp_series.index.month).mean()
            daily_highs = temp_series.resample("1D").max().dropna()
            monthly_highs = daily_highs.groupby(daily_highs.index.month).mean()
            daily_lows = temp_series.resample("1D").min().dropna()
            monthly_lows = daily_lows.groupby(daily_lows.index.month).mean()
            daily_means = temp_series.resample("1D").mean().dropna()
            hdd_daily = (18.0 - daily_means).clip(lower=0)
            monthly_hdd = hdd_daily.groupby(hdd_daily.index.month).sum()

            if not monthly_means.empty:
                warm_month = int(monthly_means.idxmax())
                warm_label = _month_name(warm_month)
                warm_high = monthly_highs.get(warm_month, monthly_means.loc[warm_month])
                highlights.append(
                    f"{warm_label} is the warmest month, with typical daytime highs near {format_temperature(warm_high)}."
                )

                cold_month = int(monthly_means.idxmin())
                cold_label = _month_name(cold_month)
                cold_low = monthly_lows.get(cold_month, monthly_means.loc[cold_month])
                hdd_val = monthly_hdd.get(cold_month)
                if pd.isna(hdd_val):
                    highlights.append(
                        f"{cold_label} is when winters bite hardest, with overnight lows around {format_temperature(cold_low)}."
                    )
                else:
                    highlights.append(
                        f"{cold_label} brings overnight lows near {format_temperature(cold_low)} and roughly {hdd_val:.0f} heating degree days (base 18 °C)."
                    )

        if "relhum" in cdf and not cdf["relhum"].dropna().empty:
            rh_mean = cdf["relhum"].mean()
            highlights.append(f"Annual mean humidity hovers around {rh_mean:.0f}% — generally a moderate moisture profile.")

        if highlights:
            st.markdown("#### 💡 Key takeaways")
            st.markdown("\n".join(f"- {text}" for text in highlights))

        # ========== HEATMAPS (moved from former Heatmaps tab) ==========
        st.divider()
        st.divider()
        
        # Dynamic header with location


        import json

        # Default thresholds
        default_thresholds = {
            "solar": [100.0, 300.0, 500.0, 700.0],
            "humidity": [40.0, 60.0, 80.0],
            "wind": [1.5, 4.5],
        }

        # Initialize/persist thresholds in session_state
        if "heatmap_thresholds" not in st.session_state:
            st.session_state["heatmap_thresholds"] = default_thresholds.copy()

        thresholds_state = st.session_state["heatmap_thresholds"]

        def _labels_from_thresholds(ths: list[float], suffix: str) -> list[str]:
            labels = []
            if not ths:
                return labels
            labels.append(f"<{ths[0]:g}{suffix}")
            for a, b in zip(ths, ths[1:]):
                labels.append(f"{a:g}–{b:g}{suffix}")
            labels.append(f">{ths[-1]:g}{suffix}")
            return labels

        invalid_thresholds = False

        with st.expander("Legend & Thresholds", expanded=False):
            c_reset = st.columns([3,1])[1]
            if c_reset.button("Reset thresholds to defaults"):
                st.session_state["heatmap_thresholds"] = default_thresholds.copy()
                _rerun()

            c_s1, c_s2, c_s3, c_s4 = st.columns(4)
            solar_t1 = c_s1.number_input("Solar t1 (W/m²)", value=float(thresholds_state["solar"][0]), step=50.0, key="solar_t1")
            solar_t2 = c_s2.number_input("Solar t2", value=float(thresholds_state["solar"][1]), step=50.0, key="solar_t2")
            solar_t3 = c_s3.number_input("Solar t3", value=float(thresholds_state["solar"][2]), step=50.0, key="solar_t3")
            solar_t4 = c_s4.number_input("Solar t4", value=float(thresholds_state["solar"][3]), step=50.0, key="solar_t4")
            solar_thresholds = [solar_t1, solar_t2, solar_t3, solar_t4]

            c_h1, c_h2, c_h3 = st.columns(3)
            hum_t1 = c_h1.number_input("Humidity t1 (%)", value=float(thresholds_state["humidity"][0]), step=5.0, key="hum_t1")
            hum_t2 = c_h2.number_input("Humidity t2", value=float(thresholds_state["humidity"][1]), step=5.0, key="hum_t2")
            hum_t3 = c_h3.number_input("Humidity t3", value=float(thresholds_state["humidity"][2]), step=5.0, key="hum_t3")
            humidity_thresholds = [hum_t1, hum_t2, hum_t3]

            c_w1, c_w2 = st.columns(2)
            wind_t1 = c_w1.number_input("Wind t1 (m/s)", value=float(thresholds_state["wind"][0]), step=0.5, key="wind_t1")
            wind_t2 = c_w2.number_input("Wind t2", value=float(thresholds_state["wind"][1]), step=0.5, key="wind_t2")
            wind_thresholds = [wind_t1, wind_t2]

            def _is_strictly_increasing(vals):
                return all(vals[i] < vals[i+1] for i in range(len(vals)-1))

            if not (_is_strictly_increasing(solar_thresholds) and _is_strictly_increasing(humidity_thresholds) and _is_strictly_increasing(wind_thresholds)):
                invalid_thresholds = True
                st.error("Thresholds must be strictly increasing for each metric.")
            else:
                thresholds_state["solar"] = solar_thresholds
                thresholds_state["humidity"] = humidity_thresholds
                thresholds_state["wind"] = wind_thresholds

        if invalid_thresholds:
            st.info("Adjust thresholds to continue.")
        else:
            location_label = get_clean_city_name()
            st.markdown(f"<h3>{location_label} – Annual Diurnal Resource Heatmaps</h3>", unsafe_allow_html=True)
            st.caption("Adjust legend thresholds to explore how different performance ranges appear across the year.")
            # Continuous heatmap helper: aggregate first (mean/median), keep thresholds for hover/legend only
            def _build_pivot_with_thresholds(df: pd.DataFrame, col: str, metric_label: str, thresholds: list[float], units_suffix: str, palette_metric: str, agg: str = "mean", raw_col: str = None):
                series = pd.to_numeric(df[col], errors="coerce")
                series = series.dropna()
                if series.empty:
                    return pd.DataFrame(), {"error": f"No valid data for {metric_label}"}

                work = pd.DataFrame({"val": series})
                if raw_col:
                    work["raw"] = pd.to_numeric(df[raw_col], errors="coerce")
                else:
                    work["raw"] = work["val"]
                    
                work["hod"] = work.index.hour
                work["doy"] = work.index.dayofyear

                if agg == "median":
                    aggfunc = "median"
                elif agg == "max":
                    aggfunc = "max"
                else:
                    aggfunc = "mean"
                pivot_raw = work.pivot_table(index="hod", columns="doy", values="val", aggfunc=aggfunc)
                pivot_raw = pivot_raw.reindex(index=range(24), columns=range(1, 367))
                
                # For labels, we use the RAW values to bin against thresholds
                # We need to aggregate raw values same way to match the grid cells?
                # Or do we bin the aggregated raw values? 
                # If we bin the raw values, we get mode? 
                # Simpler: Aggregate the RAW values too, then bin.
                pivot_val_for_labels = work.pivot_table(index="hod", columns="doy", values="raw", aggfunc=aggfunc)
                pivot_val_for_labels = pivot_val_for_labels.reindex(index=range(24), columns=range(1, 367))

                # Thresholds for interpretation (hover/legend), not for coloring
                bins = [-np.inf] + thresholds + [np.inf]
                labels = _labels_from_thresholds(thresholds, units_suffix)
                
                cat = pd.cut(pivot_val_for_labels.values.flatten(), bins=bins, labels=labels, right=False)
                label_grid = pd.Series(cat).astype(object).values.reshape(pivot_raw.shape)

                colors_default, _ = get_color_scale_for_metric(palette_metric)
                colors = colors_default[: len(labels)]

                info = {
                    "metric": metric_label,
                    "col": col,
                    "labels": labels,
                    "colors": colors,
                    "thresholds": thresholds,
                    "hover_labels": label_grid,
                }
                return pivot_raw, info

            heatmap_dict = {}
            
            # Add Dry Bulb Temperature first
            db_col = get_metric_column(cdf, ["dry_bulb_temperature", "drybulb", "temp", "db"])
            if db_col:
                # Note: We pass [] for thresholds to let defaults or custom thresholds handling kick in
                # But _build_pivot_with_thresholds expects a list. 
                # If thresholds_state has 'drybulb', use it, else empty.
                # We will assume "Dry Bulb Temperature" uses the color scale "drybulb" or "thermal" which is standard.
                # We pass agg="mean".
                db_thresholds = [10.0, 20.0, 26.6, 35.0]
                
                # Prepare categorical bins for heatmap values (0, 1, 2...)
                # Correct bin edges: -100, 10, 20, 26.6, 35, 100
                db_bins = [-100, 10.0, 20.0, 26.6, 35.0, 100.0] 
                db_series = cdf[db_col]
                cdf["db_cat"] = pd.cut(db_series, bins=db_bins, labels=[0, 1, 2, 3, 4], include_lowest=True, right=False).astype(float)

                pivot_binned, info = _build_pivot_with_thresholds(
                    cdf, "db_cat", "Dry Bulb Temperature", db_thresholds, "°C", "drybulb", agg="mean", raw_col=db_col
                )
                if not pivot_binned.empty:
                    heatmap_dict["Dry Bulb Temperature"] = (pivot_binned, info)

            solar_col = get_metric_column(cdf, ["glohorrad", "global_horizontal_radiation", "solar"])
            if solar_col:
                # Bin Solar into categories 0..4 for discrete coloring
                # 0:<100, 1:100-300, 2:300-500, 3:500-700, 4:>700
                # Ensure we handle values >= 0
                s_series = cdf[solar_col].clip(lower=0)
                sb = [0.0, 100.0, 300.0, 500.0, 700.0, s_series.max() + 1.0]
                # If max < 700, ensure bins cover it
                if sb[-1] <= 700.0: sb[-1] = 9999.0
                
                cdf["solar_cat"] = pd.cut(s_series, bins=sb, labels=[0, 1, 2, 3, 4], include_lowest=True, right=False).astype(float)
                
                # Pass "solar_cat" for the heatmap values, but RAW solar_col for labels
                pivot_binned, info = _build_pivot_with_thresholds(
                    cdf, "solar_cat", "Solar Radiation", thresholds_state["solar"], " W/m²", "solar", agg="mean", raw_col=solar_col
                )
                if not pivot_binned.empty:
                    heatmap_dict["Solar Radiation"] = (pivot_binned, info)

            rh_col = get_metric_column(cdf, ["relative_humidity", "relhum", "rh"])
            if rh_col:
                # Pre-bin humidity into categories for consistent handling
                rh_series = cdf[rh_col].clip(0, 100)
                rh_bins = [0.0, 30.0, 70.0, 100.01]  # <30, 30-70, >70
                cdf["rh_cat"] = pd.cut(rh_series, bins=rh_bins, labels=[0, 1, 2], include_lowest=True, right=False).astype(float)
                pivot_binned, info = _build_pivot_with_thresholds(cdf, "rh_cat", "Humidity", thresholds_state["humidity"], "%", "humidity", agg="mean", raw_col=rh_col)
                if not pivot_binned.empty:
                    heatmap_dict["Humidity"] = (pivot_binned, info)

            precip_col = _precip_depth_column(cdf)
            if precip_col:
                precip_series = pd.to_numeric(cdf[precip_col], errors="coerce").mask(lambda s: (s < 0) | (s > 900)).fillna(0)
                cdf["precipdepthclean"] = precip_series
                max_precip = float(precip_series.max()) if not precip_series.empty else 0.0
                trace_threshold = 0.01 if max_precip < 0.1 else 0.1
                precip_bins = [0.0, trace_threshold, 2.5, 10.0, max(max_precip + trace_threshold, 999.0)]
                cdf["precipcat"] = pd.cut(
                    precip_series,
                    bins=precip_bins,
                    labels=[0, 1, 2, 3],
                    include_lowest=True,
                    right=False,
                ).astype(float)
                pivot_binned, info = _build_pivot_with_thresholds(
                    cdf,
                    "precipcat",
                    "Precipitation",
                    [trace_threshold, 2.5, 10.0],
                    " mm",
                    "precipitation",
                    agg="max",
                    raw_col="precipdepthclean",
                )
                if not pivot_binned.empty:
                    precip_labels = ["Dry", "Light", "Moderate", "Heavy"]
                    info["labels"] = precip_labels
                    info["hover_labels"] = pivot_binned.map(
                        lambda v: precip_labels[int(v)] if pd.notna(v) and 0 <= int(v) < len(precip_labels) else np.nan
                    ).values
                    heatmap_dict["Precipitation"] = (pivot_binned, info)

            wind_col = get_metric_column(cdf, ["windspd", "wind_speed", "wspd"])
            if wind_col:
                wb = [0.0, 1.5, 4.5, cdf[wind_col].max() + 1.0]
                if wb[-1] <= 4.5: wb[-1] = 999.0
                
                cdf["wind_cat"] = pd.cut(cdf[wind_col], bins=wb, labels=[0, 1, 2], include_lowest=True, right=False).astype(float)
                
                # Pass "wind_cat" for heatmap, RAW wind_col for labels
                pivot_binned, info = _build_pivot_with_thresholds(
                    cdf, "wind_cat", "Wind Speed", thresholds_state["wind"], " m/s", "wind", agg="median", raw_col=wind_col
                )
                if not pivot_binned.empty:
                    heatmap_dict["Wind Speed"] = (pivot_binned, info)
            _wd_col = next((c for c in cdf.columns if "wind" in c.lower() and "dir" in c.lower()), None)
            if _wd_col:
                try:
                    dir_col = _wd_col
                    _COMPASS_TO_DEG = {
                        "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
                        "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
                        "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
                        "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
                    }
                    raw_dir = cdf[dir_col].astype(str).str.strip().str.upper()
                    dir_series_num = pd.to_numeric(raw_dir, errors="coerce")
                    dir_series_txt = raw_dir.map(_COMPASS_TO_DEG)
                    dir_series = dir_series_num.fillna(dir_series_txt)
                    dir_series = dir_series.mask(dir_series >= 999)
                    dir_series = dir_series.dropna()

                    if not dir_series.empty:
                        work_dir = pd.DataFrame({"val": dir_series})
                        work_dir["hod"] = work_dir.index.hour
                        work_dir["doy"] = work_dir.index.dayofyear

                        sector_names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

                        def dir_to_sector_code(deg: float) -> float:
                            if pd.isna(deg): return np.nan
                            d = deg % 360.0
                            if d >= 337.5 or d < 22.5: return 0
                            elif d < 67.5: return 1
                            elif d < 112.5: return 2
                            elif d < 157.5: return 3
                            elif d < 202.5: return 4
                            elif d < 247.5: return 5
                            elif d < 292.5: return 6
                            else: return 7

                        work_dir["sector"] = work_dir["val"].apply(dir_to_sector_code)
                        work_dir = work_dir.dropna(subset=["sector"])

                        sector_mode_series = (
                            work_dir.groupby(["hod", "doy"])["sector"]
                            .agg(lambda s: s.mode().iat[0] if not s.mode().empty else np.nan)
                        )
                        pivot_dir = sector_mode_series.unstack("doy")
                        pivot_dir = pivot_dir.reindex(index=range(24), columns=range(1, 366))

                        dir_colors = ["#4c6fff", "#3fb3ff", "#36d1a8", "#8bd36b",
                                    "#f6c445", "#f08c42", "#e15b9a", "#9d6bff"]
                        label_grid = pivot_dir.map(
                            lambda v: sector_names[int(v)] if pd.notna(v) else np.nan
                        ).values

                        info_dir = {
                            "metric": "Wind Direction",
                            "col": dir_col,
                            "labels": sector_names,
                            "colors": dir_colors,
                            "thresholds": [],
                            "hover_labels": label_grid,
                            "show_colorscale": False,
                            "colorbar": None,
                        }
                        heatmap_dict["Wind Direction"] = (pivot_dir, info_dir)

                except Exception as e:
                    st.warning(f"Wind direction heatmap failed: {e}")
    

            if not heatmap_dict:
                st.info("No metric data available for heatmap generation.")
            else:
                fig = build_diurnal_heatmap_figure(heatmap_dict, cdf, header)
                if fig:
                    _st_plotly_chart(fig, use_container_width=False, config={"responsive": False})
                    _add_manual_pdf_figure("Annual Diurnal Resource Heatmap", fig)
                    # Downloads reflecting current thresholds
                    city_clean = get_clean_city_name().replace(" ", "_").replace(",", "").replace("__", "_")
                    clean_loc = city_clean
                    c1d, c2d = st.columns(2)
                    with c1d:
                        try:
                            svg_bytes = fig.to_image(format="svg", scale=2, width=1200, height=800)
                            st.download_button(label="📥 Download heatmaps as SVG", data=svg_bytes, file_name=f"{clean_loc}_diurnal_heatmaps.svg", mime="image/svg+xml")
                        except Exception:
                            html_bytes = fig.to_html(include_plotlyjs='cdn').encode('utf-8')
                            st.download_button(label="📥 Download heatmaps (HTML)", data=html_bytes, file_name=f"{clean_loc}_diurnal_heatmaps.html", mime="text/html")

                    with c2d:
                        try:
                            export_df = cdf.copy()
                            long_records = []
                            thresholds_json = json.dumps(thresholds_state)
                            for strip_name, (pivot_binned, info) in heatmap_dict.items():
                                col = info.get('col')
                                if not col or col not in export_df.columns:
                                    continue
                                s = pd.to_numeric(export_df[col], errors="coerce").dropna()
                                if s.empty:
                                    continue
                                thresholds_list = info.get("thresholds", [])
                                bins = [-np.inf] + thresholds_list + [np.inf]
                                labels = info.get("labels", [])
                                if len(labels) != len(bins) - 1:
                                    labels = _labels_from_thresholds(thresholds_list, "")
                                assert len(labels) == len(bins) - 1, (len(labels), len(bins))
                                cat = pd.cut(s, bins=bins, labels=labels, right=False)
                                for ts, val, bl in zip(cat.index, s.values, cat.astype(object).values):
                                    long_records.append({
                                        'datetime': ts.isoformat(),
                                        'variable': strip_name,
                                        'value': float(val),
                                        'bin_label': bl if pd.notna(bl) else None,
                                        'month': int(ts.month),
                                        'hour': int(ts.hour),
                                        'thresholds': thresholds_json,
                                    })

                            if not long_records:
                                st.caption("No long-format data available for CSV export.")
                            else:
                                long_df = pd.DataFrame(long_records)
                                csv_bytes = long_df.to_csv(index=False).encode('utf-8')
                                st.download_button(label="📥 Download heatmap data (CSV)", data=csv_bytes, file_name=f"{clean_loc}_diurnal_heatmaps_data.csv", mime="text/csv")
                        except Exception as e:
                            st.caption(f"CSV export failed: {str(e)[:80]}")
                else:
                    st.warning("Could not generate heatmap figure from available data.")

    if _render_dashboard_section("Comfort & Loads"):
        st.markdown("### 😌 Thermal Comfort & Loads")
        st.caption("Explore how often indoor comfort bands are met, where overheating or cold stress creep in, and how heating/cooling loads shift through the year.")
        comfort_pkg = st.session_state.get("comfort_pkg", {}) or {}
        comfort_annual_base = comfort_pkg.get("comfort_annual")
        comfort_monthly_base = comfort_pkg.get("comfort_monthly")
        loads_annual = comfort_pkg.get("loads_annual")
        di_series = comfort_pkg.get("di")
        utci_series = comfort_pkg.get("utci")
        heat_index_series = comfort_pkg.get("heat_index")
        humidex_series = comfort_pkg.get("humidex")
        focus_threshold = int(st.session_state.get("custom_overheat_threshold", 30))
        prefer_adaptive = bool(st.session_state.get("prefer_adaptive_comfort", False))

        def _build_occupancy_mask(idx: pd.DatetimeIndex, mode: str) -> pd.Series:
            if mode == "24/7":
                return pd.Series(True, index=idx)
            if mode == "Daytime (07-22)":
                arr = (idx.hour >= 7) & (idx.hour < 22)
                return pd.Series(arr, index=idx)
            if mode == "Workday (Mon-Fri 9-17)":
                arr = ((idx.dayofweek < 5) & (idx.hour >= 9) & (idx.hour < 17))
                return pd.Series(arr, index=idx)
            return pd.Series(True, index=idx)

        with st.expander("⚙️ Comfort analysis settings", expanded=False):
            comfort_mode = st.radio(
                "Comfort band",
                ["Fixed 18–26 °C", "Adaptive (ASHRAE 55)"],
                index=1 if prefer_adaptive else 0,
                horizontal=True,
            )
            adaptive_band = None
            comfort_band = (18.0, 26.0)
            if comfort_mode == "Fixed 18–26 °C":
                comfort_band = st.slider(
                    "Comfort temperature band (°C)",
                    min_value=-10.0,
                    max_value=40.0,
                    value=(18.0, 26.0),
                    step=0.5,
                )
            else:
                if "drybulb" not in cdf.columns:
                    st.warning("Adaptive comfort requires dry-bulb data; falling back to fixed band.")
                else:
                    acceptability = st.radio("Adaptive acceptability", ["80%", "90%"], horizontal=True)
                    acc_value = 0.9 if acceptability == "90%" else 0.8
                    adaptive_band = ce.build_adaptive_band(cdf["drybulb"], acceptability=acc_value)
                    comfort_band = None

            occupancy_mode = st.selectbox(
                "Occupancy schedule",
                ["24/7", "Daytime (07-22)", "Workday (Mon-Fri 9-17)"],
                index=0,
            )
            occupancy_mask = None if occupancy_mode == "24/7" else _build_occupancy_mask(cdf.index, occupancy_mode)

            hot_thresholds = st.multiselect(
                "Overheating thresholds (°C)",
                options=list(range(24, 41)),
                default=[28, 30],
                help="Counts hours above each selected dry-bulb threshold.",
            )
            if not hot_thresholds:
                hot_thresholds = [28, 30]
            if focus_threshold not in hot_thresholds:
                hot_thresholds.append(focus_threshold)
            hot_thresholds = sorted(set(int(th) for th in hot_thresholds))
            cold_thresholds = st.multiselect(
                "Cold stress thresholds (°C)",
                options=list(range(-20, 11)),
                default=[0],
                help="Counts hours below each selected dry-bulb threshold.",
            )
            percentiles_on = st.checkbox("Show percentile diagnostics", value=True)

        comfort_annual = comfort_annual_base
        comfort_monthly = comfort_monthly_base
        percentiles = (0.9, 0.95) if percentiles_on else None
        try:
            comfort_dyn_a = ce.summarize_comfort(
                cdf,
                di_series,
                utci_series,
                freq="YE",
                comfort_band=comfort_band,
                adaptive_band=adaptive_band,
                overheating_thresholds=hot_thresholds,
                cold_thresholds=cold_thresholds,
                percentiles=percentiles,
                occupancy_mask=occupancy_mask,
            )
            if comfort_dyn_a is not None and not comfort_dyn_a.empty:
                comfort_annual = comfort_dyn_a
        except Exception as e:
            st.warning(f"comfort_annual failed: {e}")

        try:
            comfort_dyn_m = ce.summarize_comfort(
                cdf,
                di_series,
                utci_series,
                freq="ME",
                comfort_band=comfort_band,
                adaptive_band=adaptive_band,
                overheating_thresholds=hot_thresholds,
                cold_thresholds=cold_thresholds,
                percentiles=percentiles,
                occupancy_mask=occupancy_mask,
            )
            if comfort_dyn_m is not None and not comfort_dyn_m.empty:
                comfort_monthly = comfort_dyn_m
        except Exception as e:
            st.warning(f"comfort_monthly failed: {e}")

        def _fmt_hours(val: float) -> str:
            return "—" if pd.isna(val) else f"{float(val):.0f} h"

        def _fmt_value(val: float, suffix: str = "") -> str:
            return "—" if pd.isna(val) else f"{float(val):.0f}{suffix}"

        if comfort_annual is None or comfort_annual.empty:
            st.info("Comfort insights unlock automatically when dry-bulb, humidity, and wind speed data are available.")
        else:
            comfort_tab, loads_tab = st.tabs([
                "😌 Comfort Compliance & Stress",
                "🌡️ Degree Days & Loads",
            ])
            
            with comfort_tab:
                latest = comfort_annual.iloc[-1]
                comfort_pct = latest.get("fraction_in_comfort_band", np.nan)
                comfort_hours = latest.get("hours_in_comfort_band", np.nan)
                total_hours = latest.get("hours_total", np.nan)
                di_discomfort = latest.get("hours_di_discomfort", np.nan)
                utci_heat = latest.get("hours_utci_heat_stress", np.nan)
                utci_cold = latest.get("hours_utci_cold_stress", np.nan)
                hot_cols = sorted([c for c in latest.index if c.startswith("overheating_hours_")])
                cold_cols = sorted([c for c in latest.index if c.startswith("cold_hours_below_")])
                focus_col = f"overheating_hours_{focus_threshold}C"
                if focus_col in hot_cols:
                    hot_cols.remove(focus_col)
                    hot_cols.insert(0, focus_col)

                comfort_value = "—" if pd.isna(comfort_pct) else f"{comfort_pct * 100:.1f} %"
                comfort_delta = None
                if not pd.isna(comfort_hours) and not pd.isna(total_hours):
                    comfort_delta = f"{comfort_hours:.0f}/{total_hours:.0f} h"

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Comfort compliance", comfort_value, delta=comfort_delta)

                di_available = di_series is not None and not getattr(di_series, "empty", True)
                utci_available = utci_series is not None and not getattr(utci_series, "empty", True)

                di_value = _fmt_hours(di_discomfort)
                utci_value = _fmt_hours(utci_heat)
                delta_cold = None if pd.isna(utci_cold) else f"Cold: {utci_cold:.0f} h"

                mc2.metric("DI discomfort", di_value)
                mc3.metric("UTCI heat stress", utci_value, delta=delta_cold)

                if not di_available:
                    mc2.caption("Needs dry-bulb and relative humidity to compute DI.")
                if not utci_available:
                    mc3.caption("Needs dry-bulb, relative humidity, and wind speed for UTCI.")

                hot_display = []
                for col in hot_cols[:2]:
                    thresh = col.replace("overheating_hours_", "").replace("C", "")
                    try:
                        thresh_c = float(thresh)
                    except ValueError:
                        thresh_c = float(focus_threshold)
                    hot_display.append((f"{format_threshold_label(thresh_c)} hours", latest.get(col, np.nan)))
                if hot_display:
                    oc_cols = st.columns(len(hot_display))
                    for col_obj, (label, value) in zip(oc_cols, hot_display):
                        col_obj.metric(label, _fmt_hours(value))
                if cold_cols:
                    cc_cols = st.columns(min(len(cold_cols), 2))
                    for col_obj, col_name in zip(cc_cols, cold_cols[:2]):
                        thresh = col_name.replace("cold_hours_below_", "").replace("C", "")
                        try:
                            thresh_c = float(thresh)
                        except ValueError:
                            thresh_c = 0.0
                        col_obj.metric(f"{format_threshold_label(thresh_c, direction='<')} hours", _fmt_hours(latest.get(col_name, np.nan)))

                if occupancy_mode != "24/7":
                    st.caption(f"Comfort metrics filtered to {occupancy_mode.lower()} hours.")
                if comfort_mode != "Fixed 18–26 °C":
                    st.caption("Adaptive comfort band follows ASHRAE 55's running-mean method—great for naturally ventilated spaces.")
                st.caption(
                    f"Focus threshold: tracking hours above {format_threshold_label(focus_threshold, direction='>')} per the Customize Analysis panel."
                )

                if comfort_monthly is not None and not comfort_monthly.empty:
                    # Collapse multi-year monthly rows into one row per calendar month to avoid zig-zag lines
                    monthly = comfort_monthly.copy()
                    month_numbers = monthly.index.month

                    agg_spec = {}
                    for col in monthly.columns:
                        if col.startswith("fraction_in_comfort_band"):
                            agg_spec[col] = "mean"
                        else:
                            agg_spec[col] = "sum"

                    monthly_grouped = monthly.copy()
                    monthly_grouped["month_num"] = month_numbers
                    monthly_grouped = monthly_grouped.groupby("month_num").agg(agg_spec)
                    monthly_grouped = monthly_grouped.reindex(range(1, 13))

                    # Fill gaps for hour counts; keep comfort fraction as-is so missing months stay blank
                    for col in monthly_grouped.columns:
                        if not col.startswith("fraction_in_comfort_band"):
                            monthly_grouped[col] = monthly_grouped[col].fillna(0)

                    monthly_grouped.index = [pd.Timestamp(2001, int(m), 1).strftime("%b") for m in monthly_grouped.index]

                    fig_comfort = make_subplots(specs=[[{"secondary_y": True}]])
                    for idx, col_name in enumerate(hot_cols[:2]):
                        thresh = col_name.replace("overheating_hours_", "").replace("C", "")
                        try:
                            thresh_c = float(thresh)
                        except ValueError:
                            thresh_c = float(focus_threshold)
                        fig_comfort.add_bar(
                            name=format_threshold_label(thresh_c),
                            x=monthly_grouped.index,
                            y=monthly_grouped.get(col_name, pd.Series(index=monthly_grouped.index)).fillna(0),
                            marker_color="#fb923c" if idx == 0 else "#f97316",
                            opacity=0.6 if idx == 1 else 0.8,
                            secondary_y=False,
                        )

                    if "hours_utci_heat_stress" in monthly_grouped:
                        fig_comfort.add_bar(
                            name="UTCI heat stress",
                            x=monthly_grouped.index,
                            y=monthly_grouped["hours_utci_heat_stress"].fillna(0),
                            marker_color="#ef4444",
                            opacity=0.5,
                            secondary_y=False,
                        )

                    if "fraction_in_comfort_band" in monthly_grouped:
                        fig_comfort.add_scatter(
                            name="Comfort %",
                            x=monthly_grouped.index,
                            y=(monthly_grouped["fraction_in_comfort_band"] * 100),
                            mode="lines+markers",
                            line=dict(color="#34d399", width=2.5),
                            marker=dict(size=6),
                            secondary_y=True,
                        )

                    fig_comfort.update_layout(
                        bargap=0.2,
                        hovermode="x unified",
                        margin=dict(l=0, r=0, t=30, b=0),
                        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="left", x=0),
                    )
                    fig_comfort.update_yaxes(title_text="Hours", secondary_y=False)
                    fig_comfort.update_yaxes(title_text="Comfort %", range=[0, 100], secondary_y=True)
                    _st_plotly_chart(fig_comfort, use_container_width=True)
                    _add_manual_pdf_figure("Comfort Loads", fig_comfort)

                # Point-in-time probe: inspect weather and comfort metrics together at a chosen hour
                with st.expander("Point-in-time probe", expanded=False):
                    if len(cdf.index):
                        available_dates = sorted(pd.to_datetime(cdf.index.date).unique())
                        default_date = available_dates[0]
                        chosen_date = st.date_input(
                            "Date",
                            value=default_date,
                            min_value=available_dates[0],
                            max_value=available_dates[-1],
                            key="comfort_loads_pit_date"
                        )
                        chosen_hour = st.slider("Hour (0–23)", 0, 23, 14, key="comfort_loads_pit_hour")
                        ts = pd.Timestamp(year=chosen_date.year, month=chosen_date.month, day=chosen_date.day, hour=int(chosen_hour))
                        idx_tz = getattr(cdf.index, "tz", None)
                        if idx_tz is not None:
                            ts = ts.tz_localize(idx_tz, nonexistent="shift_forward", ambiguous="NaT")
                        nearest_idx = cdf.index.get_indexer([ts], method="nearest")
                        if nearest_idx[0] != -1:
                            snap = cdf.iloc[nearest_idx[0]]
                            snap_di = di_series.iloc[nearest_idx[0]] if di_series is not None and not getattr(di_series, "empty", True) else np.nan
                            snap_utci = utci_series.iloc[nearest_idx[0]] if utci_series is not None and not getattr(utci_series, "empty", True) else np.nan
                            snap_rows = [
                                ("Dry-bulb (°C)", format_temperature(snap.get("drybulb"))),
                                ("Rel humidity (%)", "—" if pd.isna(snap.get("relhum")) else f"{snap.get('relhum'):.0f} %"),
                                ("Wind speed (m/s)", "—" if pd.isna(snap.get("windspd")) else f"{snap.get('windspd'):.1f}"),
                                ("DI", "—" if pd.isna(snap_di) else f"{snap_di:.1f}"),
                                ("UTCI (°C)", "—" if pd.isna(snap_utci) else f"{snap_utci:.1f}"),
                                ("Heat index (°C)", "—" if heat_index_series is None or pd.isna(heat_index_series.iloc[nearest_idx[0]]) else f"{heat_index_series.iloc[nearest_idx[0]]:.1f}"),
                                ("Humidex (°C)", "—" if humidex_series is None or pd.isna(humidex_series.iloc[nearest_idx[0]]) else f"{humidex_series.iloc[nearest_idx[0]]:.1f}"),
                            ]
                            snap_df = pd.DataFrame(snap_rows, columns=["Metric", "Value"]).set_index("Metric")
                            st.table(snap_df)
                        else:
                            st.info("No data available for that selection.")

                if percentiles_on and not comfort_annual.empty:
                    pct_cols = [c for c in latest.index if c.startswith("temp_p") or c.startswith("di_p") or c.startswith("utci_p")]
                    pct_series = latest[pct_cols].dropna()
                    hi_pct = None
                    hum_pct = None
                    if heat_index_series is not None and not heat_index_series.empty:
                        hi_src = heat_index_series
                        if occupancy_mask is not None:
                            occ = occupancy_mask.reindex(hi_src.index).fillna(False)
                            hi_src = hi_src.loc[occ]
                        hi_pct = hi_src.quantile(0.95)
                    if humidex_series is not None and not humidex_series.empty:
                        hum_src = humidex_series
                        if occupancy_mask is not None:
                            occ = occupancy_mask.reindex(hum_src.index).fillna(False)
                            hum_src = hum_src.loc[occ]
                        hum_pct = hum_src.quantile(0.95)
                    with st.expander("Percentile & feels-like diagnostics", expanded=False):
                        if not pct_series.empty:
                            st.write(pct_series.rename(lambda c: c.replace("_", " ")))
                        hi_text = (
                            f"Heat index 95th percentile: {format_temperature(hi_pct)}"
                            if hi_pct is not None else "Heat index data unavailable."
                        )
                        hum_text = (
                            f"Humidex 95th percentile: {format_temperature(hum_pct)}"
                            if hum_pct is not None else "Humidex data unavailable."
                        )
                        st.caption(f"{hi_text}\n\n{hum_text}")

            with loads_tab:
                if loads_annual is not None and not loads_annual.empty:
                    loads_latest = loads_annual.iloc[-1]
                    l1, l2 = st.columns(2)
                    l1.metric(
                        "Heating degree days",
                        _fmt_value(loads_latest.get("heating_degree_days", np.nan)),
                        delta=_fmt_value(loads_latest.get("heating_degree_hours", np.nan), " h")
                    )
                    l2.metric(
                        "Cooling degree days",
                        _fmt_value(loads_latest.get("cooling_degree_days", np.nan)),
                        delta=_fmt_value(loads_latest.get("cooling_degree_hours", np.nan), " h")
                    )
                else:
                    st.info("Degree days metrics are not available.")
                
                # Render Degree Days Chart in Comfort & Loads tab
                fig_dd = _degree_day_dashboard_fig(cdf)
                _st_plotly_chart(fig_dd, use_container_width=True, key="comfort_loads_degree_days")
                _add_manual_pdf_figure("Heating and Cooling Degree Days (Comfort)", fig_dd)

        st.markdown("### Advanced Comfort & Loads Diagnostics")
        df_cwec = _prepare_advanced_figure_df(cdf)
        station_name = _safe_location_label(st.session_state.get("header", {}))
        for title, key, builder in [
            ("MRT Heatmap", "mrt_heatmap", build_fig_d_mrt_heatmap),
            ("Full Hourly Time Series - Dry-Bulb and Dew-Point Temperature", "hourly_timeseries_temperature", build_fig_e_hourly_timeseries_temperature),
            ("Full Hourly Time Series - Relative Humidity", "hourly_timeseries_rh", build_fig_f_hourly_timeseries_rh),
        ]:
            fig_adv, cap_adv = builder(df_cwec, station_name)
            _st_plotly_chart(fig_adv, use_container_width=True)
            if cap_adv:
                st.caption(cap_adv)
            _add_manual_pdf_figure(key, fig_adv)

        # ========== DI, UTCI, PMV (moved from separate tabs) ==========
        st.divider()
        st.markdown("### 🌡️ Discomfort Index (DI)")
        render_di_page()

        st.divider()
        st.markdown("### 🥵 UTCI")
        render_utci_page()

        st.divider()
        st.markdown("### 🏠 PMV")
        render_pmv_page()



    if _render_dashboard_section("Temp & Humidity"):
        render_trends_page()
        render_temperature_page()
        render_heatmap_page()
        render_humidity_page()
        
    if _render_dashboard_section("Solar Analysis"):
        render_solar_page()
        
    if _render_dashboard_section("Psychrometrics"):
        try:
            render_psychrometrics_page()
        except Exception as e:
            st.error(f"❌ Psychrometrics error: {str(e)}")
            import traceback
            st.error(f"Details: {traceback.format_exc()}")
        
    if _render_dashboard_section("Wind"):
        render_wind_page()

    if _render_dashboard_section("Precipitation"):
        st.markdown("### 💧 Precipitation & Snow")
        st.caption("Understand rainfall and snowfall volume and seasonal distribution.")
        render_precipitation_thermal_load_page()
        st.divider()

    if _render_dashboard_section("Raw Data"):

        # ---- Data Quality (moved from former Data Quality tab) ----
        st.markdown("### 📋 Data completeness (non-null coverage)")
        st.caption("Quickly confirm which weather variables are fully populated and which ones have gaps before trusting downstream analytics.")
        null_ct = cdf.isna().sum()
        cov_pct = ((1 - null_ct / len(cdf)) * 100).round(1)

        cov_df = (
            pd.DataFrame({"Coverage %": cov_pct, "Missing": null_ct})
            .sort_values("Coverage %", ascending=True)
            .head(12)
        )

        if (cov_df["Coverage %"] == 100).all():
            st.success("All shown columns are complete (100% coverage).")
        st.dataframe(cov_df, use_container_width=True)
        st.caption("Data quality diagnostics shown above.")
        st.divider()

        # ---- Raw data table ----
        render_raw_data_page()

# ====================== TEMPERATURE & HUMIDITY (CLEAN) ======================

def render_trends_page():
    effective_page = st.session_state.get("nav_page")
    cdf = st.session_state.get("cdf")
    if cdf is None: return

    # Original 'if' check at 3527 is removed, but we check specific conditions if needed or rely on main() dispatch.
    # Since main() dispatches here only for "Temp & Humidity" etc., we are good.
    
    # We kept the body indentation (4 spaces), so this function wrapper fits perfectly.
    # if True: # removed to fix indentation error

    # -------------------- Controls --------------------
    c1, c2, c3 = st.columns([1.2, 1, 1.2])
    agg = c1.selectbox("Aggregation", ["Hourly", "Daily", "Monthly"], index=1)
    smooth_n = c2.slider("Smoothing (periods)", 1, 15, 7, help="Rolling mean applied after aggregation.")
    rh_lo, rh_hi = c3.slider("RH comfort band (%)", 0, 100, (30, 70))
    c4, c5 = st.columns([1, 1])
    temp_band = c4.selectbox("Temperature comfort band", ["None", "ASHRAE 80%", "ASHRAE 80% + 90%"], index=1)
    show_temp_range = c5.checkbox("Show temperature range", True)

    location_label = get_clean_city_name()
    st.markdown(f"<h3>{location_label} – Temperature & Humidity</h3>", unsafe_allow_html=True)
    st.caption("Clean reference plots with comfort ribbons and a single linked time window. Use this space to compare how temperature and humidity evolve at hourly, daily, or monthly scales.")


    # -------------------- Helpers --------------------
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    def resample_mean_range(series: pd.Series, g: str, smooth_override: int = None):
        """Return mean, min, max resampled series for Hourly/Daily/Monthly, on a dense index."""
        if g == "Hourly":
            rule = "1H"
        elif g == "Daily":
            rule = "1D"
        else:  # Monthly
            rule = "MS"
        # base aggregates
        s_mean = series.resample(rule).mean()
        s_min  = series.resample(rule).min()
        s_max  = series.resample(rule).max()
        # --- NEW: build a complete time index and reindex everything onto it
        if rule == "MS":
            # MonthBegin is non-fixed → compute month starts explicitly
            smin = series.index.min()
            smax = series.index.max()
            start = pd.Timestamp(smin.year, smin.month, 1)
            end   = pd.Timestamp(smax.year, smax.month, 1)
        else:
            start = series.index.min().floor(rule)
            end   = series.index.max().ceil(rule)
        idx_full = pd.date_range(start, end, freq=rule)

        s_mean = s_mean.reindex(idx_full)
        s_min  = s_min.reindex(idx_full)
        s_max  = s_max.reindex(idx_full)
        # --- NEW: interpolate/forward-fill to remove small gaps so lines don't break
        # mean: interpolate in time (best visual continuity)
        s_mean = s_mean.interpolate("time").ffill().bfill()
        # range: forward/back fill (interpolating a min/max does not make sense visually)
        s_min  = s_min.ffill().bfill()
        s_max  = s_max.ffill().bfill()
        
        # Determine effective smoothing
        # If Monthly, force 1 (raw arithmetic mean).
        # Else if override provided, use it.
        # Else use slider value.
        if g == "Monthly":
            eff_smooth = 1
        elif smooth_override is not None:
            eff_smooth = smooth_override
        else:
            eff_smooth = smooth_n

        if eff_smooth and eff_smooth > 1:
            s_mean = s_mean.rolling(eff_smooth, min_periods=1, center=True).mean()
        return s_mean, s_min, s_max

    def ashrae_adaptive_daily(drybulb_hourly: pd.Series):
        """
        ASHRAE adaptive 80% & 90% comfort bands based on running-mean outdoor temp.
        Trm ~ EWMA of DAILY mean dry-bulb (slow memory; feels right for the ribbon).
        """
        daily = drybulb_hourly.resample("1D").mean().dropna()
        if daily.empty:
            idx = daily.index
            z = pd.Series(index=idx, dtype=float)
            return z, z, z, z, z  # Trm, 80lo, 80hi, 90lo, 90hi
        # Shift by 1 so today's comfort limit depends on *previous* days' history
        Trm = daily.ewm(alpha=0.2, adjust=False).mean().shift(1).bfill()
        Tcomf = (0.31 * Trm + 17.8).clip(-30, 60)
        lo80, hi80 = Tcomf - 2.5, Tcomf + 2.5
        lo90, hi90 = Tcomf - 3.5, Tcomf + 3.5
        return Tcomf, lo80, hi80, lo90, hi90
    def upsample_to(idx: pd.DatetimeIndex, s: pd.Series):
        """Forward fill a lower-freq series to target index (for ribbons/guides)."""
        if s.empty:
            return pd.Series(index=idx, dtype=float)
        return s.reindex(idx.union(s.index)).ffill().reindex(idx)
    
    def split_segments(x, y, lo, hi):
        """Return dict of masked arrays for above / within / below comfort."""
        yv  = np.asarray(y,  dtype=float)
        lov = np.asarray(lo, dtype=float)
        hiv = np.asarray(hi, dtype=float)
        above  = np.where(yv >  hiv, yv, np.nan)
        within = np.where((yv >= lov) & (yv <= hiv), yv, np.nan)
        below  = np.where(yv <  lov, yv, np.nan)
        return {"above": (x, above), "within": (x, within), "below": (x, below)}

    # === X-labels per aggregation for hover ===
    def _xlabels_from_index(idx: pd.DatetimeIndex, agg: str) -> np.ndarray:
        if agg == "Hourly":
            # show hour prominently + date context
            return idx.strftime("Hour: %H:%M · %b %d").to_numpy()
        elif agg == "Daily":
            return idx.strftime("Date: %b %d").to_numpy()
        else:  # "Monthly"
            return idx.strftime("Month: %b").to_numpy()

    
    def add_range_bars(fig, x, s_min, s_max, name, color, row, col, opacity=0.28, color_arr=None, xlabels=None, unit_label=""):
        _color = color_arr.tolist() if color_arr is not None else color
        if xlabels is None:
            customdata = np.column_stack([s_min.values, ((s_min + s_max) / 2).values, s_max.values])
            hovertemplate = ("Avg: %{customdata[1]:.2f} " + unit_label + "<br>"
                            "Max: %{customdata[2]:.2f} " + unit_label + "<br>"
                            "Min: %{customdata[0]:.2f} " + unit_label + "<extra></extra>")
        else:
            customdata = np.column_stack([s_min.values, ((s_min + s_max) / 2).values, s_max.values, xlabels])
            hovertemplate = ("<b>%{customdata[3]}</b><br>"
                            "Avg: %{customdata[1]:.2f} " + unit_label + "<br>"
                            "Max: %{customdata[2]:.2f} " + unit_label + "<br>"
                            "Min: %{customdata[0]:.2f} " + unit_label + "<extra></extra>")

        fig.add_trace(go.Bar(
            x=x,
            y=(s_max - s_min),
            base=s_min,
            name=name,
            marker=dict(color=_color, opacity=opacity, line=dict(width=0)),
            customdata=customdata,
            hovertemplate=hovertemplate,
            showlegend=True
        ), row=row, col=col)




    # -------------------- Data --------------------
    # -------------------- Data --------------------
    # Canonicalize all timestamps to a single display year (so Jan..Dec always show)
    def _canon_index(idx: pd.DatetimeIndex, base_year: int = 2001) -> pd.DatetimeIndex:
        months = idx.month
        # handle Feb-29 safely
        days = np.minimum(idx.day, pd.to_datetime(
            [f"{base_year}-{m:02d}-01" for m in months]
        ).days_in_month.values)
        return pd.to_datetime(dict(year=np.full(len(idx), base_year),
                                month=months, day=days,
                                hour=idx.hour, minute=idx.minute, second=idx.second))
    cdf_can = cdf.copy()
    cdf_can.index = _canon_index(cdf_can.index, 2001)
    T_hourly  = cdf_can["drybulb"].dropna()
    RH_hourly = cdf_can["relhum"].dropna()

    # mean + range per aggregation
    # Plotting lines rely on smoothed data (default behavior of resample_mean_range)
    T_mean,  T_min,  T_max  = resample_mean_range(T_hourly, agg)
    RH_mean, RH_min, RH_max = resample_mean_range(RH_hourly, agg)

    # Calculate RAW (unsmoothed) means for the tooltip
    T_mean_raw,  _, _ = resample_mean_range(T_hourly, agg, smooth_override=1)
    RH_mean_raw, _, _ = resample_mean_range(RH_hourly, agg, smooth_override=1)

    # -------- Hover customdata for temperature (max, min, raw_mean, smoothed_mean) --------
    # Provide BOTH raw and smoothed means so the user sees the trend value (curve) AND the day's actual average
    temp_hover_cdata = np.c_[T_max.values, T_min.values, T_mean_raw.values, T_mean.values]

    # comfort (calculate on daily; project to the plotting index)
    Tcomf_d, T80_lo_d, T80_hi_d, T90_lo_d, T90_hi_d = ashrae_adaptive_daily(T_hourly)
    T80_lo = upsample_to(T_mean.index, T80_lo_d)
    T80_hi = upsample_to(T_mean.index, T80_hi_d)
    T90_lo = upsample_to(T_mean.index, T90_lo_d)
    T90_hi = upsample_to(T_mean.index, T90_hi_d)
    RH_lo  = pd.Series(rh_lo, index=RH_mean.index)
    RH_hi  = pd.Series(rh_hi, index=RH_mean.index)
    # Color palettes
    colT_above  = "#E74C3C"  # red
    colT_within = "#2ECC71"  # green
    colT_below  = "#3498DB"  # blue
    # RH: three shades of blue (dark=above, medium=within, light=below)
    colRH_above  = "#1B4F72"  # too humid
    colRH_within = "#5DADE2"  # comfort
    colRH_below  = "#AED6F1"  # too dry

    # ----- comfort colors per timestamp -----
    t_cat_colors = np.where(
        T_mean > T80_hi,  colT_above,
        np.where(T_mean < T80_lo, colT_below, colT_within)
    )
    rh_cat_colors = np.where(
        RH_mean > RH_hi, colRH_above,
        np.where(RH_mean < RH_lo, colRH_below, colRH_within)
    )

    # Build segmented lines on the resampled means
    segT  = split_segments(T_mean.index,  T_mean,  T80_lo, T80_hi)
    segRH = split_segments(RH_mean.index, RH_mean, RH_lo, RH_hi)
    # ---------- Hover customdata (Temp & RH) ----------
    # shape per point: [min, mean, max, month_str, day_int]
    # ---------- Hover labels (include aggregation-specific text) ----------
    xlab_T  = _xlabels_from_index(T_mean.index,  agg)
    xlab_RH = _xlabels_from_index(RH_mean.index, agg)

    # Hover customdata for mean lines: [min, mean_raw, max, xlabel, mean_smooth]
    # Hover customdata for mean lines: [min, mean_raw, max, xlabel, mean_smooth]
    def _hover_customdata(s_min, s_mean_raw, s_max, xlabels, s_mean_smooth):
        return np.column_stack([s_min.values, s_mean_raw.values, s_max.values, xlabels, s_mean_smooth.values])

    cd_T  = _hover_customdata(T_min,  T_mean_raw,  T_max,  xlab_T,  T_mean)
    cd_RH = _hover_customdata(RH_min, RH_mean_raw, RH_max, xlab_RH, RH_mean)

    # Month ticks & fixed window (Jan..Dec of display year 2001)
    month_ticks = pd.date_range(pd.Timestamp(2001, 1, 1), pd.Timestamp(2001, 12, 1), freq="MS")
    x_range = [pd.Timestamp(2001, 1, 1), pd.Timestamp(2001, 12, 31, 23, 59, 59)]





    # -------------------- Figure --------------------
    # Tighten the vertical spacing while keeping a unified time axis
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    # ===== Top: Temperature =====
    # (a) faint range as ribbon (min->max)
    # ===== Top: Temperature =====

    # Add the inner 80% band when either 80% or 80%+90% is selected
    if temp_band in ("ASHRAE 80%", "ASHRAE 80% + 90%"):
        # 80% top (legend on this one)
        fig.add_trace(go.Scatter(
            x=T_mean.index, y=T80_hi, mode="lines",
            line=dict(width=1.2, color="rgba(46, 204, 113, 0.9)"), hoverinfo="skip",
            name="ASHRAE adaptive comfort (80%)",
            showlegend=True, fill=None
        ), row=1, col=1)
        # 80% bottom (fills to the 80% top)
        fig.add_trace(go.Scatter(
            x=T_mean.index, y=T80_lo, mode="lines",
            line=dict(width=0), hoverinfo="skip",
            fill="tonexty", fillcolor="rgba(46, 204, 113, 0.22)",
            showlegend=False
        ), row=1, col=1)
    # --- Temperature ribbons: ABOVE (red), WITHIN (green), BELOW (blue) ---
    # ABOVE comfort: from T80_hi up to T_max (light red)
    fig.add_trace(go.Scatter(
        x=T_mean.index, y=T_max, mode="lines",
        line=dict(width=1.2, color="rgba(231, 76, 60, 0.85)"), hoverinfo="skip",
        name="Above comfort (T)", showlegend=True
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=T_mean.index, y=T80_hi, mode="lines",
        line=dict(width=0), hoverinfo="skip",
        fill="tonexty", fillcolor="rgba(231, 76, 60, 0.20)",
        showlegend=False
    ), row=1, col=1)
    # WITHIN comfort: from T80_lo to T80_hi (soft green)
    fig.add_trace(go.Scatter(
        x=T_mean.index, y=T80_hi, mode="lines",
        line=dict(width=1.2, color="rgba(46, 204, 113, 0.82)"), hoverinfo="skip",
        name="Within comfort (T)", showlegend=True
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=T_mean.index, y=T80_lo, mode="lines",
        line=dict(width=0), hoverinfo="skip",
        fill="tonexty", fillcolor="rgba(46, 204, 113, 0.22)",
        showlegend=False
    ), row=1, col=1)
    # BELOW comfort: from T_min up to T80_lo (light blue)
    fig.add_trace(go.Scatter(
        x=T_mean.index, y=T80_lo, mode="lines",
        line=dict(width=1.1, color="rgba(52, 152, 219, 0.78)"), hoverinfo="skip",
        name="Below comfort (T)", showlegend=True
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=T_mean.index, y=T_min, mode="lines",
        line=dict(width=0), hoverinfo="skip",
        fill="tonexty", fillcolor="rgba(52, 152, 219, 0.18)",
        showlegend=False
    ), row=1, col=1)
    
    # Seattle-style bars + mean line (Temperature)
    if show_temp_range:
        add_range_bars(
            fig, T_mean.index, T_min, T_max,
            name="Dry bulb temperature range", color="#E74C3C",
            row=1, col=1, opacity=0.35, color_arr=t_cat_colors, xlabels=xlab_T, unit_label="°C"
        )
    fig.add_trace(go.Scatter(
        x=T_mean.index, y=T_mean, mode="lines",
        name="Average Dry bulb temperature",
        line=dict(width=2.8, color="#ff5c52"),
        customdata=cd_T,
        hovertemplate="<b>%{customdata[3]}</b><br>Daily Mean: %{customdata[1]:.2f} °C<br>Smoothed: %{customdata[4]:.2f} °C<br>Max: %{customdata[2]:.2f} °C<br>Min: %{customdata[0]:.2f} °C<extra></extra>"
    ), row=1, col=1)

    

    # ===== Bottom: Relative Humidity =====
    # --- RH ribbons: ABOVE (dark blue), WITHIN (teal/greenish), BELOW (light blue) ---
    # ABOVE comfort: from RH_hi up to RH_max
    fig.add_trace(go.Scatter(
        x=RH_mean.index, y=RH_max, mode="lines",
        line=dict(width=1.1, color="rgba(27, 79, 114, 0.82)"), hoverinfo="skip",
        name="Above comfort (RH)", showlegend=True
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=RH_mean.index, y=RH_hi, mode="lines",
        line=dict(width=0), hoverinfo="skip",
        fill="tonexty", fillcolor="rgba(27, 79, 114, 0.20)",  # colRH_above @ ~20%
        showlegend=False
    ), row=2, col=1)
    # WITHIN comfort: from RH_lo to RH_hi
    fig.add_trace(go.Scatter(
        x=RH_mean.index, y=RH_hi, mode="lines",
        line=dict(width=1.1, color="rgba(93, 173, 226, 0.82)"), hoverinfo="skip",
        name="Within comfort (RH)", showlegend=True
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=RH_mean.index, y=RH_lo, mode="lines",
        line=dict(width=0), hoverinfo="skip",
        fill="tonexty", fillcolor="rgba(93, 173, 226, 0.22)",  # colRH_within slightly stronger
        showlegend=False
    ), row=2, col=1)
    # BELOW comfort: from RH_min up to RH_lo
    fig.add_trace(go.Scatter(
        x=RH_mean.index, y=RH_lo, mode="lines",
        line=dict(width=1.0, color="rgba(174, 214, 241, 0.74)"), hoverinfo="skip",
        name="Below comfort (RH)", showlegend=True
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=RH_mean.index, y=RH_min, mode="lines",
        line=dict(width=0), hoverinfo="skip",
        fill="tonexty", fillcolor="rgba(174, 214, 241, 0.18)",  # colRH_below light
        showlegend=False
    ), row=2, col=1)


    # (c) segmented RH mean line (three blues) + rich hover
    # Only show RH range bars when monthly to avoid a congested daily slider
    if agg == "Monthly":
        add_range_bars(
            fig, RH_mean.index, RH_min, RH_max,
            name="Relative humidity range", color="#1B4F72",
            row=2, col=1, opacity=0.5, color_arr=rh_cat_colors, xlabels=xlab_RH, unit_label="%"
        )
    fig.add_trace(go.Scatter(
        x=RH_mean.index, y=RH_mean, mode="lines",
        name="Average Relative humidity",
        line=dict(width=2.8, color="#7cc7ff"),
        customdata=cd_RH,
        hovertemplate="<b>%{customdata[3]}</b><br>Daily Mean: %{customdata[1]:.2f} %<br>Smoothed: %{customdata[4]:.2f} %<br>Max: %{customdata[2]:.2f} %<br>Min: %{customdata[0]:.2f} %<extra></extra>"
    ), row=2, col=1)

    # Range-slider preview: include both humidity and temperature with distinct styling
    fig.add_trace(go.Scatter(
        x=RH_mean.index,
        y=RH_mean,
        mode="lines",
        line=dict(width=1.2, color="rgba(93, 173, 226, 0.6)"),
        hoverinfo="skip",
        showlegend=False,
        name="RH (slider preview)",
        opacity=0.6,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=T_mean.index,
        y=T_mean,
        mode="lines",
        line=dict(width=1.0, color="rgba(231, 76, 60, 0.55)", dash="dot"),
        hoverinfo="skip",
        showlegend=False,
        name="Temperature (slider preview)",
        opacity=0.55,
        yaxis="y2"
    ), row=2, col=1)




    # -------------------- Axes / slider --------------------
    def xaxis_base():
        return dict(
            type="date",
            ticklabelmode="period",
            ticks="outside", ticklen=6,
            showgrid=True, gridcolor="rgba(255,255,255,0.08)",
            showline=True, linewidth=1.1, linecolor="rgba(255,255,255,0.35)",
            tickfont=dict(size=12), tickangle=0
        )

    # ---- Axis tick format by aggregation ----
    if agg == "Monthly":
        tickformat_main = "%b"        # Jan … Dec
        dtick_main = "M1"
    elif agg == "Daily":
        tickformat_main = "%b %d"     # Jan 15, …
        dtick_main = "M1"             # monthly stride keeps labels tidy
    else:  # Hourly
        tickformat_main = "%b"        # months for readability over a year
        dtick_main = "M1"


    # Top x (no slider)
    fig.update_xaxes(
        **xaxis_base(),
        matches="x",
        range=x_range,
        autorange=False,
        showticklabels=False,         # hide duplicate top ticks for a single shared axis
        rangeslider=dict(visible=False),
        tickformat=tickformat_main,   # <-- add
        dtick=dtick_main,             # <-- add
        row=1, col=1
    )

    fig.update_xaxes(
        **xaxis_base(),
        matches="x",
        range=x_range,
        autorange=False,
        fixedrange=False,
        rangeslider=dict(visible=True, thickness=0.10, bgcolor="rgba(255,255,255,0.03)"),
        rangeselector=dict(
            y=1.0, yanchor="top",
            buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(step="all")
            ]
        ),
        tickformat=tickformat_main,   # <-- add
        dtick=dtick_main,             # <-- add
        title_text="Month",
        title_standoff=30,
        row=2, col=1
    )


    # top y (T)
    fig.update_yaxes(
        title="Dry-bulb temperature (°C)",
        title_standoff=24,            # a little more space from ticks
        automargin=True,              # let Plotly grow the left margin if needed
        showticklabels=True,          # make sure labels are drawn
        tickfont=dict(size=12, color="rgba(240,240,240,0.96)"),  # visible on dark bg
        ticks="outside", ticklen=6,
        showgrid=True, gridcolor="rgba(255,255,255,0.08)",
        showline=True, linecolor="rgba(255,255,255,0.38)", linewidth=1.1,
        dtick=5,
        row=1, col=1
    )


    # bottom y (RH)
    fig.update_yaxes(
        title="Relative Humidity (%)",
        title_standoff=24,
        automargin=True,
        showticklabels=True,         # <- force tick labels
        tickfont=dict(size=12, color="rgba(240,240,240,0.96)"),
        ticks="outside", ticklen=6,
        showgrid=True, gridcolor="rgba(255,255,255,0.08)",
        showline=True, linecolor="rgba(255,255,255,0.38)", linewidth=1.1,
        range=[0, 100], dtick=10,
        row=2, col=1
    )

    # Reduce excess padding so the two panels read as a single, compact stack
    fig.update_layout(
        height=780,
        margin=dict(l=80, r=40, t=125, b=95),
        plot_bgcolor="rgba(12, 17, 26, 1)",
        paper_bgcolor="rgba(12, 17, 26, 1)",
        legend=dict(
            orientation="h",
            x=0,
            xanchor="left",
            y=1.12,
            yanchor="bottom",
            font=dict(color="#e5e7eb", size=10),
            bgcolor="rgba(12, 17, 26, 0.94)",
            bordercolor="rgba(148, 163, 184, 0.28)",
            borderwidth=1,
        ),
        hoverlabel=dict(bgcolor="#0f172a", font=dict(color="#e5e7eb"))
    )

    # Keep a single axis in the slider preview to ensure both traces render together
    # -------------------- Legend ordering (exact order requested) --------------------

    # ---------- FORCE LEGEND ORDER (must be RIGHT BEFORE plotly_chart) ----------
    # -------------------- FORCE LEGEND ORDER (bulletproof) --------------------
    desired_order = [
        ("Avg dry-bulb",          dict(mode="lines", line=dict(width=2.8, color="#ff5c52"))),
        ("Dry-bulb range",        dict(mode="markers", marker=dict(size=10, color="#E74C3C"))),
        ("T below comfort",       dict(mode="lines", line=dict(width=2, color="rgba(52, 152, 219, 0.78)"))),
        ("T within comfort",      dict(mode="lines", line=dict(width=2, color="rgba(46, 204, 113, 0.82)"))),
        ("T above comfort",       dict(mode="lines", line=dict(width=2, color="rgba(231, 76, 60, 0.85)"))),
        ("ASHRAE 80% band",       dict(mode="lines", line=dict(width=2, color="rgba(46, 204, 113, 0.9)"))),
        ("Avg RH",                dict(mode="lines", line=dict(width=2.8, color="#7cc7ff"))),
        ("RH below comfort",      dict(mode="lines", line=dict(width=2, color="rgba(174, 214, 241, 0.74)"))),
        ("RH within comfort",     dict(mode="lines", line=dict(width=2, color="rgba(93, 173, 226, 0.82)"))),
        ("RH above comfort",      dict(mode="lines", line=dict(width=2, color="rgba(27, 79, 114, 0.82)"))),
    ]

    # 1) Hide legend for ALL real traces (keeps visuals unchanged)
    for tr in fig.data:
        tr.showlegend = False

    # 2) Add legend-only dummy traces in the exact order you want
    for name, style in desired_order:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            name=name,
            hoverinfo="skip",
            showlegend=True,
            **style
        ))

    fig.update_layout(legend=dict(traceorder="normal"))
# -------------------------------------------------------------------------


    _st_plotly_chart(fig, use_container_width=True)
    _add_manual_pdf_figure("Annual Climate Statistics", fig)
    # Download buttons for Trends
    d1, d2 = st.columns(2)
    clean_loc = get_clean_city_name().replace(" ", "_").replace(",", "").replace("__", "_")
    with d1:
        try:
            svg_bytes = fig.to_image(format="svg", width=1400, height=800, scale=2)
            st.download_button("📥 Download Trends (SVG)", svg_bytes, f"{clean_loc}_trends_chart.svg", "image/svg+xml")
        except Exception as e:
            st.download_button("📥 Download Trends (SVG) - Unavailable", b"", disabled=True, help=f"PNG export failed. Requires working Kaleido installation. Error: {str(e)[:50]}")
    with d2:
        try:
            html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
            st.download_button("📥 Download Trends (HTML)", html_bytes, f"{clean_loc}_trends_chart.html", "text/html")
        except Exception as e:
            st.warning(f"Download Trends (HTML) failed: {e}")


# ------------- GENERIC CHART HELPERS -------------
def _render_bar_chart(cdf, col, title_suffix, y_label, color, key_suffix):
    import plotly.express as px
    st.markdown("---")
    location_label = get_clean_city_name()
    c1, c2 = st.columns([4, 1])
    with c1:
        st.markdown(f"<h4>{location_label} – {title_suffix}</h4>", unsafe_allow_html=True)
        st.caption("Monthly view only to keep the chart readable.")
    with c2:
        stat = st.selectbox("Statistic", ["Mean", "Median"], index=0, key=f"{col}_stat_{key_suffix}")
    df_col = cdf[[col]].dropna().copy()
    if df_col.empty:
        st.info(f"No data for {title_suffix}.")
        return
    df_col["month"] = df_col.index.month
    monthly = df_col.groupby("month")[col]
    y = monthly.mean() if stat == "Mean" else monthly.median()
    df_bar = y.reset_index().rename(columns={col: "VAL"})
    df_bar["month_name"] = df_bar["month"].apply(lambda m: pd.Timestamp(2001, int(m), 1).strftime("%b"))
    fig = px.bar(
        df_bar, x="month_name", y="VAL",
        labels={"month_name": "Month", "VAL": y_label},
        title=f"Monthly {stat.lower()} {format_figure_title(f'{col} Monthly Bar').replace('Monthly ', '')}"
    )
    fig.update_xaxes(type="category")
    fig.update_traces(marker_line_width=0.2, opacity=0.9, marker_color=color)
    fig.update_layout(
        yaxis=dict(title=y_label),
        margin=dict(l=40, r=20, t=60, b=30),
        legend=dict(orientation="h", x=0, y=1.02),
        bargap=0.15
    )
    _st_plotly_chart(fig, use_container_width=True)
    _add_manual_pdf_figure(f"{col} Monthly Bar", fig)
    d1, d2 = st.columns(2)
    clean_loc = get_clean_city_name().replace(" ", "_").replace(",", "").replace("__", "_")
    with d1:
        try:
            st.download_button(f"📥 Download {title_suffix} (SVG)", fig.to_image(format="svg", width=1200, height=600, scale=2), f"{clean_loc}_{col}_chart.svg", "image/svg+xml", key=f"dl_{col}_png_{key_suffix}")
        except Exception as e: 
            st.download_button(f"📥 Download {title_suffix} (SVG) - Unavailable", b"", disabled=True, key=f"dl_{col}_png_{key_suffix}_err", help=f"PNG export failed. Requires working Kaleido installation. Error: {str(e)[:50]}")
    with d2:
        try:
            st.download_button(f"📥 Download {title_suffix} (HTML)", fig.to_html(include_plotlyjs="cdn").encode("utf-8"), f"{clean_loc}_{col}_chart.html", "text/html", key=f"dl_{col}_html_{key_suffix}")
        except Exception as e:
            st.warning(f"Download {title_suffix} (HTML) failed: {e}")

def _render_daily_scatter(cdf, col, title_suffix, y_label, line_color, key_suffix):
    import plotly.express as px
    import plotly.graph_objects as go
    st.markdown("---")
    location_label = get_clean_city_name()
    st.markdown(f"<h4>{location_label} – {title_suffix}</h4>", unsafe_allow_html=True)
    st.caption("Each panel bundles all hours for a given month.")
    
    scat = cdf[[col]].dropna().copy()
    if scat.empty:
        st.info(f"No data for {title_suffix}.")
        return
    scat["month"] = scat.index.month
    scat["hour"]  = scat.index.hour
    ordered_months = list(range(1, 13))
    scat["month"]  = pd.Categorical(scat["month"], categories=ordered_months, ordered=True)
    curve = scat.groupby(["month", "hour"], observed=False)[col].mean().reset_index()
    curve["smooth"] = curve.groupby("month", observed=False)["hour"].transform(lambda h: 0)
    curve["smooth"] = curve.groupby("month", observed=False)[col].transform(lambda s: s.rolling(3, center=True, min_periods=1).mean())
    
    fig_sc = px.scatter(
        scat, x="hour", y=col,
        facet_col="month", facet_col_wrap=4,
        category_orders={"month": ordered_months},
        opacity=0.35,
        labels={"hour": "Hour", col: y_label},
        height=720, color_discrete_sequence=[line_color]
    )
    fig_ln = px.line(
        curve, x="hour", y="smooth",
        facet_col="month", facet_col_wrap=4,
        category_orders={"month": ordered_months},
    )
    for t in fig_ln.data:
        t.showlegend = False
        t.mode = "lines"
        t.line.width = 2
        t.line.color = line_color
        fig_sc.add_trace(t)
        
    fig_sc.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=7, color=line_color, opacity=0.6), name="Hourly points", showlegend=True), row=1, col=1)
    fig_sc.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(width=3, color=line_color), name="Monthly mean (smoothed)", showlegend=True), row=1, col=1)
    fig_sc.for_each_annotation(lambda a: a.update(text=pd.Timestamp(2001, int(a.text.split("=")[-1]), 1).strftime("%b"), y=a.y - 0.03))
    fig_sc.update_layout(legend=dict(orientation="h", x=0, xanchor="left", y=1.08, yanchor="bottom"), margin=dict(t=105, b=70, l=50, r=30))
    cname = get_clean_city_name().replace(" ", "_")
    _st_plotly_chart(fig_sc, use_container_width=True, config={"toImageButtonOptions": {"filename": f"{cname}_{col}_scatter", "format": "png", "scale": 2}, "displayModeBar": True})
    _add_manual_pdf_figure(f"{col} Hourly Dot Plot", fig_sc)
    d1, d2 = st.columns(2)
    with d1:
        try:
            st.download_button(f"📥 Download {title_suffix} (SVG)", fig_sc.to_image(format="svg", width=1400, height=800, scale=2), f"{cname}_{col}_scatter.svg", "image/svg+xml", key=f"dl_{col}_scat_png_{key_suffix}")
        except Exception as e: 
            st.download_button(f"📥 Download {title_suffix} (SVG) - Unavailable", b"", disabled=True, key=f"dl_{col}_scat_png_{key_suffix}_err", help=f"PNG export failed. Requires working Kaleido installation. Error: {str(e)[:50]}")
    with d2:
        try:
            st.download_button(f"📥 Download {title_suffix} (HTML)", fig_sc.to_html(include_plotlyjs="cdn").encode("utf-8"), f"{cname}_{col}_scatter.html", "text/html", key=f"dl_{col}_scat_html_{key_suffix}")
        except Exception as e:
            st.warning(f"Download {title_suffix} (HTML) failed: {e}")



_UTCI_BANDS = [
    (-100, -40, "Extreme Cold Stress (<-40°C)", "#1a1040"),
    (-40, -27, "Very Strong Cold Stress (-40 - -27°C)", "#2f3183"),
    (-27, -13, "Strong Cold Stress (-27 - -13°C)", "#3559a6"),
    (-13, 0, "Moderate Cold Stress (-13 - 0°C)", "#4bb3d4"),
    (0, 9, "Slight Cold Stress (0 - 9°C)", "#8fe0ee"),
    (9, 26, "No Stress (9 - 26°C)", "#a7f3d0"),
    (26, 32, "Moderate Heat Stress (26 - 32°C)", "#ffc080"),
    (32, 38, "Strong Heat Stress (32 - 38°C)", "#fc6554"),
    (38, 46, "Very Strong Heat Stress (38 - 46°C)", "#d12229"),
    (46, 100, "Extreme Heat Stress (>46°C)", "#7c1114"),
]

_PMV_BANDS = [
    (-100, -3.0, "Cold (< -3)", "#3559a6"),
    (-3.0, -2.0, "Cool (-3 to -2)", "#4bb3d4"),
    (-2.0, -1.0, "Slightly Cool (-2 to -1)", "#8fe0ee"),
    (-1.0, 1.0, "Neutral (-1 to +1)", "#a7f3d0"),
    (1.0, 2.0, "Slightly Warm (+1 to +2)", "#ffc080"),
    (2.0, 3.0, "Warm (+2 to +3)", "#fc6554"),
    (3.0, 100, "Hot (> +3)", "#d12229"),
]

_DI_BANDS = [
    (-100, 21.0, "Comfortable (<= 21)", "#a7f3d0"),
    (21.0, 24.0, "Slight Discomfort (21 - 24)", "#ffffb2"),
    (24.0, 27.0, "Discomfort (24 - 27)", "#fd8d3c"),
    (27.0, 29.0, "Strong Discomfort (27 - 29)", "#f03b20"),
    (29.0, 100, "Medical Emergency (> 29)", "#bd0026"),
]

def _render_categorical_heatmap(cdf, col, title_suffix, bands, key_suffix):
    import plotly.graph_objects as go
    import numpy as np
    st.markdown("---")
    location_label = get_clean_city_name()
    st.markdown(f"<h4>{location_label} – {title_suffix}</h4>", unsafe_allow_html=True)
    st.caption("Rows track hours and columns track calendar days. Legend colors represent discrete stress categories.")
    tmp = pd.DataFrame({"doy": cdf.index.dayofyear, "hour": cdf.index.hour, "val": cdf[col]}).dropna()
    if tmp.empty:
        st.info(f"No data for {title_suffix}.")
        return
    mat_df = tmp.pivot_table(index="doy", columns="hour", values="val", aggfunc="mean")
    #mat_df = mat_df.reindex(index=range(1, 366), columns=range(24))
    mat_df = mat_df.reindex(index=range(1, 367), columns=range(24))
    mat_val = mat_df.values
    mat_cat = np.full(mat_val.shape, np.nan)
    colors = []
    labels = []
    for i, (lower, upper, label, color) in enumerate(bands):
        mask = (mat_val > lower) & (mat_val <= upper)
        mat_cat[mask] = i
        colors.append(color)
        labels.append(label)
    n_colors = len(colors)
    bvals = np.linspace(0, 1, n_colors + 1)
    dscale = []
    for i in range(n_colors):
        dscale.append([bvals[i], colors[i]])
        dscale.append([bvals[i+1], colors[i]])
    fig = go.Figure()
    hovertext = np.full(mat_val.shape, "", dtype=object)
    n_doys = mat_val.shape[0]
    ref_year = 2021  # Non-leap reference year for DOY→date conversion
    for doy_i in range(n_doys):
        try:
            dt = pd.Timestamp(ref_year, 1, 1) + pd.Timedelta(days=doy_i)
            date_str = dt.strftime("%m/%d")
        except Exception:
            date_str = f"Day {doy_i+1}"
        for hr_i in range(24):
            v = mat_val[doy_i, hr_i]
            c = int(mat_cat[doy_i, hr_i]) if not np.isnan(mat_cat[doy_i, hr_i]) else -1
            time_str = f"{hr_i:02d}:00:00"
            if c != -1:
                hovertext[doy_i, hr_i] = f"{date_str} {time_str}<br>Val: {v:.1f}<br>{labels[c]}"
    fig.add_trace(go.Heatmap(
        x=mat_df.index, y=mat_df.columns,
        z=mat_cat.T,
        colorscale=dscale,
        zmin=0, zmax=n_colors-1,
        showscale=False,
        hoverinfo="text",
        text=hovertext.T
    ))
    for i, (lower, upper, label, color) in enumerate(bands):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=14, color=color, symbol="square", line=dict(width=1, color="#444")),
            name=label, showlegend=True, hoverinfo="skip"
        ))
    month_days = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    fig.update_xaxes(tickvals=month_days, ticktext=month_names, ticklen=5, range=[0.5, 366.5], showgrid=False)
    fig.update_yaxes(tickvals=[0, 6, 12, 18, 23], ticktext=["12AM", "6AM", "12PM", "6PM", "11PM"], autorange="reversed", ticklen=5, range=[-0.5, 23.5], showgrid=False)
    fig.update_layout(height=450, margin=dict(t=30, b=40, l=50, r=20), legend=dict(y=1, yanchor="top", x=1.02, xanchor="left", traceorder="reversed"))
    clean_loc = get_clean_city_name().replace(" ", "_").replace(",", "").replace("__", "_")
    _st_plotly_chart(fig, use_container_width=True, config={"toImageButtonOptions": {"filename": f"{clean_loc}_{col}_{key_suffix}", "format": "png"}, "displayModeBar": True})
    _add_manual_pdf_figure(f"{col} Annual Heatmap", fig)
    d1, d2 = st.columns(2)
    with d1:
        try:
            st.download_button(f"📥 Download {title_suffix} (SVG)", fig.to_image(format="svg", width=1200, height=600, scale=2), f"{clean_loc}_{col}_{key_suffix}.svg", "image/svg+xml", key=f"dl_{col}_{key_suffix}_svg")
        except Exception as e:
            st.download_button(f"📥 Download {title_suffix} (SVG) - Unavailable", b"", disabled=True, key=f"dl_{col}_{key_suffix}_err", help=f"SVG export failed. Requires working Kaleido installation. Error: Image export using the \"kaleido\" engine requires")
    with d2:
        try:
            st.download_button(f"📥 Download {title_suffix} (HTML)", fig.to_html(include_plotlyjs="cdn").encode("utf-8"), f"{clean_loc}_{col}_{key_suffix}.html", "text/html", key=f"dl_{col}_{key_suffix}_html")
        except Exception as e:
            st.warning(f"Download {title_suffix} (HTML) failed: {e}")

def _render_heatmap(cdf, col, title_suffix, y_label, color_scale):
    import plotly.graph_objects as go
    st.markdown("---")
    location_label = get_clean_city_name()
    st.markdown(f"<h4>{location_label} – {title_suffix}</h4>", unsafe_allow_html=True)
    st.caption("Rows track hours and columns track calendar days.")
    tmp = pd.DataFrame({"doy": cdf.index.dayofyear, "hour": cdf.index.hour, "val": cdf[col]}).dropna()
    if tmp.empty:
        st.info(f"No data for {title_suffix}.")
        return
    mat = (
        tmp.pivot_table(index="hour", columns="doy", values="val", aggfunc="mean")
        .reindex(index=range(24))
        .sort_index(axis=1)
    )
    ref_year = 2020 if 366 in mat.columns else 2021
    x_dates = [pd.Timestamp(ref_year, 1, 1) + pd.Timedelta(days=int(doy) - 1) for doy in mat.columns]
    month_starts = pd.date_range(pd.Timestamp(ref_year, 1, 1), pd.Timestamp(ref_year, 12, 1), freq="MS")
    fig_hm = go.Figure(
        go.Heatmap(
            z=mat.values,
            x=x_dates,
            y=mat.index,
            colorscale=color_scale,
            colorbar=dict(title=y_label),
            hovertemplate="Date: %{x|%b %d}<br>Hour: %{y}:00<br>Value: %{z:.1f} " + y_label + "<extra></extra>",
        )
    )
    fig_hm.update_xaxes(
        title="Calendar day",
        tickmode="array",
        tickvals=month_starts,
        ticktext=[d.strftime("%b") for d in month_starts],
        showgrid=False,
    )
    fig_hm.update_yaxes(
        title="Hour of day",
        tickmode="array",
        tickvals=[0, 6, 12, 18, 23],
        ticktext=["12AM", "6AM", "12PM", "6PM", "11PM"],
        autorange="reversed",
    )
    fig_hm.update_layout(height=420, margin=dict(l=55, r=60, t=25, b=55))
    _st_plotly_chart(fig_hm, use_container_width=True)
    _add_manual_pdf_figure(f"{col} Annual Heatmap", fig_hm)
    d1, d2 = st.columns(2)
    clean_loc = get_clean_city_name().replace(" ", "_").replace(",", "").replace("__", "_")
    with d1:
        try:
            st.download_button(f"📥 Download {title_suffix} (SVG)", fig_hm.to_image(format="svg", width=1200, height=600, scale=2), f"{clean_loc}_{col}_heatmap.svg", "image/svg+xml", key=f"dl_{col}_hm_png")
        except Exception as e: 
            st.download_button(f"📥 Download {title_suffix} (SVG) - Unavailable", b"", disabled=True, key=f"dl_{col}_hm_png_err", help=f"PNG export failed. Requires working Kaleido installation. Error: {str(e)[:50]}")
    with d2:
        try:
            st.download_button(f"📥 Download {title_suffix} (HTML)", fig_hm.to_html(include_plotlyjs="cdn").encode("utf-8"), f"{clean_loc}_{col}_heatmap.html", "text/html", key=f"dl_{col}_hm_html")
        except Exception as e:
            st.warning(f"Download {title_suffix} (HTML) failed: {e}")

def render_temperature_page():
    cdf = st.session_state.get("cdf")
    if cdf is None: return
    _render_bar_chart(cdf, "drybulb", "Temperature (Bar Chart)", "Temperature (°C)", "crimson", "temp_bar")
    _render_daily_scatter(cdf, "drybulb", "Daily scatter (hourly points, faceted by month)", "Dry-bulb temperature (°C)", "crimson", "temp_scat")

def render_heatmap_page():
    cdf = st.session_state.get("cdf")
    if cdf is None: return
    _render_heatmap(cdf, "drybulb", "Annual Heatmap (Hour x Day)", "°C", "RdYlBu_r")

def render_humidity_page():
    cdf = st.session_state.get("cdf")
    if cdf is None: return
    if "relhum" not in cdf:
        st.info("This EPW has no Relative Humidity column.")
        return
    _render_bar_chart(cdf, "relhum", "Humidity (Bar Chart)", "Relative Humidity (%)", "dodgerblue", "hum_bar")
    _render_daily_scatter(cdf, "relhum", "Humidity Daily scatter", "Relative Humidity (%)", "dodgerblue", "hum_scat")
    _render_heatmap(cdf, "relhum", "Humidity Annual Heatmap (Hour x Day)", "%", "Blues")

def render_utci_page():
    cdf = st.session_state.get("cdf")
    if cdf is None: return
    
    utci_df = st.session_state.get("comfort_pkg", {}).get("utci")
    
    if utci_df is None or utci_df.empty:
        st.info("UTCI index could not be computed. Ensure dry-bulb temperature, relative humidity, and wind speed are present.")
        return
        
    temp_df = cdf.copy()
    temp_df["utci_index"] = utci_df
    
    st.markdown("<h3>Universal Thermal Climate Index (UTCI)</h3>", unsafe_allow_html=True)
    st.caption("UTCI is an advanced thermal comfort metric that combines temperature, humidity, wind speed, and radiation estimating the physiological response of the human body.")
        
    _render_categorical_heatmap(temp_df, "utci_index", "UTCI Annual Heatmap", _UTCI_BANDS, "utci")

    st.markdown("---")
    st.markdown("### Advanced Comfort Diagnostics")
    st.caption("Publication-standard diagnostics generated by the PDF engine.")
    extra = _get_extra_figures()
    
    if "UTCI Annual Time Series" in extra:
        _st_plotly_chart(extra["UTCI Annual Time Series"], use_container_width=True)
        
    st.markdown("#### Diurnal Thermal Comfort Frequency Scenarios")
    st.caption("Evaluates the specific impact of shading, constant wind, and normalized humidity on thermal stress hours.")
    c1, c2, c3 = st.columns(3)
    if "Diurnal Thermal Comfort Frequency — Shading Scenario" in extra:
        with c1:
            _st_plotly_chart(extra["Diurnal Thermal Comfort Frequency — Shading Scenario"], use_container_width=True)
    if "Diurnal Thermal Comfort Frequency — Wind Scenario" in extra:
        with c2:
            _st_plotly_chart(extra["Diurnal Thermal Comfort Frequency — Wind Scenario"], use_container_width=True)
    if "Diurnal Thermal Comfort Frequency — Humidity Scenario" in extra:
        with c3:
            _st_plotly_chart(extra["Diurnal Thermal Comfort Frequency — Humidity Scenario"], use_container_width=True)


def render_pmv_page():
    try:
        import pythermalcomfort  # noqa
    except ImportError:
        st.error("PMV requires the `pythermalcomfort` package. Add it to requirements.txt and redeploy.")
        return

    cdf = st.session_state.get("cdf")
    if cdf is None: return
    
    pmv_df = st.session_state.get("comfort_pkg", {}).get("pmv")
    
    if pmv_df is None or (hasattr(pmv_df, 'empty') and pmv_df.empty):
        # Try computing PMV on the fly and show specific error
        missing = [c for c in ("drybulb", "relhum", "windspd") if c not in cdf.columns]
        if missing:
            st.info(f"PMV index could not be computed. Missing columns: {', '.join(missing)}. Ensure dry-bulb temperature, relative humidity, and wind speed are present.")
        else:
            try:
                pmv_df = ce.compute_pmv(cdf)
                if pmv_df is not None and not pmv_df.empty:
                    st.session_state.setdefault("comfort_pkg", {})["pmv"] = pmv_df
                else:
                    st.info("PMV index returned empty results. The pythermalcomfort library may need to be installed or updated.")
                    return
            except Exception as exc:
                import traceback
                st.error(f"PMV computation failed: {exc}")
                with st.expander("Traceback (for debugging)"):
                    st.code(traceback.format_exc())
                return
        if pmv_df is None or (hasattr(pmv_df, 'empty') and pmv_df.empty):
            return
        
    temp_df = cdf.copy()
    temp_df["pmv_index"] = pmv_df
    
    st.markdown("<h3>Predicted Mean Vote (PMV)</h3>", unsafe_allow_html=True)
    st.caption("PMV is a thermal comfort index that predicts the mean value of the thermal votes of a large group of people on a 7-point thermal sensation scale (-3 cold to +3 hot).")
        
    _render_categorical_heatmap(temp_df, "pmv_index", "PMV Annual Heatmap", _PMV_BANDS, "pmv")

def render_di_page():
    cdf = st.session_state.get("cdf")
    if cdf is None: return
    
    di_df = st.session_state.get("comfort_pkg", {}).get("di")
    
    if di_df is None or di_df.empty:
        st.info("DI index could not be computed. Ensure dry-bulb temperature and relative humidity are present.")
        return
        
    temp_df = cdf.copy()
    temp_df["di_index"] = di_df
    
    st.markdown("<h3>Discomfort Index (DI)</h3>", unsafe_allow_html=True)
    st.caption("Thom's Discomfort Index (DI) is an indicator of heat stress combining temperature and humidity.")
    
    st.markdown("#### Custom Discomfort Band")
    st.caption("Select the temperature band at which you feel discomfort.")
    
    # Initialize slider state
    if "di_discomfort_band" not in st.session_state:
        st.session_state["di_discomfort_band"] = (24.0, 27.0)

    # Unit based conversion
    unit_label = "°F" if _temp_unit() == "F" else "°C"
    if _temp_unit() == "F":
        min_v = int(round(_c_to_f(0.0)))
        max_v = int(round(_c_to_f(45.0)))
        val = (int(round(_c_to_f(st.session_state["di_discomfort_band"][0]))),
               int(round(_c_to_f(st.session_state["di_discomfort_band"][1]))))
        slider_val = st.slider(f"Discomfort Band ({unit_label})", min_value=min_v, max_value=max_v, value=val, step=1)
        st.session_state["di_discomfort_band"] = (_f_to_c(slider_val[0]), _f_to_c(slider_val[1]))
    else:
        min_v = 0.0
        max_v = 45.0
        val = st.session_state["di_discomfort_band"]
        slider_val = st.slider(f"Discomfort Band ({unit_label})", min_value=min_v, max_value=max_v, value=val, step=0.5)
        st.session_state["di_discomfort_band"] = slider_val

    # Read the updated bounds (in C)
    low_bound, high_bound = st.session_state["di_discomfort_band"]
    
    # Calculate discomfort hours
    discomfort_mask = (temp_df["di_index"] >= low_bound) & (temp_df["di_index"] <= high_bound)
    discomfort_hours = int(discomfort_mask.sum())
    total_hours = int(temp_df["di_index"].notna().sum())
    if total_hours > 0:
        pct = (discomfort_hours / total_hours) * 100
    else:
        pct = 0.0
        
    st.metric(f"Hours within Discomfort Band ({format_temperature(low_bound)} to {format_temperature(high_bound)})", f"{discomfort_hours} h", f"{pct:.1f}% of year", delta_color="off")
    
    _render_categorical_heatmap(temp_df, "di_index", "DI Annual Heatmap", _DI_BANDS, "di")

# ---------------------- SUN & CLOUDS ----------------------


# ---------- Data model ----------

@dataclass
class Site:
    lat: float
    lon: float
    tz: object     # was str
    elev_m: float = 0.0
    north_deg_ccw_from_y: float = 0.0

@dataclass
class Options2D:
    projection: str = "stereographic"  # "stereographic" | "orthographic"
    radius: float = 1.0
    show_hour_labels: bool = True
    hour_label_step: int = 1
    show_envelope: bool = True
    show_annual_grid: bool = True
    use_solar_time: bool = False

@dataclass
class Options3D:
    show_3d: bool = False
    show_rays: bool = True
    # Simple building massing: list of prisms (x, y, w, d, z0, z1)
    massing: Optional[List[Tuple[float, float, float, float, float, float]]] = None


# --- ADVANCED FIGURE BUILDERS ---

def build_fig_a_longwave_heatmap(df, station_name):
    df = _prepare_advanced_figure_df(df)
    lw_col = resolve_longwave_column(df)
    if not lw_col:
        return placeholder_figure("Longwave data not available in source EPW file."), ""
    
    lw_data = pd.to_numeric(df[lw_col], errors='coerce')
    mat = _advanced_day_hour_matrix(df, lw_data)
    if mat.empty:
        return placeholder_figure("Not enough hourly data to build longwave heatmap."), ""
    fig = go.Figure(data=go.Heatmap(
        z=mat.values,
        x=mat.columns,
        y=mat.index,
        colorscale='Reds',
        colorbar=dict(title='W/m²')
    ))
    fig.update_layout(
        title="Longwave Horizontal Irradiance by Hour and Day",
        xaxis_title="Day of Year",
        yaxis_title="Hour of Day",
        yaxis=dict(autorange="reversed"),
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    month_index = pd.Series(df.index.month, index=df.index) if isinstance(df.index, pd.DatetimeIndex) else pd.to_numeric(df.get("month"), errors="coerce")
    summer_peak_lw = lw_data[month_index.isin([6, 7, 8])].mean().round(1)
    winter_min_lw = lw_data[month_index.isin([12, 1, 2])].mean().round(1)
    
    caption_template = "Longwave horizontal irradiance is mapped by day of year and hour of day, revealing the site's background thermal radiation environment driven by sky and ground temperatures. For {station}, summer mean longwave flux is {summer_peak_lw} W/m² — elevating nighttime envelope heat load — while winter mean of {winter_min_lw} W/m² indicates strong radiative cooling potential after sunset."
    caption = safe_format_caption(caption_template, {
        "station": station_name,
        "summer_peak_lw": summer_peak_lw,
        "winter_min_lw": winter_min_lw
    })
    return fig, caption

def build_fig_b_hourly_incoming_radiation_boxwhisker(df, station_name):
    df = _prepare_advanced_figure_df(df)
    lw_col = resolve_longwave_column(df)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    if isinstance(df.index, pd.DatetimeIndex):
        month_series = pd.Series(df.index.month, index=df.index)
    else:
        month_series = pd.to_numeric(df.get("month"), errors="coerce")
    month_arr = month_series.map(lambda x: months[int(x)-1] if pd.notna(x) and 1 <= int(x) <= 12 else None)
    
    fig = go.Figure()
    available = False
    for col, color, name in [('dni_Wm2', 'red', 'DNI'), ('dhi_Wm2', 'blue', 'DHI'), ('ghi_Wm2', 'orange', 'GHI')]:
        if col not in df.columns:
            continue
        available = True
        fig.add_trace(go.Box(
            y=df[col], x=month_arr, name=name, marker_color=color, boxpoints=False, whiskerwidth=0.5
        ))
    if lw_col:
        available = True
        fig.add_trace(go.Box(
            y=df[lw_col], x=month_arr, name='Longwave', marker_color='purple', boxpoints=False, whiskerwidth=0.5
        ))
    if not available:
        return placeholder_figure("Incoming radiation components are not available in this source file."), ""
    
    fig.update_layout(
        boxmode='group',
        height=400,
        yaxis_title="Irradiance (W/m²)",
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    monthly_ghi = df['ghi_Wm2'].groupby(month_series).sum() if "ghi_Wm2" in df.columns else pd.Series(dtype=float)
    monthly_dhi = df['dhi_Wm2'].groupby(month_series).sum() if "dhi_Wm2" in df.columns else pd.Series(dtype=float)
    ghi_peak_idx = monthly_ghi.dropna().idxmax() if not monthly_ghi.dropna().empty else np.nan
    peak_solar_month = months[int(ghi_peak_idx) - 1] if pd.notna(ghi_peak_idx) else "[data unavailable]"
    ratio = monthly_dhi / monthly_ghi.replace(0, np.nan)
    ratio_peak_idx = ratio.dropna().idxmax() if not ratio.dropna().empty else np.nan
    high_overcast_month = months[int(ratio_peak_idx) - 1] if pd.notna(ratio_peak_idx) else "[data unavailable]"
    
    caption_template = "Monthly box-whisker plots show the distribution of hourly direct normal, diffuse horizontal, global horizontal, and longwave radiation with interquartile boxes and full-range whiskers. For {station}, GHI peaks in {peak_solar_month} while the diffuse-to-global ratio is highest in {high_overcast_month}, signaling when shading devices can be relaxed and diffuse daylighting strategies should dominate."
    caption = safe_format_caption(caption_template, {
        "station": station_name,
        "peak_solar_month": peak_solar_month,
        "high_overcast_month": high_overcast_month
    })
    return fig, caption

def build_fig_c_inclined_surface_insolation(df, station_name, epw_metadata):
    df = _prepare_advanced_figure_df(df)
    if not PVLIB_AVAILABLE:
        return placeholder_figure("pvlib not installed — install pvlib to enable inclined surface analysis."), ""
    
    missing = [c for c in ["dni_Wm2", "ghi_Wm2", "dhi_Wm2"] if c not in df.columns]
    if missing:
        return placeholder_figure("DNI, GHI, and DHI are required for inclined surface analysis."), ""

    latitude = epw_metadata.get('latitude') or epw_metadata.get('lat') or df.attrs.get('latitude', None)
    longitude = epw_metadata.get('longitude') or epw_metadata.get('lon') or df.attrs.get('longitude', 0)
    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except Exception:
        latitude = None
    if latitude is None:
        fig = placeholder_figure("Latitude not available in EPW metadata.")
        fig.add_annotation(text="Optimal fixed tilt: None° from horizontal", x=0.5, y=0.1, showarrow=False)
        return fig, "Latitude not available for inclined surface calculations."
        
    try:
        from pvlib import location, irradiance
        loc = location.Location(latitude, longitude)
        times = df.index
        if not isinstance(times, pd.DatetimeIndex):
            return placeholder_figure("Datetime index is required for inclined surface analysis."), ""
        solar_position = loc.get_solarposition(times)
        solar_position = _drop_datetime_timezone(solar_position)
        times_for_grouping = times.tz_localize(None) if getattr(times, "tz", None) is not None else times
        
        # Tilt sweep
        tilts = list(range(0, 71, 10))
        surfaces = [(0, 180, 'Horizontal'), (90, 180, 'S-Vert'), (90, 0, 'N-Vert')] + [(t, 180, f'{t}° S') for t in tilts]
        
        monthly_totals = {}
        annual_totals = {}
        for tilt, azimuth, name in surfaces:
            poa = irradiance.get_total_irradiance(
                surface_tilt=tilt,
                surface_azimuth=azimuth,
                dni=pd.to_numeric(df['dni_Wm2'], errors="coerce").fillna(0).to_numpy(float),
                ghi=pd.to_numeric(df['ghi_Wm2'], errors="coerce").fillna(0).to_numpy(float),
                dhi=pd.to_numeric(df['dhi_Wm2'], errors="coerce").fillna(0).to_numpy(float),
                solar_zenith=pd.to_numeric(solar_position['apparent_zenith'], errors="coerce").to_numpy(float),
                solar_azimuth=pd.to_numeric(solar_position['azimuth'], errors="coerce").to_numpy(float)
            )
            poa_global = pd.Series(np.asarray(poa['poa_global'], dtype=float), index=times_for_grouping).fillna(0)
            monthly = poa_global.groupby(times_for_grouping.month).sum() / 1000.0  # kWh/m2
            monthly_totals[name] = monthly
            annual_totals[name] = float(poa_global.sum()) / 1000.0
            
        optimal_tilt_tuple = max([(t, 180, f'{t}° S') for t in tilts], key=lambda x: annual_totals[x[2]])
        optimal_tilt = optimal_tilt_tuple[0]
        optimal_name = optimal_tilt_tuple[2]
        
        fig = go.Figure()
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        plot_surfaces = ['Horizontal', 'S-Vert', optimal_name, 'N-Vert']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for name, color in zip(plot_surfaces, colors):
            fig.add_trace(go.Bar(
                x=months, y=monthly_totals[name].values, name=name, marker_color=color
            ))
            
        fig.update_layout(
            barmode='group',
            yaxis_title="Total Insolation (kWh/m²)",
            height=400,
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.add_annotation(text=f"Optimal fixed tilt: {optimal_tilt}° from horizontal", x=0.01, y=0.99, xref="paper", yref="paper", showarrow=False, align="left")
        
        annual_kwh = round(annual_totals[optimal_name], 0)
        horiz_kwh = round(annual_totals['Horizontal'], 0)
        tilt_gain_pct = round((annual_kwh - horiz_kwh) / horiz_kwh * 100, 1) if horiz_kwh > 0 else 0
        
        caption_template = "Total monthly solar insolation on horizontal, vertical, and south-facing tilted surfaces reveals the performance gain from inclination at this latitude. For {station} at {latitude}°N, the optimal fixed tilt of {optimal_tilt}° from horizontal maximises annual collection at {annual_kwh} kWh/m²/yr — {tilt_gain_pct}% more than horizontal — a key input for PV panel orientation, solar thermal collectors, and passive south-facade sizing."
        caption = safe_format_caption(caption_template, {
            "station": station_name,
            "latitude": round(latitude, 1),
            "optimal_tilt": optimal_tilt,
            "annual_kwh": annual_kwh,
            "tilt_gain_pct": tilt_gain_pct
        })
        return fig, caption
    except Exception as e:
        return placeholder_figure(f"pvlib calculation error: {str(e)}"), ""

def render_solar_page():
    effective_page = st.session_state.get("nav_page")
    # Solar page logic handled self-contained data loading if needed, or uses session state
    if not PVLIB_AVAILABLE:
        st.warning(
            "Solar analysis is unavailable because pvlib dependencies could not be loaded on this machine. "
            "Windows Application Control appears to be blocking an h5py DLL."
        )
        if PVLIB_IMPORT_ERROR is not None:
            st.caption(f"Import detail: {type(PVLIB_IMPORT_ERROR).__name__}: {PVLIB_IMPORT_ERROR}")
        st.info(
            "You can still use the rest of the dashboard. To re-enable solar features, allow the blocked DLL "
            "or run in a Python environment where pvlib+h5py can load successfully."
        )

        # Fallback: render EPW-native solar/cloud charts that do not depend on pvlib.
        cdf = st.session_state.get("cdf")
        if cdf is None or cdf.empty:
            return

        location_label = get_clean_city_name()
        st.markdown(f"<h3>{location_label} – Solar Analysis (Fallback)</h3>", unsafe_allow_html=True)
        st.caption("Showing irradiance and cloud plots that do not require pvlib.")

        _ghi_col = get_metric_column(cdf, ["glohorrad", "ghi", "global_horizontal", "global_horiz", "solar", "radiation"])
        _dhi_col = get_metric_column(cdf, ["difhorrad", "dhi", "diffuse_horizontal", "dif_hor_rad"])
        _dni_col = get_metric_column(cdf, ["dirnorrad", "dni", "direct_normal", "dir_nor_rad"])

        if _ghi_col or _dhi_col or _dni_col:
            fig_solar = go.Figure()
            if _ghi_col:
                s = pd.to_numeric(cdf[_ghi_col], errors="coerce").dropna()
                if not s.empty:
                    d = pd.DataFrame({"doy": s.index.dayofyear, "val": s.values}).groupby("doy")["val"].mean()
                    fig_solar.add_trace(go.Scatter(x=d.index, y=d.values, mode="lines", name="GHI", line=dict(color="#fbbf24", width=2)))
            if _dni_col:
                s = pd.to_numeric(cdf[_dni_col], errors="coerce").dropna()
                if not s.empty:
                    d = pd.DataFrame({"doy": s.index.dayofyear, "val": s.values}).groupby("doy")["val"].mean()
                    fig_solar.add_trace(go.Scatter(x=d.index, y=d.values, mode="lines", name="DNI", line=dict(color="#f97316", width=2, dash="dot")))
            if _dhi_col:
                s = pd.to_numeric(cdf[_dhi_col], errors="coerce").dropna()
                if not s.empty:
                    d = pd.DataFrame({"doy": s.index.dayofyear, "val": s.values}).groupby("doy")["val"].mean()
                    fig_solar.add_trace(go.Scatter(x=d.index, y=d.values, mode="lines", name="DHI", line=dict(color="#60a5fa", width=2, dash="dash")))

            fig_solar.update_layout(
                title="Solar Radiation — Annual Daily Means (W/m²)",
                yaxis_title="Irradiance (W/m²)",
                height=340,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            _st_plotly_chart(fig_solar, use_container_width=True)
            _add_manual_pdf_figure("Monthly Solar Insolation", fig_solar)
        if "totskycvr" in cdf:
            cc = cdf[["totskycvr"]].copy()
            cc["month"] = cc.index.month

            def _bucket_cloud(v):
                if pd.isna(v):
                    return np.nan
                if v <= 3:
                    return "Clear (0–3/10)"
                if v <= 7:
                    return "Intermediate (4–7/10)"
                return "Cloudy (8–10/10)"

            cc["category"] = cc["totskycvr"].apply(_bucket_cloud)
            counts = cc.value_counts(["month", "category"]).rename("n").reset_index()
            counts["pct"] = 100 * counts["n"] / counts.groupby("month")["n"].transform("sum")
            freq = counts.drop(columns="n")
            fig_cloud = px.bar(
                freq,
                x="month",
                y="pct",
                color="category",
                barmode="stack",
                labels={"month": "Month", "pct": "% of hours", "category": ""},
                title="Cloud coverage by month (stacked frequency)",
            )
            _st_plotly_chart(fig_cloud, use_container_width=True)
            _add_manual_pdf_figure("Cloud Coverage", fig_cloud)
            tmp = pd.DataFrame({"doy": cdf.index.dayofyear, "hour": cdf.index.hour, "val": cdf["totskycvr"]}).dropna()
            if not tmp.empty:
                mat = tmp.pivot_table(index="hour", columns="doy", values="val", aggfunc="mean").sort_index()
                fig_cloud_hm = px.imshow(
                    mat.values,
                    origin="lower",
                    aspect="auto",
                    labels=dict(x="Day of Year", y="Hour", color="Total sky cover (tenths)"),
                    title="Annual heatmap — Total sky cover (tenths)",
                    height=340,
                    color_continuous_scale="Blues",
                )
                _st_plotly_chart(fig_cloud_hm, use_container_width=True)
                _add_manual_pdf_figure("Cloud Coverage Heatmap", fig_cloud_hm)
        return
    


# sunpath.py
# A compact, production-minded sun-path plotter (2D angular + optional 3D)
# Features:
# - Compass roses (N, NE, E, SE, S, SW, W, NW)
# - Elevation circles every 10° (altitude)
# - Seasonal envelopes (solstice/equinox daily arcs + light annual grid)
# - Hour markings on the selected date path (solar time option)
# - Optional 3D sky dome + massing blocks + sun rays
#
# Deps: pvlib, numpy, pandas, matplotlib
#
# Usage example (CLI):
#   python sunpath.py --lat 42.365 --lon -71.009 --tz "America/New_York" \
#     --date 2025-09-23 --projection stereographic --show3d
#
# Streamlit tip:
# - Wrap the draw_* calls inside Streamlit pages; keep figures separate.




    # ---------- Data model moved to global scope ----------


    # ---------- Solar helpers ----------

    def solar_positions(site: Site, date: pd.Timestamp) -> pd.DataFrame:
        """Hourly sun positions for a given civil date (local time: site.tz)."""
        # Local times from 0..23 at whole hours
        idx = pd.date_range(
            start=pd.Timestamp(date.date(), tz=site.tz),
            periods=24, freq="h", tz=site.tz
        )
        solpos = pvlib.solarposition.get_solarposition(
            idx, site.lat, site.lon, altitude=site.elev_m
        )[["apparent_zenith", "azimuth", "elevation"]]
        # elevation = 90 - zenith; pvlib already provides elevation
        return solpos


    def solar_positions_solar_time(df: pd.DataFrame, site: Site) -> pd.DataFrame:
        """Shift timestamps to apparent solar time (hour-angle basis)."""
        ts = df.index  # tz-aware DatetimeIndex

        # --- robust Equation of Time (EoT) ---
        if hasattr(pvlib.solarposition, "equation_of_time"):
            # Newer pvlib: accepts timestamps directly
            eot = pvlib.solarposition.equation_of_time(ts)  # minutes
        elif hasattr(pvlib.solarposition, "equation_of_time_spencer71"):
            # Older pvlib: expects DAY-OF-YEAR numbers (NOT timestamps)
            # Ensure plain integer array to avoid DatetimeArray ops
            doy = pd.Index(ts.dayofyear).to_numpy()
            eot = pvlib.solarposition.equation_of_time_spencer71(doy)  # minutes
        else:
            # NOAA approximation (minutes)
            day_angle = 2*np.pi*(ts.dayofyear - 1)/365.0
            eot = 229.18*(0.000075 + 0.001868*np.cos(day_angle) - 0.032077*np.sin(day_angle)
                        - 0.014615*np.cos(2*day_angle) - 0.040849*np.sin(2*day_angle))

        # Local Standard Meridian (deg) from timezone offset at that date
        offset_hours = ts[0].utcoffset().total_seconds() / 3600.0
        lsm = 15.0 * offset_hours

        # Time correction (minutes): positive => solar time ahead of clock time
        time_correction_min = eot + 4.0 * (site.lon - lsm)

        out = df.copy()
        out["solar_time"] = ts + pd.to_timedelta(time_correction_min, unit="m")
        return out




    # ---------- Projections (Angular plot) ----------

    def project_angular(az_deg: np.ndarray, alt_deg: np.ndarray, projection: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map azimuth (0=N, 90=E) & altitude (0..90) to plane:
        - Stereographic: r = 2 * tan((90-alt)/2)
        - Orthographic:  r = sin(90-alt)
        Output radius normalized so horizon is R=1.
        """
        alt = np.deg2rad(alt_deg)
        zen = np.deg2rad(90.0 - alt_deg)

        if projection.lower().startswith("stereo"):
            r = 2.0 * np.tan(zen / 2.0)
            R_h = 2.0 * np.tan(np.deg2rad(90.0) / 2.0)  # -> infinite; clamp using zen=90 -> large
            # Normalize by horizon radius (zen→90° -> r→∞). We cap at horizon ring = 1
            # Practical trick: map r via arctan scaling for a finite rim=1.
            r = np.arctan(r) / (np.pi/2)  # maps [0,∞) -> [0,1)
        elif projection.lower().startswith("ortho"):
            r = np.sin(zen)  # 0 at zen=0, 1 at zen=90
        else:
            raise ValueError("projection must be 'stereographic' or 'orthographic'.")

        # Azimuth: 0=N, 90=E -> turn to math angle (0° at +Y, clockwise)
        # For a compass-like plot (0 at top, increasing clockwise):
        theta = np.deg2rad(az_deg)
        # Convert to screen x,y with 0° at North (up)
        x = r * np.sin(theta)
        y = r * np.cos(theta) * -1.0  # flip so North at top, East right
        return x, y

        # --- Temperature color + EPW helpers ---

    # 1) Map temp (°C) to a color; or fall back to season color when requested.
    def _map_color(temp_c: float, month: int, mode: str = "temperature") -> str:
        if mode == "season":
            season_colors = {12:"#2D7DD2", 1:"#2D7DD2", 2:"#2D7DD2",   # winter blue
                            3:"#00B0F0", 4:"#00B0F0", 5:"#00B0F0",   # spring cyan
                            6:"#FF6B3D", 7:"#FF6B3D", 8:"#FF6B3D",   # summer orange-red
                            9:"#FFA14A",10:"#FFA14A",11:"#FFA14A"}   # autumn orange
            return season_colors.get(month, "#AAAAAA")

        # temperature mode (cool→blue, warm→red). Use Plotly's RdYlBu_r range.
        import matplotlib
        cmap = matplotlib.colormaps["RdYlBu_r"]
        # clamp to [-10, 35] for stable scale
        t = max(-10.0, min(35.0, float(temp_c)))
        val = (t + 10.0) / 45.0
        return matplotlib.colors.to_hex(cmap(val))

    # 2) Attach solar-time column to a DF (you already have this for 2D; re-use here)
    def _with_solar_time(df: pd.DataFrame, site: Site) -> pd.DataFrame:
        df = solar_positions_solar_time(df, site)
        return df

    # 3) Given EPW dataframe and a Series of solar_time timestamps, return nearest dry-bulb temps (°C)
    def _nearest_temp_by_solar_time(epw: pd.DataFrame, solar_time: pd.Series, max_gap="2H") -> pd.Series:
        """
        For each solar_time timestamp, return nearest dry-bulb temp from epw.
        Uses merge_asof (sorted ascending) and handles tz alignment + index naming.
        """
        # 1) find temp column
        temp_col = next((c for c in ["temp_air","DryBulb","Dry_Bulb","drybulb","Temperature"] if c in epw.columns), None)
        if temp_col is None:
            raise ValueError("Dry-bulb temperature column not found in EPW.")

        # 2) tz-align (use tz from solar_time)
        stz = solar_time.dt.tz
        if stz is None:
            raise ValueError("solar_time must be timezone-aware.")
        if epw.index.tz is None:
            epw = epw.tz_localize(stz)

        # convert both to UTC for clean asof
        epw_utc   = epw.tz_convert("UTC")
        solar_utc = solar_time.dt.tz_convert("UTC")

        # 3) de-dup + sort
        epw_utc = epw_utc[~epw_utc.index.duplicated(keep="first")].sort_index()

        # 4) build right table with a guaranteed 'ts' column
        idx_name = epw_utc.index.name if epw_utc.index.name is not None else "index"
        right = epw_utc.reset_index().rename(columns={idx_name: "ts"})
        right_sorted = right.sort_values("ts")

        # 5) left table (solar_time), sorted
        left = pd.DataFrame({"solar_time": solar_utc})
        left["_orig_order"] = np.arange(len(left))
        left_sorted = left.sort_values("solar_time")

        # 6) nearest join with tolerance
        out = pd.merge_asof(
            left_sorted,
            right_sorted[["ts", temp_col]],
            left_on="solar_time",
            right_on="ts",
            direction="nearest",
            tolerance=pd.Timedelta(max_gap),
        )

        # 7) restore original order and index to original solar_time
        out = out.sort_values("_orig_order").reset_index(drop=True)
        return pd.Series(out[temp_col].to_numpy(), index=solar_time.index, name="temp_c")


    def _nearest_epw_by_solar_time(epw: pd.DataFrame, solar_time: pd.Series, preferred_cols: list[str], max_gap: str = "2H") -> Optional[pd.Series]:
        """Nearest EPW variable to solar_time using merge_asof; returns Series aligned to solar_time index or None if missing."""
        col = next((c for c in preferred_cols if c in epw.columns), None)
        if col is None:
            return None

        stz = solar_time.dt.tz
        if stz is None:
            return None
        epw_work = epw.copy()
        if epw_work.index.tz is None:
            epw_work.index = epw_work.index.tz_localize(stz)

        epw_utc = epw_work.tz_convert("UTC")
        solar_utc = solar_time.dt.tz_convert("UTC")
        epw_utc = epw_utc[~epw_utc.index.duplicated(keep="first")].sort_index()

        idx_name = epw_utc.index.name if epw_utc.index.name is not None else "index"
        right = epw_utc.reset_index().rename(columns={idx_name: "ts"})
        right_sorted = right.sort_values("ts")

        left = pd.DataFrame({"solar_time": solar_utc})
        left["_orig_order"] = np.arange(len(left))
        left_sorted = left.sort_values("solar_time")

        out = pd.merge_asof(
            left_sorted,
            right_sorted[["ts", col]],
            left_on="solar_time",
            right_on="ts",
            direction="nearest",
            tolerance=pd.Timedelta(max_gap),
        )

        out = out.sort_values("_orig_order").reset_index(drop=True)
        return pd.Series(out[col].to_numpy(), index=solar_time.index, name=col)


    # 4) Label formatter: “21 SEP 11:00 19.40°C” (solar time)
    def _fmt_label(ts_solar: pd.Timestamp, temp_c: float) -> str:
        return f"{ts_solar.strftime('%d %b').upper()} {ts_solar.strftime('%H:%M')} {temp_c:.2f}°C"


    def sunpath_plotly_2d(
        site: Site,
        date: pd.Timestamp,
        projection: str,
        month_range: tuple[int, int] = (1, 12),
        show_envelope: bool = True,
        show_analemmas: bool = True,
        hours_for_analemma=range(6, 19),
        analemma_step_days: int = 7,
        epw: pd.DataFrame | None = None,
        color_var: str = "temperature",
        show_labels: bool = True,
        label_every: int = 1,
        marker_every: int = 1,
    ) -> go.Figure:
        from datetime import timedelta

        # Helper: compute sunpath samples for a day (only sun-above-horizon)
        def sunpath_for_day(day, freq="10min"):
            idx = pd.date_range(day, day + timedelta(days=1), freq=freq, tz=site.tz)[:-1]
            sp = pvlib.solarposition.get_solarposition(idx, site.lat, site.lon, altitude=site.elev_m)
            sp = sp[sp["elevation"] > 0]
            if sp.empty:
                return np.array([]), np.array([]), np.array([]), np.array([]), idx
            az = sp["azimuth"].to_numpy()
            el = sp["elevation"].to_numpy()
            xs, ys = project_angular(az, el, projection)
            return xs, ys, az, el, sp.index

        # Utility: unit vector for an azimuth angle (deg) in plot coords
        def az_unit(az_deg):
            a = np.deg2rad(az_deg)
            # North-up: az=0 points to +Y; East to +X
            return np.sin(a), np.cos(a)

        fig = go.Figure()

        # Build shapes: rim + altitude circles + azimuth ticks + hour radial lines
        shapes = []

        # Rim
        shapes.append(dict(type="circle", xref="x", yref="y", x0=-1, y0=-1, x1=1, y1=1, line=dict(width=1.2, color="rgba(255,255,255,0.4)"), fillcolor="rgba(30, 41, 59, 0.4)"))

        # Altitude concentric rings (10..80)
        for alt in range(10, 90, 10):
            x0, y0 = project_angular(np.array([0.0]), np.array([alt]), projection)
            r = float(np.hypot(x0[0], y0[0]))
            shapes.append(dict(type="circle", xref="x", yref="y", x0=-r, y0=-r, x1=r, y1=r, line=dict(width=0.7, color="rgba(255,255,255,0.25)", dash="dot")))

        # Azimuth ticks (every 10°; major every 90°)
        for az in range(0, 360, 10):
            ux, uy = az_unit(az)
            inner = 0.98
            outer = 1.03 if az % 90 == 0 else 1.01
            lw = 2 if az % 90 == 0 else 1
            shapes.append(dict(type="line", x0=ux * inner, y0=uy * inner, x1=ux * outer, y1=uy * outer, line=dict(width=lw, color="rgba(255,255,255,0.5)")))

        # Hour radial lines (6..18), light dash
        for h in range(6, 19):
            az = (h - 12) * 15 + 180
            ux, uy = az_unit(az)
            shapes.append(dict(type="line", xref="x", yref="y", x0=0, y0=0, x1=ux * 1.0, y1=uy * 1.0, line=dict(width=1, color="rgba(255,255,255,0.25)", dash="dot")))

        fig.update_layout(shapes=shapes)

        # Cardinal labels (N,E,S,W) at perimeter
        cardinals = [("N", 0), ("E", 90), ("S", 180), ("W", 270)]
        fig.add_trace(go.Scatter(
            x=[az_unit(a)[0] * 1.12 for _, a in cardinals],
            y=[az_unit(a)[1] * 1.12 for _, a in cardinals],
            text=[t for t, _ in cardinals], mode="text", showlegend=False, hoverinfo="skip",
            textfont=dict(size=16, color="#e5e7eb", family="Arial Black")
        ))

        # Altitude ring labels (placed toward southern rim for readability)
        for alt in range(10, 90, 10):
            x0, y0 = project_angular(np.array([0.0]), np.array([alt]), projection)
            r = float(np.hypot(x0[0], y0[0]))
            fig.add_trace(go.Scatter(x=[0.0], y=[-r], text=[f"{alt}°"], mode="text", showlegend=False, hoverinfo="skip",
                                     textfont=dict(size=11, color="#94a3b8")))

        # Perimeter azimuth degree labels (every 30°)
        az_lab = [a for a in range(0, 360, 30) if a % 90 != 0]
        az_x = [az_unit(a)[0] * 1.05 for a in az_lab]
        az_y = [az_unit(a)[1] * 1.05 for a in az_lab]
        az_text = [f"{a}°" for a in az_lab]
        fig.add_trace(go.Scatter(x=az_x, y=az_y, text=az_text, mode="text", showlegend=False, hoverinfo="skip",
                     textfont=dict(size=9, color="#d1d5db")))

        # Hour labels placed inside the rim (6..18)
        hr_x = []
        hr_y = []
        hr_text = []
        for h in range(6, 19):
            az = (h - 12) * 15 + 180
            ux, uy = az_unit(az)
            hr_x.append(ux * 0.94)
            hr_y.append(uy * 0.94)
            hr_text.append(str(h))
            fig.add_trace(go.Scatter(x=hr_x, y=hr_y, text=hr_text, mode="text", showlegend=False, hoverinfo="skip",
                                     textfont=dict(size=11, color="#fbbf24")))

        # Pre-determine color properties for overall plot
        color_cols = ["temp_air","DryBulb","Dry_Bulb","drybulb","Temperature"]
        color_title = "Dry Bulb (°C)"
        colorscale = "RdYlBu_r"
        cmin = None
        cmax = None
        colorbar_added = False

        if epw is not None:
            if color_var == "temperature":
                color_cols = ["temp_air","DryBulb","Dry_Bulb","drybulb","Temperature"]
                color_title = "Dry Bulb (°C)"
            elif color_var == "solar":
                color_cols = ["glohorrad","ghi","global_horiz","global_horizontal","solar","radiation"]
                color_title = "Solar Radiation (W/m²)"
            elif color_var == "humidity":
                color_cols = ["relhum","relative_humidity","rh"]
                color_title = "Relative Humidity (%)"
            elif color_var == "wind":
                color_cols = ["windspd","wind_speed","wspd","wind"]
                color_title = "Wind Speed (m/s)"
            
            found_col = next((c for c in color_cols if c in epw.columns), None)
            if found_col is not None:
                epw_vals = pd.to_numeric(epw[found_col], errors="coerce").dropna()
                if not epw_vals.empty:
                    low, high = np.percentile(epw_vals, [1, 99])
                    if low == high:
                        high = low + 1e-6
                    cmin, cmax = low, high

        def get_colors(df_in):
            """Look up EPW color values by matching month-day-hour.
            This approach is immune to year mismatches between TMY EPW and current-year sun positions."""
            if epw is None or not color_cols:
                return None
            target_col = next((c for c in color_cols if c in epw.columns), None)
            if target_col is None:
                return None
            try:
                # Convert both to the site timezone for consistent month/day/hour
                epw_local = epw.copy()
                if epw_local.index.tz is None:
                    epw_local.index = epw_local.index.tz_localize("UTC")
                epw_local = epw_local.tz_convert(site.tz)
                
                idx = df_in.index
                if idx.tz is None:
                    idx = idx.tz_localize(site.tz)
                else:
                    idx = idx.tz_convert(site.tz)
                
                # Build a lookup dict: (month, day, hour) -> value
                epw_vals = pd.to_numeric(epw_local[target_col], errors="coerce")
                lookup = {}
                for ts, val in zip(epw_local.index, epw_vals):
                    key = (ts.month, ts.day, ts.hour)
                    lookup[key] = val
                
                # Match each sun position to EPW by month-day-hour
                result = np.full(len(idx), np.nan)
                for i, ts in enumerate(idx):
                    key = (ts.month, ts.day, ts.hour)
                    if key in lookup:
                        result[i] = lookup[key]
                    else:
                        # Fallback: try same month/hour, any day (nearest day)
                        best_val = np.nan
                        best_dist = 999
                        for (m, d, h), v in lookup.items():
                            if m == ts.month and h == ts.hour:
                                dist = abs(d - ts.day)
                                if dist < best_dist:
                                    best_dist = dist
                                    best_val = v
                        result[i] = best_val
                
                return result
            except Exception:
                return None

        if show_analemmas:
            # Generate analemma (figure 8) points by taking 1 sample per week at each hour
            year = date.year
            analemma_dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq=f"{analemma_step_days}D", tz=site.tz)
            analemma_dates = analemma_dates[(analemma_dates.month >= month_range[0]) & (analemma_dates.month <= month_range[1])]
            
            an_sp_all = pvlib.solarposition.get_solarposition(analemma_dates, site.lat, site.lon, altitude=site.elev_m)
            
            # Since an analemma links the same clock hour across days, we need full days of hours
            # To be efficient, we compute exactly the requested hours for all analemma_dates
            idx_list = []
            for d in analemma_dates:
                for h in hours_for_analemma:
                    idx_list.append(d.replace(hour=h, minute=0, second=0))
            idx_analemma = pd.DatetimeIndex(idx_list).tz_localize(None).tz_localize(site.tz) if idx_list[0].tz is None else pd.DatetimeIndex(idx_list)
            sp_an = pvlib.solarposition.get_solarposition(idx_analemma, site.lat, site.lon, altitude=site.elev_m)
            sp_an = sp_an[sp_an["elevation"] > 0]
            
            # Draw an analemma line for each hour
            for h in hours_for_analemma:
                mask = sp_an.index.hour == h
                hr_data = sp_an[mask].sort_index()
                if hr_data.empty: continue
                # Close the loop
                hr_data = pd.concat([hr_data, hr_data.iloc[[0]]])
                hx, hy = project_angular(hr_data["azimuth"].values, hr_data["elevation"].values, projection)
                fig.add_trace(go.Scatter(
                    x=hx, y=hy, mode="lines", 
                    line=dict(color="rgba(100,116,139,0.3)", width=1.5), 
                    showlegend=False, hoverinfo="skip"
                ))
                
                c_vals_an = get_colors(hr_data)
                show_cb_an = (c_vals_an is not None) and not colorbar_added
                if show_cb_an: colorbar_added = True
                
                fig.add_trace(go.Scatter(
                    x=hx, y=hy, mode="markers",
                    marker=dict(
                        size=3.5,
                        color=c_vals_an if c_vals_an is not None else "#2962ff",
                        colorscale=colorscale, cmin=cmin, cmax=cmax,
                        showscale=show_cb_an,
                        colorbar=(dict(title=color_title or "Value", thickness=10, len=0.5, x=1.03) if show_cb_an else None),
                        opacity=0.85
                    ),
                    showlegend=False, hoverinfo="skip"
                ))

        # ===== Solstice & Equinox arcs (3 key days) =====
        key_days = [
            (6, 21, "Summer Solstice", "#FF6B3D"),
            (12, 21, "Winter Solstice", "#3498DB"),
            (3, 21, "Equinox", "#2ECC71"),
        ]
        year = date.year
        for km, kd, klabel, kcolor in key_days:
            kts = pd.Timestamp(year, km, kd, tz=site.tz)
            kxs, kys, kaz, kel, kidx = sunpath_for_day(kts)
            if len(kxs) == 0:
                continue
            # Arc line
            fig.add_trace(go.Scatter(
                x=kxs, y=kys, mode="lines",
                line=dict(color=kcolor, width=2.5),
                name=klabel, showlegend=True, hoverinfo="skip"
            ))
            # Colored markers on this arc
            kdf = solar_positions(site, kts)
            if not kdf.empty:
                kcolors = get_colors(kdf)
                kx_m, ky_m = project_angular(kdf["azimuth"].values, kdf["elevation"].values, projection)
                show_cb_k = (kcolors is not None) and not colorbar_added
                if show_cb_k: colorbar_added = True
                fig.add_trace(go.Scatter(
                    x=kx_m, y=ky_m, mode="markers",
                    marker=dict(
                        size=6,
                        color=kcolors if kcolors is not None else kcolor,
                        colorscale=colorscale, cmin=cmin, cmax=cmax,
                        showscale=show_cb_k,
                        colorbar=(dict(title=color_title or "Value", thickness=10, len=0.5, x=1.03) if show_cb_k else None),
                        line=dict(width=0.5, color=kcolor),
                        opacity=0.9
                    ),
                    showlegend=False, hoverinfo="skip"
                ))
            # Label at the apex of the arc
            apex_idx = np.argmax(kel)
            fig.add_annotation(
                x=kxs[apex_idx], y=kys[apex_idx],
                text=klabel, showarrow=True, arrowhead=0, arrowcolor=kcolor,
                font=dict(color=kcolor, size=10, family="Arial"),
                bgcolor="rgba(13,17,23,0.7)", borderpad=2,
                ax=0, ay=-20
            )

        # Selected date: compute positions and set up data-driven colors
        xs_sel, ys_sel, az_sel, el_sel, idx_sel = sunpath_for_day(date)
        df_sel = solar_positions(site, date)
        if not df_sel.empty:
            df_sel = _with_solar_time(df_sel, site)
            df_sel = df_sel.iloc[::max(1, int(1))]  # default 10min -> hourly (already hourly from solar_positions)
            
            colors = get_colors(df_sel)

            xs_hr, ys_hr = project_angular(df_sel["azimuth"].values, df_sel["elevation"].values, projection)
            
            # Plot the selected day path (line behind points)
            fig.add_trace(go.Scatter(x=xs_hr, y=ys_hr, mode="lines", line=dict(color="rgba(255,255,255,0.4)", width=2, dash="dot"), name=f"{date:%b %d} Path", showlegend=False, hoverinfo="skip"))
            if len(xs_hr) > 0:
                idx_mid = len(xs_hr) // 2
                fig.add_annotation(x=xs_hr[idx_mid], y=ys_hr[idx_mid], text=f"{date:%b %d}", showarrow=False, font=dict(color="rgba(255,255,255,0.8)", size=10), yshift=12)
                
            show_cb_sel = (colors is not None) and not colorbar_added
            if show_cb_sel: colorbar_added = True
            
            fig.add_trace(go.Scatter(
                x=xs_hr, y=ys_hr,
                mode="markers+text" if show_labels else "markers",
                text=[t.strftime("%H:%M") for t in pd.to_datetime(df_sel["solar_time"]).dt.tz_convert(site.tz)] if show_labels else None,
                textposition="top center",
                textfont=dict(size=11, color="rgba(240,240,240,0.92)"),
                marker=dict(
                    size=10,
                    color=colors if colors is not None else "#fbbf24",
                    colorscale=colorscale, cmin=cmin, cmax=cmax,
                    showscale=show_cb_sel,
                    colorbar=(dict(title=color_title or "Value", thickness=12, len=0.55, y=0.5, x=1.03) if show_cb_sel else None),
                    line=dict(width=1, color="rgba(255,255,255,0.8)")
                ),
                name="Hourly markers",
                showlegend=False,
                hovertemplate="Time: %{text}<br>Az: %{customdata[0]:.1f}°<br>Alt: %{customdata[1]:.1f}°<extra></extra>",
                customdata=np.c_[df_sel["azimuth"], df_sel["elevation"]]
            ))

        # Current sun marker: pick nearest to 'now' in site tz (if within same day samples)
        now_local = pd.Timestamp.now(tz=site.tz)
        sun_marker_x = None
        sun_marker_y = None
        azimuth = altitude = None
        solar_time = "--"
        if xs_sel.size > 0:
            # build a full-day index (10min samples) anchored to the selected date's local midnight
            try:
                day_start = pd.Timestamp(date)
                if getattr(day_start, 'tzinfo', None) is None:
                    day_start = day_start.tz_localize(site.tz)
                else:
                    day_start = day_start.tz_convert(site.tz)
                day_start = day_start.normalize()
            except Exception:
                day_start = pd.Timestamp(date).tz_localize(site.tz).normalize()

            full_idx = pd.date_range(day_start, day_start + timedelta(days=1), freq="10min", tz=site.tz)[:-1]

            # find nearest time in full-day samples
            diffs_full = np.abs((full_idx - now_local) / np.timedelta64(1, 's')).astype(float)
            now_idx_full = int(np.argmin(diffs_full))
            target_time = full_idx[now_idx_full]

            # If the exact target_time exists in the filtered sun-above-horizon index, use it
            if len(idx_sel) > 0 and target_time in idx_sel:
                now_pos = int(np.where(idx_sel == target_time)[0][0])
            elif len(idx_sel) > 0:
                # fallback: nearest within filtered (sun-above-horizon) samples
                diffs = np.abs((idx_sel - now_local) / np.timedelta64(1, 's')).astype(float)
                now_pos = int(np.argmin(diffs))
            else:
                now_pos = None

            if now_pos is not None:
                sun_marker_x = float(xs_sel[now_pos])
                sun_marker_y = float(ys_sel[now_pos])
                azimuth = float(az_sel[now_pos])
                altitude = float(el_sel[now_pos])
                solar_time = idx_sel[now_pos].strftime("%H:%M")
                fig.add_trace(go.Scatter(x=[sun_marker_x], y=[sun_marker_y], mode="markers+text", text=["Current Sun"], textposition="top center", textfont=dict(color="#ff4d4d", size=11), marker=dict(size=16, color="#ff4d4d", line=dict(width=3, color="rgba(255,255,255,0.7)")), name="Current Sun", showlegend=False,
                                         hovertemplate=f"Time: {solar_time}<br>Az: {azimuth:.1f}°<br>Alt: {altitude:.1f}°<extra></extra>"))
                # radial line
                fig.add_shape(type="line", x0=0, y0=0, x1=sun_marker_x, y1=sun_marker_y, line=dict(color="#ff4d4d", width=2))

        info_dict = {
            "solar_time": solar_time,
            "azimuth": azimuth,
            "altitude": altitude,
            "sunrise": None,
            "sunset": None,
            "civil": None,
            "nautical": None,
            "astronomical": None,
            "lat": site.lat,
            "lon": site.lon
        }

        # compute rise/set/twilight from full-day samples if available
        if len(idx_sel) > 0:
            sp_full = pvlib.solarposition.get_solarposition(idx_sel, site.lat, site.lon, altitude=site.elev_m)
            positives = sp_full[sp_full["elevation"] > 0]
            if not positives.empty:
                info_dict["sunrise"] = positives.index[0].strftime("%H:%M")
                info_dict["sunset"] = positives.index[-1].strftime("%H:%M")

            def tw(elev):
                mask = sp_full[sp_full["elevation"] > elev]
                if mask.empty:
                    return "--","--"
                return mask.index[0].strftime("%H:%M"), mask.index[-1].strftime("%H:%M")

            civil = tw(-6); naut = tw(-12); astro = tw(-18)
            info_dict["civil"] = f"{civil[0]}–{civil[1]}"
            info_dict["nautical"] = f"{naut[0]}–{naut[1]}"
            info_dict["astronomical"] = f"{astro[0]}–{astro[1]}"

        fig.update_layout(
            xaxis=dict(scaleanchor="y", range=[-1.12, 1.12], visible=False),
            yaxis=dict(range=[-1.12, 1.12], visible=False),
            margin=dict(l=20, r=20, t=10, b=20),
            height=700,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                bgcolor="rgba(0,0,0,0.55)",
                bordercolor="#444",
                borderwidth=1,
                font=dict(color="#e5e7eb")
            ),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e5e7eb"),
            uirevision="sunpath_2d_arch"
        )

        return fig, info_dict



    def sunpath_plotly_3d(
        site: Site,
        date: pd.Timestamp,
        month_range: tuple[int, int] = (1, 12),
        show_rays: bool = True,
        massing=None,
        epw: pd.DataFrame | None = None,
        color_var: str = "temperature",  # "temperature" | "solar" | "humidity" | "wind"
        hour_stride: int = 1,
        show_labels: bool = True,
        label_every: int = 1,
        marker_every: int = 1,
        camera_eye: Optional[Dict[str, float]] = None,
    ):
        # Full celestial sphere + sun paths for seasonal dates and the selected date.
        fig = go.Figure()

        # --- Ground disk / Compass Base ---
        # Generate a faint flat Mesh3d disk for the compass base (prevents funneling artifacts)
        base_theta = np.linspace(0, 2 * np.pi, 60, endpoint=False)
        base_x = 1.05 * np.sin(base_theta)
        base_y = 1.05 * np.cos(base_theta)
        base_x = np.append(base_x, 0) # Center point
        base_y = np.append(base_y, 0)
        base_z = np.zeros_like(base_x) - 0.005 # slightly below 0
        
        i_idx, j_idx, k_idx = [], [], []
        center_idx = len(base_x) - 1
        for i in range(len(base_theta)):
            i_idx.append(i)
            j_idx.append((i + 1) % len(base_theta))
            k_idx.append(center_idx)

        fig.add_trace(go.Mesh3d(
            x=base_x, y=base_y, z=base_z,
            i=i_idx, j=j_idx, k=k_idx,
            color="rgba(255, 255, 255, 0.5)",
            lighting=dict(ambient=1, diffuse=0, specular=0),
            hoverinfo="skip",
            showlegend=False,
        ))

        # Helper geometry: horizon ring, altitude rings, azimuth spokes, cardinal labels
        def _ring_points(alt_deg: float, num=180):
            az = np.linspace(0, 360, num, endpoint=False)
            alt = np.full_like(az, alt_deg)
            az_r = np.deg2rad(az)
            alt_r = np.deg2rad(alt)
            x = np.cos(alt_r) * np.sin(az_r)
            y = np.cos(alt_r) * np.cos(az_r)
            z = np.sin(alt_r)
            return x, y, z

        # Horizon ring (thick wireframe base)
        hx, hy, hz = _ring_points(0)
        fig.add_trace(go.Scatter3d(
            x=hx, y=hy, z=hz, mode="lines",
            line=dict(color="rgba(255, 255, 255, 0.35)", width=2),
            showlegend=False, hoverinfo="skip",
        ))
        
        # Compass Tick marks every 10 degrees on the horizon
        tick_x, tick_y, tick_z = [], [], []
        for az_deg in range(0, 360, 10):
            if az_deg % 90 == 0: continue # Skip cardinals
            az_r = np.deg2rad(az_deg)
            tick_x.extend([1.0 * np.sin(az_r), 1.03 * np.sin(az_r), None])
            tick_y.extend([1.0 * np.cos(az_r), 1.03 * np.cos(az_r), None])
            tick_z.extend([0, 0, None])
            
        fig.add_trace(go.Scatter3d(
            x=tick_x, y=tick_y, z=tick_z, mode="lines",
            line=dict(color="rgba(255, 255, 255, 0.2)", width=1.5),
            showlegend=False, hoverinfo="skip",
        ))
        
        # Altitude rings every 15° with labels placed at az=180° (south)
        for alt_deg in range(15, 90, 15):
            rx, ry, rz = _ring_points(alt_deg)
            fig.add_trace(go.Scatter3d(
                x=rx, y=ry, z=rz, mode="lines",
                line=dict(color="rgba(255, 255, 255, 0.08)", width=1),
                showlegend=False, hoverinfo="skip",
            ))
            alt_r = np.deg2rad(alt_deg)
            az_r = np.deg2rad(180)
            fig.add_trace(go.Scatter3d(
                x=[1.02 * np.cos(alt_r) * np.sin(az_r)], y=[1.02 * np.cos(alt_r) * np.cos(az_r)], z=[np.sin(alt_r)],
                mode="text", text=[f"{alt_deg}°"], textposition="middle left",
                textfont=dict(size=10, color="rgba(255, 255, 255, 0.5)"),
                showlegend=False, hoverinfo="skip",
            ))

        # Azimuth meridians (Sky Vault wireframe arcs)
        for az_deg in range(0, 360, 30):
            az_r = np.deg2rad(az_deg)
            alts = np.linspace(0, np.pi/2, 40)
            x = np.cos(alts) * np.sin(az_r)
            y = np.cos(alts) * np.cos(az_r)
            z = np.sin(alts)
            is_cardinal = az_deg % 90 == 0
            
            # Ground spoke
            fig.add_trace(go.Scatter3d(
                x=[0, x[0]], y=[0, y[0]], z=[0, 0], mode="lines",
                line=dict(color="rgba(255,255,255,0.15)" if is_cardinal else "rgba(255,255,255,0.06)", 
                          width=1.5 if is_cardinal else 1),
                showlegend=False, hoverinfo="skip",
            ))
            # Sky Vault arc
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z, mode="lines",
                line=dict(color="rgba(255,255,255,0.12)" if is_cardinal else "rgba(255,255,255,0.05)", 
                          width=1.5 if is_cardinal else 1),
                showlegend=False, hoverinfo="skip",
            ))

        # Cardinal labels on horizon
        cardinals = {
            "N": (0, 1.12, 0), "E": (1.12, 0, 0), "S": (0, -1.12, 0), "W": (-1.12, 0, 0)
        }
        fig.add_trace(go.Scatter3d(
            x=[v[0] for v in cardinals.values()],
            y=[v[1] for v in cardinals.values()],
            z=[v[2] for v in cardinals.values()],
            mode="text", text=list(cardinals.keys()), textposition="middle center",
            textfont=dict(color="#facc15", size=20, family="Arial Black"),
            showlegend=False, hoverinfo="skip",
        ))

        # Pre-determine color properties for overall plot
        color_cols = ["temp_air","DryBulb","Dry_Bulb","drybulb","Temperature"]
        color_title = "Dry Bulb (°C)"
        colorscale = "RdYlBu_r"
        cmin = None
        cmax = None
        colorbar_added = False

        if epw is not None:
            if color_var == "temperature":
                color_cols = ["temp_air","DryBulb","Dry_Bulb","drybulb","Temperature"]
                color_title = "Dry Bulb (°C)"
            elif color_var == "solar":
                color_cols = ["glohorrad","ghi","global_horiz","global_horizontal","solar","radiation"]
                color_title = "Solar Radiation (W/m²)"
            elif color_var == "humidity":
                color_cols = ["relhum","relative_humidity","rh"]
                color_title = "Relative Humidity (%)"
            elif color_var == "wind":
                color_cols = ["windspd","wind_speed","wspd","wind"]
                color_title = "Wind Speed (m/s)"
            
            found_col = next((c for c in color_cols if c in epw.columns), None)
            if found_col is not None:
                epw_vals = pd.to_numeric(epw[found_col], errors="coerce").dropna()
                if not epw_vals.empty:
                    low, high = np.percentile(epw_vals, [1, 99])
                    if low == high:
                        high = low + 1e-6
                    cmin, cmax = low, high

        def get_colors(df_in):
            """Look up EPW color values by matching month-day-hour.
            Immune to year mismatches between TMY EPW and current-year sun positions."""
            if epw is None or not color_cols:
                return None
            target_col = next((c for c in color_cols if c in epw.columns), None)
            if target_col is None:
                return None
            try:
                epw_vals = pd.to_numeric(epw[target_col], errors="coerce")
                lookup = {}
                
                # Use columns instead of index timezones to avoid crashes on basic RangeIndex
                for m, d, h, val in zip(epw["month"], epw["day"], epw["hour"], epw_vals):
                    # pvlib uses 0-23 hours; EPW natively often uses 1-24. Adjust if needed.
                    h_adj = int(h) - 1
                    key = (int(m), int(d), h_adj)
                    lookup[key] = val
                
                idx = df_in.index
                result = np.full(len(idx), np.nan)
                
                # Check timezone and extract properties safe
                if hasattr(idx, "tz") and idx.tz is not None:
                    # Convert to target TZ
                    idx_local = idx.tz_convert(site.tz)
                elif hasattr(idx, "tz_localize"):
                    idx_local = idx.tz_localize(site.tz)
                else:
                    idx_local = idx
                
                # Assign values
                for i, ts in enumerate(idx_local):
                    try:
                        m, d, h = ts.month, ts.day, ts.hour
                        
                        # exact match
                        if (m, d, h) in lookup:
                            result[i] = lookup[(m, d, h)]
                            continue
                            
                        # nearest hour fallback (e.g. for pvlib shifts or analemmas)
                        best_val = np.nan
                        best_dist = 9999
                        for (lm, ld, lh), v in lookup.items():
                            if lh == h and lm == m:
                                dist = abs(ld - d)
                                if dist < best_dist:
                                    best_dist = dist
                                    best_val = v
                        
                        # Full fallback if still NaN (e.g. missing month data in EPW)
                        if pd.isna(best_val):
                            for (lm, ld, lh), v in lookup.items():
                                if lh == h:
                                    dist = abs(lm - m)*30 + abs(ld - d)
                                    if dist < best_dist:
                                        best_dist = dist
                                        best_val = v
                                        
                        result[i] = best_val
                    except AttributeError:
                        continue
                
                # If all NaNs, return None to trigger solid colors
                if np.isnan(result).all():
                    return None
                    
                return result
            except Exception:
                return None

        def _sunpath_for(ts: pd.Timestamp, color: str, label: str):
            nonlocal colorbar_added
            df = solar_positions(site, ts)
            if df.empty: return
            
            # Subsample for markers
            df_markers = df.iloc[::max(1, int(marker_every))]
            
            az, alt = np.deg2rad(df["azimuth"].values), np.deg2rad(df["elevation"].values)
            x, y, z = np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z, mode="lines",
                line=dict(width=1, color=color), name=label,
                hovertemplate="%{customdata[0]:.1f}° az<br>%{customdata[1]:.1f}° alt<extra></extra>",
                customdata=np.c_[df["azimuth"].values, df["elevation"].values],
            ))
            
            # Subsample for markers
            df_markers = df.iloc[::max(1, int(marker_every))]
            if not df_markers.empty:
                c_vals = get_colors(df_markers)
                m_az, m_alt = np.deg2rad(df_markers["azimuth"].values), np.deg2rad(df_markers["elevation"].values)
                m_x, m_y, m_z = np.cos(m_alt) * np.sin(m_az), np.cos(m_alt) * np.cos(m_az), np.sin(m_alt)
                
                show_cb = (c_vals is not None) and not colorbar_added
                if show_cb: colorbar_added = True
                
                fig.add_trace(go.Scatter3d(
                    x=m_x, y=m_y, z=m_z, mode="markers",
                    marker=dict(
                        size=5,
                        color=c_vals if c_vals is not None else "#fbbf24",
                        colorscale=colorscale, cmin=cmin, cmax=cmax,
                        showscale=show_cb,
                        colorbar=(dict(title=color_title or "Value", thickness=10, len=0.5, x=1.03) if show_cb else None),
                        opacity=0.85
                    ),
                    showlegend=False, hoverinfo="skip"
                ))

        # Volumetric Sky Vault Surface (Andrew Marsh style)
        year = date.year
        surf_dates = pd.date_range(start=f"{year}-12-21", end=f"{year+1}-06-21", periods=15, tz=site.tz)
        hours = np.linspace(5, 19, 30)
        X_surf, Y_surf, Z_surf = [], [], []
        for d in surf_dates:
            dt_list = [pd.Timestamp(f"{d.strftime('%Y-%m-%d')} {int(h):02d}:{int((h%1)*60):02d}:00", tz=site.tz) for h in hours]
            sp_surf = pvlib.solarposition.get_solarposition(pd.DatetimeIndex(dt_list), site.lat, site.lon, altitude=site.elev_m)
            az_surf = np.deg2rad(sp_surf["azimuth"].values)
            alt_surf = np.deg2rad(sp_surf["elevation"].values)
            alt_surf = np.clip(alt_surf, 0, None) # clip at horizon
            X_surf.append(np.cos(alt_surf) * np.sin(az_surf))
            Y_surf.append(np.cos(alt_surf) * np.cos(az_surf))
            Z_surf.append(np.sin(alt_surf))
        fig.add_trace(go.Surface(
            x=np.array(X_surf), y=np.array(Y_surf), z=np.array(Z_surf),
            colorscale=[[0, "rgba(100, 149, 237, 0.1)"], [1, "rgba(100, 149, 237, 0.1)"]],
            showscale=False, hoverinfo="skip"
        ))

        # seasonal arcs (solid red/orange Andrew Marsh style)
        season_days = [
            (12, 21, "rgba(230, 57, 70, 0.25)", "Winter Solstice"),
            (3, 21, "rgba(230, 57, 70, 0.25)", "Equinox"),
            (6, 21, "rgba(230, 57, 70, 0.25)", "Summer Solstice"),
        ]
        for m, d, col, label in season_days:
            ts = pd.Timestamp(year=date.year, month=m, day=d, tz=site.tz)
            _sunpath_for(ts, col, label)

        # Analemmas (Figure 8s)
        analemma_dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="7D", tz=site.tz)
        
        hours_for_analemma = range(6, 19)
        idx_list = []
        for d in analemma_dates:
            for h in hours_for_analemma:
                idx_list.append(d.replace(hour=h, minute=0, second=0))
        idx_analemma = pd.DatetimeIndex(idx_list).tz_localize(None).tz_localize(site.tz) if idx_list[0].tz is None else pd.DatetimeIndex(idx_list)
        sp_an = pvlib.solarposition.get_solarposition(idx_analemma, site.lat, site.lon, altitude=site.elev_m)
        sp_an = sp_an[sp_an["elevation"] > 0]
        
        for h in hours_for_analemma:
            mask = sp_an.index.hour == h
            hr_data = sp_an[mask].sort_index()
            if hr_data.empty: continue
            hr_data = pd.concat([hr_data, hr_data.iloc[[0]]])
            
            az_an = np.deg2rad(hr_data["azimuth"].values)
            alt_an = np.deg2rad(hr_data["elevation"].values)
            x_an = np.cos(alt_an) * np.sin(az_an)
            y_an = np.cos(alt_an) * np.cos(az_an)
            z_an = np.sin(alt_an)
            
            fig.add_trace(go.Scatter3d(
                x=x_an, y=y_an, z=z_an, mode="lines", 
                line=dict(color="rgba(41, 98, 255, 0.2)", width=1), 
                showlegend=False, hoverinfo="skip"
            ))

            c_vals_an = get_colors(hr_data)
            show_cb_an = (c_vals_an is not None) and not colorbar_added
            if show_cb_an: colorbar_added = True
            
            fig.add_trace(go.Scatter3d(
                x=x_an, y=y_an, z=z_an, mode="markers",
                marker=dict(
                    size=5,
                    color=c_vals_an if c_vals_an is not None else "#2962ff",
                    colorscale=colorscale, cmin=cmin, cmax=cmax,
                    showscale=show_cb_an,
                    colorbar=(dict(title=color_title or "Value", thickness=10, len=0.5, x=1.03) if show_cb_an else None),
                    opacity=0.85
                ),
                showlegend=False, hoverinfo="skip"
            ))

        # selected day: hourly positions (solar time labels + data-driven colors)
        df = solar_positions(site, date)
        if not df.empty:
            df = _with_solar_time(df, site)
            # subsample per requested hour stride (keep first row)
            df = df.iloc[::max(1, int(hour_stride))]
            # keep marker cadence by hour modulo marker_every
            df = df[df.index.hour % max(1, int(marker_every)) == 0]

            colors = get_colors(df)
            unit = "" if color_title is None else color_title.split("(")[-1].replace(")", "")

            az, alt = np.deg2rad(df["azimuth"].values), np.deg2rad(df["elevation"].values)
            sx = np.cos(alt) * np.sin(az)
            sy = np.cos(alt) * np.cos(az)
            sz = np.sin(alt)

            path_customdata = np.c_[df["azimuth"].values, df["elevation"].values]
            local_times = pd.to_datetime(df["solar_time"])
            if local_times.dt.tz is None:
                local_times = local_times.dt.tz_localize(site.tz)
            else:
                local_times = local_times.dt.tz_convert(site.tz)
            date_strs = local_times.dt.strftime("%Y-%m-%d").to_numpy()
            time_strs = local_times.dt.strftime("%H:%M").to_numpy()
            value_labels = []
            if colors is not None:
                for v in colors:
                    if pd.isna(v):
                        value_labels.append("n/a")
                    else:
                        suffix = unit.strip()
                        value_labels.append(f"{v:.1f} {suffix}" if suffix else f"{v:.1f}")
            else:
                value_labels = ["n/a"] * len(df)

            fig.add_trace(go.Scatter3d(
                x=sx, y=sy, z=sz,
                mode="lines",
                line=dict(width=3, color="#e63946"),
                name="Selected day",
                hoverinfo="skip",
                customdata=path_customdata,
                showlegend=True,
            ))
            
            # Sun Vector Arrow (pointing to noon or first point)
            if len(df) > 0:
                noon_idx = len(df)//2
                vx, vy, vz = sx[noon_idx], sy[noon_idx], sz[noon_idx]
                fig.add_trace(go.Scatter3d(
                    x=[0, vx], y=[0, vy], z=[0, vz], mode="lines",
                    line=dict(color="#2962ff", width=3.5), showlegend=False, hoverinfo="skip"
                ))
                fig.add_trace(go.Cone(
                    x=[vx], y=[vy], z=[vz], u=[vx], v=[vy], w=[vz],
                    sizemode="absolute", sizeref=0.08, anchor="tip",
                    colorscale=[[0, "#2962ff"], [1, "#2962ff"]], showscale=False, hoverinfo="skip"
                ))

            every = max(1, int(label_every))
            text_labels = []
            if show_labels:
                for i, t in enumerate(time_strs):
                    text_labels.append(t if i % every == 0 else "")
            else:
                text_labels = [""] * len(time_strs)

            show_cb_sel = (colors is not None) and not colorbar_added
            if show_cb_sel: colorbar_added = True
            
            fig.add_trace(go.Scatter3d(
                x=sx, y=sy, z=sz,
                mode="markers+text" if show_labels else "markers",
                text=text_labels,
                textposition="top center",
                textfont=dict(size=11, color="rgba(255, 255, 255, 0.95)", family="Arial Black"),
                marker=dict(
                    size=5,
                    color=colors if colors is not None else "#fbbf24",
                    colorscale=colorscale, cmin=cmin, cmax=cmax,
                    showscale=show_cb_sel,
                    colorbar=(dict(title=color_title or "Value", thickness=10, len=0.5, x=1.03) if show_cb_sel else None),
                    symbol="circle",
                    opacity=0.9,
                ),
                name="Hour markers",
                hovertemplate="Local %{customdata[1]}<br>Az %{customdata[2]:.1f}° · Alt %{customdata[3]:.1f}°<br>%{customdata[4]}<extra></extra>",
                customdata=np.c_[date_strs, time_strs, df["azimuth"].values, df["elevation"].values, value_labels],
                showlegend=False,
            ))

            if show_rays:
                # Rays suppressed to reduce clutter
                pass

        fig.update_layout(
            height=750,
            margin=dict(l=0, r=0, t=0, b=0),
            title_text="",
            legend_title_text="",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(13,17,23,0.85)",
                bordercolor="#2a2f3a",
                borderwidth=1,
                font=dict(color="#e5e7eb", size=12)
            ),
            scene=dict(
                xaxis=dict(range=[-1.2, 1.2], visible=True, showbackground=False, showgrid=True, gridcolor="#475569", zeroline=True, zerolinecolor="#94a3b8", title="", tickfont=dict(color="#94a3b8")),
                yaxis=dict(range=[-1.2, 1.2], visible=True, showbackground=False, showgrid=True, gridcolor="#475569", zeroline=True, zerolinecolor="#94a3b8", title="", tickfont=dict(color="#94a3b8")),
                zaxis=dict(range=[0.0, 1.1], visible=False),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=0.5),
                bgcolor="#0d1117",
            ),
            scene_camera=dict(
                eye=camera_eye if camera_eye is not None else dict(x=1.2, y=1.8, z=0.8),
                up=dict(x=0, y=0, z=1)
            ),
            plot_bgcolor="#0d1117",
            paper_bgcolor="#0d1117",
            font=dict(color="#e5e7eb"),
            dragmode="turntable",
            scene_dragmode="turntable",
            uirevision="sunpath3d"
        )

        return fig


    def draw_elevation_circles(ax, projection: str, step=10):
        # Concentric alt circles: 10,20,...,80 degrees. 0 = outer rim, 90 = center.
        alts = np.arange(10, 90, step)
        for alt in alts:
            # For each circle, compute radius via projection at uniform azimuth
            az = np.array([0.0])
            x, y = project_angular(az, np.array([alt]), projection)
            r = np.hypot(x, y)[0]
            ax.add_artist(plt.Circle((0, 0), r, fill=False, linewidth=0.6))
            ax.text(0, -r, f"{alt}°", ha="center", va="top", fontsize=7)


    def draw_day_arc(ax, site: Site, date: pd.Timestamp, projection: str,
                    label_hours: bool = True, hour_step: int = 1, **line_kws):
        df = solar_positions(site, date)
        # Keep sun-above-horizon only
        df = df[df["elevation"] > 0]
        if df.empty:
            return

        # Azimuth/elevation arrays for the daily arc
        az = df["azimuth"].to_numpy()
        alt = df["elevation"].to_numpy()
        x, y = project_angular(az, alt, projection)
        ax.plot(x, y, **line_kws)

        if label_hours and len(df) > 0:
            for i, (xx, yy, ts) in enumerate(zip(x, y, df.index)):
                if i % hour_step == 0:
                    ax.text(xx, yy, f"{ts.hour}", fontsize=7, ha="center", va="center")


    def draw_annual_grid(ax, site: Site, projection: str, months=(1, 3, 6, 9, 12)):
        """Light grid of seasonal daily arcs for context."""
        # Use solstices & equinox + a couple extras for envelope feel
        dates = [pd.Timestamp(year=2025, month=m, day=21, tz=site.tz) for m in months]
        for d in dates:
            df = solar_positions(site, d)
            df = df[df["elevation"] > 0]
            if df.empty:
                continue
            x, y = project_angular(df["azimuth"].to_numpy(), df["elevation"].to_numpy(), projection)
            ax.plot(x, y, linestyle="--", linewidth=0.6)


    def draw_sunpath_2d(site: Site, date: pd.Timestamp, opt: Options2D):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect("equal")
        ax.axis("off")

        # Rim = horizon
        ax.add_artist(plt.Circle((0, 0), 1.0, fill=False, linewidth=1.2))

        draw_compass(ax, 1.0)
        draw_elevation_circles(ax, opt.projection, step=10)

        if opt.show_annual_grid:
            # denser envelope feel
            draw_annual_grid(ax, site, opt.projection, months=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))

        # Main daily path (equinox default shown here; pass any date you like)
        draw_day_arc(
            ax, site, date, opt.projection,
            label_hours=opt.show_hour_labels, hour_step=opt.hour_label_step, linewidth=2
        )

        title = f"{date.strftime('%b %d, %Y')}  |  {site.lat:.3f}, {site.lon:.3f}  |  {site.tz}"
        ax.set_title(title, fontsize=10, pad=10)
        return fig, ax


    # ---------- 3D plot (optional) ----------

    def sun_sphere_points(lat: float, lon: float, tz: str, daylist: Iterable[pd.Timestamp]) -> pd.DataFrame:
        rows = []
        for d in daylist:
            df = solar_positions(Site(lat, lon, tz), d)
            df = df[df["elevation"] > 0]
            if df.empty:
                continue
            # Convert (az, alt) to unit-direction vectors
            az = np.deg2rad(df["azimuth"].to_numpy())
            alt = np.deg2rad(df["elevation"].to_numpy())
            x = np.cos(alt) * np.sin(az)
            y = np.cos(alt) * np.cos(az)
            z = np.sin(alt)
            tmp = pd.DataFrame({"x": x, "y": y, "z": z}, index=df.index)
            tmp["day"] = d.strftime("%b %d")
            rows.append(tmp)
        return pd.concat(rows) if rows else pd.DataFrame(columns=["x", "y", "z", "day"])


    def draw_massing(ax3d, blocks: List[Tuple[float, float, float, float, float, float]]):
        for (x, y, w, d, z0, z1) in blocks:
            # 8 corners
            X = [x, x+w, x+w, x, x, x+w, x+w, x]
            Y = [y, y, y+d, y+d, y, y, y+d, y+d]
            Z = [z0, z0, z0, z0, z1, z1, z1, z1]
            # vertical edges
            for i in range(4):
                ax3d.plot([X[i], X[i+4]], [Y[i], Y[i+4]], [Z[i], Z[i+4]])

            # Main plot
            fig = go.Figure()
            # Add horizon and altitude circles
            fig.update_layout(shapes=horizon_shapes()+hour_lines())

            # Add azimuth degree marks around perimeter
            az_marks = np.arange(0, 360, 10)
            az_x = np.sin(np.deg2rad(az_marks))
            az_y = -np.cos(np.deg2rad(az_marks))
            az_text = [f"{az}°" if az % 90 != 0 else "" for az in az_marks]
            fig.add_trace(go.Scatter(x=az_x, y=az_y, text=az_text, mode="text", showlegend=False,
                                     textfont=dict(size=9, color="#444"), hoverinfo="skip"))

            # Add bold cardinal direction labels (N, E, S, W) at perimeter
            cardinals = [("N", 0), ("E", 90), ("S", 180), ("W", 270)]
            fig.add_trace(go.Scatter(x=[np.sin(np.deg2rad(a)) for _,a in cardinals],
                                     y=[-np.cos(np.deg2rad(a)) for _,a in cardinals],
                                     text=[t for t,_ in cardinals], mode="text", showlegend=False,
                                     textfont=dict(size=18, color="#222", family="Arial Black"), hoverinfo="skip"))

            # Add altitude labels
            for alt in range(10, 100, 10):
                x, y = project_angular(np.array([0.0]), np.array([alt]), projection)
                fig.add_trace(go.Scatter(x=[x[0]], y=[y[0]], text=[f"{alt}°"], mode="text", showlegend=False,
                                         textfont=dict(size=12, color="#888")))

            # Style hour lines: radiate from center, label at perimeter
            for h in range(6, 19):
                az = (h-12)*15 + 180
                x = np.sin(np.deg2rad(az))
                y = -np.cos(np.deg2rad(az))
                fig.add_shape(type="line", x0=0, y0=0, x1=x, y1=y,
                              line=dict(width=1, color="#bbb", dash="dot"))
                # Hour label at perimeter
                fig.add_trace(go.Scatter(x=[x*1.08], y=[y*1.08], text=[str(h)], mode="text", showlegend=False,
                                         textfont=dict(size=11, color="#E67E22"), hoverinfo="skip"))

            # Add hour labels (on the March path)
            march21 = pd.Timestamp(year=date.year, month=3, day=21, tz=site.tz)
            xs_m, ys_m, az_m, el_m, idx_m = sunpath_for_day(march21)
            hour_marks = [i for i in range(len(idx_m)) if idx_m[i].minute==0 and 6<=idx_m[i].hour<=18]
            hour_marks = [i for i in hour_marks if i < len(xs_m) and i < len(ys_m)]
            fig.add_trace(go.Scatter(x=[xs_m[i] for i in hour_marks], y=[ys_m[i] for i in hour_marks],
                                     text=[str(idx_m[i].hour) for i in hour_marks], mode="text", showlegend=False,
                                     textfont=dict(size=11, color="#E67E22")))
        days = [
            pd.Timestamp(year=date.year, month=12, day=21, tz=site.tz),
            pd.Timestamp(year=date.year, month=3, day=21, tz=site.tz),
            pd.Timestamp(year=date.year, month=6, day=21, tz=site.tz),
            pd.Timestamp(year=date.year, month=9, day=21, tz=site.tz),
        ]
        pts = sun_sphere_points(site.lat, site.lon, site.tz, days)
        if not pts.empty:
            for day in pts["day"].unique():
                ddf = pts[pts["day"] == day]
                ax.plot(ddf["x"], ddf["y"], ddf["z"], marker="o", markersize=2)

        # Rays from a reference point to sun positions for selected date
        df = solar_positions(site, date)
        df = df[df["elevation"] > 0]
        if opt3d.show_rays and not df.empty:
            az = np.deg2rad(df["azimuth"].to_numpy())
            alt = np.deg2rad(df["elevation"].to_numpy())
            sx = np.cos(alt) * np.sin(az)
            sy = np.cos(alt) * np.cos(az)
            sz = np.sin(alt)
            px, py, pz = rays_point
            for x, y, z in zip(sx, sy, sz):
                ax.plot([px, x], [py, y], [pz, z], linewidth=0.8)

        # Simple massing
        if opt3d.massing:
            draw_massing(ax, opt3d.massing)

        ax.set_xlabel("E")
        ax.set_ylabel("N")
        ax.set_zlabel("Up")
        ax.set_title(f"3D Sun Path • {date.strftime('%b %d, %Y')}  |  {site.lat:.3f},{site.lon:.3f}")
        ax.set_box_aspect([1, 1, 0.6])
        return fig, ax


    # ---------- CLI / Demo ----------

    def main():
        if "plotly_templates_ready" not in st.session_state:
            pio.templates["bevldark"] = buildaccessibleplotlytemplate("dark")
            pio.templates["bevllight"] = buildaccessibleplotlytemplate("light")
            pio.templates.default = "bevldark"
            px.defaults.template = "bevldark"
            st.session_state["plotly_templates_ready"] = True
        p = argparse.ArgumentParser()
        p.add_argument("--lat", type=float, required=True)
        p.add_argument("--lon", type=float, required=True)
        p.add_argument("--tz", type=str, required=True)
        p.add_argument("--date", type=str, default="2025-09-23")
        p.add_argument("--projection", type=str, default="stereographic", choices=["stereographic", "orthographic"])
        p.add_argument("--show3d", action="store_true")
        args = p.parse_args()

        site = Site(args.lat, args.lon, args.tz)
        date = pd.Timestamp(args.date, tz=site.tz)

        opt2d = Options2D(projection=args.projection, show_annual_grid=True, show_hour_labels=True)
        fig2d, _ = draw_sunpath_2d(site, date, opt2d)

        opt3d = Options3D(show_3d=args.show3d, show_rays=True,
                        massing=[(-0.2, -0.1, 0.2, 0.3, 0.0, 0.25)])
        if opt3d.show_3d:
            fig3d, _ = draw_sunpath_3d(site, date, opt3d, rays_point=(0.0, 0.0, 0.0))

        plt.show()

    # ===== SUN PATH (render above cloud coverage) =====
    from datetime import timezone, timedelta

    cdf = st.session_state.get("cdf")
    header = st.session_state.get("header")
    if cdf is None or header is None:
        st.error("No EPW data loaded.")
        return

    # Build timezone from EPW header offset, then a Site from the same header
    loc = header["location"]
    tz_hours = float(loc.get("timezone") or 0.0)            # e.g., -5 for Buffalo
    tzinfo = timezone(timedelta(hours=tz_hours))

    site = Site(
        lat=float(loc.get("latitude") or 0.0),
        lon=float(loc.get("longitude") or 0.0),
        tz=tzinfo,                                           # tzinfo is fine here
        elev_m=float(loc.get("elevation_m") or 0.0)
    )


    location_label = get_clean_city_name()
    st.markdown(f"<h3>{location_label} – Solar Analysis</h3>", unsafe_allow_html=True)
    st.caption(
        "Trace the sun’s path, check solar-time positions, and pair those tracks with EPW temperatures "
        "before taking shading or PV decisions. The plots highlight seasonal envelopes so you can "
        "see when the sun is high, low, or missing entirely."
    )

    # ===== SUN PATH (interactive) =====
    # Sun-path date input removed per UX request; use today's date in site timezone
    sel_date = pd.Timestamp.now(tz=tzinfo).date()

    # Prepare EPW-derived dataframe with relevant variables if present
    epw_df = pd.DataFrame(index=cdf.index)
    temp_col = get_metric_column(cdf, ["drybulb", "temp_air", "temperature", "tdb"])
    if temp_col:
        epw_df["temp_air"] = cdf[temp_col]
    solar_col = get_metric_column(cdf, ["glohorrad", "ghi", "global_horiz", "global_horizontal", "solar", "radiation"])
    if solar_col:
        epw_df["glohorrad"] = cdf[solar_col]
    rh_col = get_metric_column(cdf, ["relhum", "relative_humidity", "rh"])
    if rh_col:
        epw_df["relhum"] = cdf[rh_col]
    wind_col = get_metric_column(cdf, ["windspd", "wind_speed", "wspd", "wind"])
    if wind_col:
        epw_df["windspd"] = cdf[wind_col]
        
    # Explicitly add time components needed by the color lookups
    epw_df["month"] = cdf.index.month
    epw_df["day"] = cdf.index.day
    epw_df["hour"] = cdf.index.hour

    # Ensure tz-awareness consistent with site tz, then convert to UTC for clean joins
    if epw_df.index.tz is None:
        epw_df.index = epw_df.index.tz_localize(tzinfo)
    epw_df = epw_df.tz_convert("UTC").sort_index()
    epw_df = epw_df[~epw_df.index.duplicated(keep="first")]


    # Available options for coloring
    avail_options = []
    option_map = {}
    if "temp_air" in epw_df.columns:
        avail_options.append("Dry Bulb Temperature (°C)")
        option_map["Dry Bulb Temperature (°C)"] = "temperature"
    if "glohorrad" in epw_df.columns:
        avail_options.append("Solar Radiation (W/m²)")
        option_map["Solar Radiation (W/m²)"] = "solar"
    if "relhum" in epw_df.columns:
        avail_options.append("Relative Humidity (%)")
        option_map["Relative Humidity (%)"] = "humidity"
    if "windspd" in epw_df.columns:
        avail_options.append("Wind Speed (m/s)")
        option_map["Wind Speed (m/s)"] = "wind"
    if not avail_options:
        avail_options = ["Dry Bulb Temperature (°C)"]
        option_map["Dry Bulb Temperature (°C)"] = "temperature"

    with st.expander("⚙️ Sun Path Display Settings", expanded=True):
        sc1, sc2, sc3 = st.columns([1.2, 1.2, 1])
        proj = sc1.selectbox("2D projection", ["stereographic", "orthographic"], index=0)
        color_choice_label = sc2.selectbox("Color sun points by", options=avail_options, index=0)
        color_var = option_map.get(color_choice_label, "temperature")
        sc3.markdown("<br>", unsafe_allow_html=True)
        
        sc4, sc5, sc6 = st.columns([1.2, 1.2, 1])
        hour_stride = sc4.slider("Show sun every N hours", min_value=1, max_value=4, value=1, step=1)
        marker_every = sc5.slider("Show markers every N hours", min_value=1, max_value=4, value=1, step=1)
        label_every = marker_every
        sc6.markdown("<br>", unsafe_allow_html=True)
        show_labels = sc6.checkbox("Show labels on markers", value=True)
        
        st.markdown("**Filters**")
        month_range = st.slider("Month Range", 1, 12, (1, 12))
        
        st.markdown("**3D Camera View**")
        view_state_raw = st.radio("3D Camera View", ["Default", "Top", "South", "East"], horizontal=True, label_visibility="collapsed")
        view_mapping = {"Default": "default", "Top": "top", "South": "south", "East": "east"}
        view_state = view_mapping.get(view_state_raw, "default")

    sel_ts = pd.Timestamp(sel_date, tz=tzinfo)
    display_date = sel_ts.strftime("%b %d, %Y")
    
    st.markdown(f"#### Sun Path Diagram • {display_date}")
    display_date = sel_ts.strftime("%b %d, %Y")

    fig2d, info_dict = sunpath_plotly_2d(
        site, sel_ts, proj,
        month_range=month_range,
        epw=epw_df,
        color_var=color_var,
        show_labels=show_labels,
        label_every=label_every,
        marker_every=marker_every
    )
    cname = get_clean_city_name().replace(" ", "_")
    _st_plotly_chart(fig2d, use_container_width=True, config={"displayModeBar": True, "toImageButtonOptions": {"filename": f"{cname}_sunpath_2d", "format": "png", "scale": 2}})
    _add_manual_pdf_figure("Sun Path 2D", fig2d)
    # --- Render Solar Info Panel ---
    if info_dict.get("azimuth") is not None:
        st.markdown("##### Solar Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Azimuth & Altitude", f"{info_dict['azimuth']:.1f}° / {info_dict['altitude']:.1f}°", f"Solar Time: {info_dict['solar_time']}")
        m2.metric("Sunrise / Sunset", f"{info_dict.get('sunrise', '--')}", f"{info_dict.get('sunset', '--')}")
        m3.metric("Civil Twilight", f"{info_dict.get('civil', '--')}")
        m4.metric("Coordinates", f"{info_dict['lat']:.2f}, {info_dict['lon']:.2f}")
    else:
        st.info("No sun above horizon on this date at this location.")
    st.markdown("---")



    st.subheader(f"Sun Path (3D) — {display_date}")

    def _camera_eye_for(view: str) -> Dict[str, float]:
        base = {
            "default": dict(x=1.25, y=1.25, z=0.9),
            "top": dict(x=0.0, y=0.0, z=2.2),
            "south": dict(x=0.0, y=-2.0, z=1.2),
            "east": dict(x=2.0, y=0.0, z=1.2),
        }.get(view, dict(x=1.25, y=1.25, z=0.9))
        return base

    camera_eye = _camera_eye_for(view_state)

    fig3d = sunpath_plotly_3d(
        site, sel_ts,
        month_range=month_range,
        show_rays=True,
        massing=[(-0.2,-0.1,0.2,0.3,0.0,0.25)],
        epw=epw_df,
        color_var=color_var,
        hour_stride=hour_stride,
        show_labels=show_labels,
        label_every=label_every,
        marker_every=marker_every,
        camera_eye=camera_eye,
    )
    # Coerce common non-figure returns and guard against bad types before plotting.
    if isinstance(fig3d, (dict, list)):
        fig3d = go.Figure(fig3d)
    if isinstance(fig3d, go.Figure):
        _st_plotly_chart(
            fig3d,
            use_container_width=True,
            config={
                "scrollZoom": True,
                "displayModeBar": True,
                "displaylogo": False,
                "toImageButtonOptions": {"filename": f"{cname}_sunpath_3d", "format": "png", "scale": 2},
            },
        )
        _add_manual_pdf_figure("Sun Path 3D", fig3d)
        st.caption("Sun position colored by selected environmental variable.")
    else:
        st.warning(f"3D sun path unavailable (got {type(fig3d).__name__}).")

    # ======================== PLOT: CARTESIAN (PVSyst-style) ========================
    st.markdown("#### Cartesian sun path (PVSyst style)")

    # Link the Cartesian graph to the dynamically selected color variable
    color_mode = color_var # inherit from the 3D expandable settings above

    # ---- Controls (feel free to move to a UI row above) ----
    cA, cB, cC, cD = st.columns([1,1,1,1])
    show_colorbar   = cA.checkbox("Show colorbar", value=(color_mode=="temperature"))
    show_analemmas  = cB.checkbox("Show analemmas", value=True, help="Curves for fixed clock-hours across the year")
    horizon_deg     = cC.number_input("Horizon (°)", min_value=0, max_value=90, value=0, step=1)
    show_hour_labels= cD.checkbox("Hour labels (largest alt)", value=True)

    # Optional: "now" marker (in site tz)
    show_now = True
    now_local = pd.Timestamp.now(tz=site.tz)
    def solar_pos(time_index_utc):
        sp = pvlib.solarposition.get_solarposition(
            time_index_utc, site.lat, site.lon, altitude=site.elev_m
        )
        # pvlib azimuth is 0..360 from North, elevation provided
        return sp["azimuth"].to_numpy(), sp["elevation"].to_numpy(), sp

    color_title = "Value"
    colorscale = "RdYlBu_r"
    cmin = None
    cmax = None
    
    if epw_df is not None and not epw_df.empty and color_mode is not None:
        if color_mode == "temperature":
            target_col = next((c for c in ["temp_air","DryBulb","Dry_Bulb","drybulb","Temperature"] if c in epw_df.columns), None)
            color_title = "Dry Bulb (°C)"
        elif color_mode == "solar":
            target_col = next((c for c in ["glohorrad","ghi","global_horiz","global_horizontal","solar","radiation"] if c in epw_df.columns), None)
            color_title = "Solar Radiation (W/m²)"
        elif color_mode == "humidity":
            target_col = next((c for c in ["relhum","relative_humidity","rh"] if c in epw_df.columns), None)
            color_title = "Relative Humidity (%)"
        elif color_mode == "wind":
            target_col = next((c for c in ["windspd","wind_speed","wspd","wind"] if c in epw_df.columns), None)
            color_title = "Wind Speed (m/s)"
        else:
            target_col = None

        if target_col is not None:
            epw_vals = pd.to_numeric(epw_df[target_col], errors="coerce").dropna()
            if not epw_vals.empty:
                low, high = np.percentile(epw_vals, [1, 99])
                if low == high:
                    high = low + 1e-6
                cmin, cmax = low, high

    def _colors_for_index(idx_local):
        """Return EPW data aligned to idx_local using month-day-hour matching (year-proof)."""
        if epw_df is None or epw_df.empty:
            return np.full(len(idx_local), np.nan)
        
        # Determine the column to extract based on color_mode setting
        target_col = None
        if color_mode == "temperature":
            target_col = next((c for c in ["temp_air","DryBulb","Dry_Bulb","drybulb","Temperature"] if c in epw_df.columns), None)
        elif color_mode == "solar":
            target_col = next((c for c in ["glohorrad","ghi","global_horiz","global_horizontal","solar","radiation"] if c in epw_df.columns), None)
        elif color_mode == "humidity":
            target_col = next((c for c in ["relhum","relative_humidity","rh"] if c in epw_df.columns), None)
        elif color_mode == "wind":
            target_col = next((c for c in ["windspd","wind_speed","wspd","wind"] if c in epw_df.columns), None)
        
        if target_col is None:
            return np.full(len(idx_local), np.nan)
        
        # Convert EPW to site local time for month/day/hour matching
        epw = epw_df.copy()
        if epw.index.tz is None:
            epw.index = epw.index.tz_localize("UTC")
        epw_local = epw.tz_convert(site.tz)
        
        # Convert query index to site local time
        if idx_local.tz is None:
            idx_local = idx_local.tz_localize(site.tz)
        else:
            idx_local = idx_local.tz_convert(site.tz)
        
        # Build lookup dict: (month, day, hour) -> value
        epw_vals = pd.to_numeric(epw_local[target_col], errors="coerce")
        lookup = {}
        for ts, val in zip(epw_local.index, epw_vals):
            lookup[(ts.month, ts.day, ts.hour)] = val
        
        result = np.full(len(idx_local), np.nan)
        for i, ts in enumerate(idx_local):
            key = (ts.month, ts.day, ts.hour)
            if key in lookup:
                result[i] = lookup[key]
            else:
                best_val = np.nan
                best_dist = 999
                for (m, d, h), v in lookup.items():
                    if m == ts.month and h == ts.hour:
                        dist = abs(d - ts.day)
                        if dist < best_dist:
                            best_dist = dist
                            best_val = v
                result[i] = best_val
            
        return result

    def _season_color(month):
        return {12:"#2D7DD2",1:"#2D7DD2",2:"#2D7DD2",
                3:"#00B0F0",4:"#00B0F0",5:"#00B0F0",
                6:"#FF6B3D",7:"#FF6B3D",8:"#FF6B3D",
                9:"#FFA14A",10:"#FFA14A",11:"#FFA14A"}[month]

    # ---- Build traces ----
    month_traces_cart = []
    envelope_cart     = []
    analemma_cart     = []
    suns_traces_cart  = []

    year = sel_ts.year
    tz   = site.tz

    # (A) Monthly arcs for the 21st of each month (daily path; points hourly)
    for m in range(month_range[0], month_range[1] + 1):
        day = 21 if m != 2 else 20  # avoid Feb-31 style pitfalls
        ts_local = pd.date_range(pd.Timestamp(year, m, day, tz=tz), periods=24, freq="1h")
        az, alt, _ = solar_pos(ts_local.tz_convert("UTC"))
        mask = alt > 0
        if not mask.any():
            continue
        x = az[mask]; y = alt[mask]
        
        # Add a text label to the highest altitude point for this month's arc
        lbl_idx = np.argmax(y)
        month_lbls = [pd.Timestamp(year, m, day).strftime("%b")] if show_hour_labels else [""]
        text_array = [""] * len(x)
        text_array[lbl_idx] = pd.Timestamp(year, m, day).strftime("%b")

        if color_mode is not None:
            c_vals = _colors_for_index(ts_local[mask])
            tr = go.Scatter(
                x=x, y=y, mode="lines+markers+text",
                text=text_array, textposition="top center", textfont=dict(color="rgba(200,200,200,0.9)", size=10),
                marker=dict(size=4, color=c_vals, colorscale=colorscale,
                            cmin=cmin, cmax=cmax, showscale=False),
                line=dict(width=2, color="rgba(200,200,200,0.75)"),
                name=pd.Timestamp(year, m, day).strftime("%b 21"),
                showlegend=False
            )
        else:
            col = _season_color(m)
            tr = go.Scatter(
                x=x, y=y, mode="lines+markers+text",
                text=text_array, textposition="top center", textfont=dict(color=col, size=10),
                marker=dict(size=4, color=col),
                line=dict(width=2, color=col),
                name=pd.Timestamp(year, m, day).strftime("%b 21"),
                showlegend=False
            )
        month_traces_cart.append(tr)

    # (B) Solstice envelope (Jun 21 vs Dec 21)
    def _daily_arc(month, day):
        tloc = pd.date_range(pd.Timestamp(year, month, day, tz=tz), periods=24, freq="1h")
        az, alt, _ = solar_pos(tloc.tz_convert("UTC"))
        m = alt > 0
        return (az[m], alt[m]) if m.any() else (np.array([]), np.array([]))

    az_su, alt_su = _daily_arc(6, 21)
    az_wi, alt_wi = _daily_arc(12, 21)
    if len(az_su) and len(az_wi):
        envelope_cart.append(go.Scatter(x=az_su, y=alt_su, mode="lines",
                                        line=dict(width=2.6, color="#E67E22"),
                                        name="Summer solstice", showlegend=False))
        # Add label at noon
        envelope_cart.append(go.Scatter(x=[az_su[len(az_su)//2]], y=[alt_su[len(az_su)//2]], mode="text", text=["Summer Solstice"], textposition="top center", textfont=dict(color="#E67E22", size=10), hoverinfo="skip", showlegend=False))
        
        envelope_cart.append(go.Scatter(x=az_wi, y=alt_wi, mode="lines",
                                        line=dict(width=2.6, color="#3498DB"),
                                        name="Winter solstice", showlegend=False))
        envelope_cart.append(go.Scatter(x=[az_wi[len(az_wi)//2]], y=[alt_wi[len(az_wi)//2]], mode="text", text=["Winter Solstice"], textposition="top center", textfont=dict(color="#3498DB", size=10), hoverinfo="skip", showlegend=False))

    # (C) Analemmas (fixed clock hour over the year)
    if show_analemmas:
        hours = range(6, 19)  # 6..18
        # Filter days based on month slider
        days  = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="7D", tz=tz)
        days = days[(days.month >= month_range[0]) & (days.month <= month_range[1])]
        for hh in hours:
            idx = [pd.Timestamp(d.date(), tz=tz) + pd.Timedelta(hours=hh) for d in days]
            idx = pd.DatetimeIndex(idx)
            az, alt, _ = solar_pos(idx.tz_convert("UTC"))
            m = alt > 0
            if not m.any(): 
                continue
            
            # Label at highest altitude of analemma
            lbl_idx = np.argmax(alt[m])
            analemma_text = [""] * len(az[m])
            if hh in (6, 9, 12, 15, 18):
                analemma_text[lbl_idx] = f"{hh:02d}:00"

            analemma_cart.append(
                go.Scatter(
                    x=az[m], y=alt[m], mode="lines+text",
                    text=analemma_text, textposition="top center", textfont=dict(color="rgba(100,120,200,0.8)", size=9),
                    line=dict(width=1, dash="dot", color="rgba(60,80,160,0.55)"),
                    name=(f"{hh:02d}:00" if hh in (6, 12, 18) else None),
                    showlegend=False
                )
            )

    # (D) Selected day (suns) with optional temperature colors & labels
    idx_sel = pd.date_range(pd.Timestamp(sel_ts.date(), tz=tz), periods=24, freq="1h")
    az_sel, alt_sel, _ = solar_pos(idx_sel.tz_convert("UTC"))
    m_sel = alt_sel > 0
    x_sel = az_sel[m_sel]; y_sel = alt_sel[m_sel]
    labels = [f"{t.hour:02d}:00" for t in idx_sel[m_sel]]
    if color_mode is not None:
        c_vals_sel = _colors_for_index(idx_sel[m_sel])
        suns_traces_cart.append(
            go.Scatter(
                x=x_sel, y=y_sel, mode="lines+markers+text",
                text=labels if show_hour_labels else None,
                textposition="top center",
                marker=dict(size=6, color=c_vals_sel, colorscale=colorscale,
                            cmin=cmin, cmax=cmax, showscale=False, line=dict(width=0.5, color="#222")),
                line=dict(width=2, color="#CCCCCC"),
                name=f"{sel_ts:%b %d} (solar day)"
            )
        )
    else:
        suns_traces_cart.append(
            go.Scatter(
                x=x_sel, y=y_sel, mode="lines+markers+text",
                text=labels if show_hour_labels else None,
                textposition="top center",
                marker=dict(size=6, color="#222", line=dict(width=0.5, color="#fff")),
                line=dict(width=2, color="#888"),
                name=f"{sel_ts:%b %d} (solar day)"
            )
        )

    # ---- Assemble figure ----
    fig_cart = go.Figure()
    for tr in month_traces_cart: fig_cart.add_trace(tr)
    for tr in envelope_cart:     fig_cart.add_trace(tr)
    for tr in analemma_cart:     fig_cart.add_trace(tr)
    for tr in suns_traces_cart:  fig_cart.add_trace(tr)

    # Optional coloraxis with colorbar
    if color_mode is not None and show_colorbar:
        # Add a tiny hidden dummy trace to carry the colorbar
        fig_cart.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=0.1, color=[cmin, cmax], colorscale=colorscale,
                        cmin=cmin, cmax=cmax, showscale=True,
                        colorbar=dict(title=color_title, len=0.85)),
            showlegend=False, hoverinfo="skip"
        ))

    # Obstacle horizon band
    if horizon_deg > 0:
        fig_cart.add_shape(
            type="rect", x0=0, x1=360, y0=0, y1=horizon_deg,
            line=dict(width=0), fillcolor="rgba(120,120,120,0.18)", layer="below"
        )

    # "Now" marker (site local time)
    if show_now:
        az_now, alt_now, _ = solar_pos(pd.DatetimeIndex([now_local]).tz_convert("UTC"))
        if alt_now[0] > 0:
            fig_cart.add_trace(go.Scatter(
                x=[float(az_now[0])], y=[float(alt_now[0])], mode="markers",
                marker=dict(size=10, color="#ff4d4d", line=dict(width=1, color="#fff")),
                name="Now", showlegend=False
            ))

    # Hour labels (largest elevation per clock hour across whole year)
    if show_hour_labels:
        idx_loc = pd.date_range(pd.Timestamp(year,1,1,tz=tz),
                                pd.Timestamp(year,12,31,23,tz=tz), freq="1h")
        az_h, alt_h, _ = solar_pos(idx_loc.tz_convert("UTC"))
        mask = alt_h > 0
        if mask.any():
            dfh = pd.DataFrame({"hour": idx_loc.hour, "az": az_h, "alt": alt_h}).loc[mask]
            lab = dfh.sort_values("alt", ascending=False).groupby("hour", as_index=False).first()
            for _, r in lab.iterrows():
                fig_cart.add_annotation(
                    x=float(r["az"]), y=float(r["alt"]), text=str(int(r["hour"])),
                    showarrow=False, font=dict(size=10, color="#888")
                )

    fig_cart.update_layout(
        xaxis=dict(title="Azimuth (°)", range=[0, 360], dtick=45, mirror=True, ticks="outside", showgrid=False),
        yaxis=dict(title="Altitude (°)", range=[0, 90],  dtick=10, mirror=True, ticks="outside", showgrid=False),
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=50, r=25, t=30, b=70),
        height=520,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    _st_plotly_chart(fig_cart, use_container_width=True, config={"displayModeBar": True, "toImageButtonOptions": {"filename": f"{cname}_sunpath_cartesian", "format": "png", "scale": 2}})
    _add_manual_pdf_figure("Sun Path Cartesian", fig_cart)
    # Solar Irradiance Components (GHI, DNI, DHI) — above cloud coverage
    import plotly.express as px
    st.markdown("---")
    st.markdown("#### Solar Irradiance — GHI / DNI / DHI")
    st.caption("Annual overlay of Global Horizontal (GHI), Direct Normal (DNI), and Diffuse Horizontal (DHI) irradiance — identical layout to the Short-Term Prediction tab.")

    _ghi_col = get_metric_column(cdf, ["glohorrad", "ghi", "global_horizontal", "global_horiz", "solar", "radiation"])
    _dhi_col = get_metric_column(cdf, ["difhorrad", "dhi", "diffuse_horizontal", "dif_hor_rad"])
    _dni_col = get_metric_column(cdf, ["dirnorrad", "dni", "direct_normal", "dir_nor_rad"])

    if _ghi_col or _dhi_col or _dni_col:
        # Combined overlay line chart (daily means)
        fig_solar = go.Figure()
        _month_days = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        _month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        if _ghi_col:
            _ghi_s = pd.to_numeric(cdf[_ghi_col], errors="coerce").dropna()
            if not _ghi_s.empty:
                _ghi_doy = pd.DataFrame({'doy': _ghi_s.index.dayofyear, 'val': _ghi_s.values})
                _ghi_d = _ghi_doy.groupby('doy')['val'].mean()
                fig_solar.add_trace(go.Scatter(
                    x=_ghi_d.index, y=_ghi_d.values,
                    mode="lines", name="Global Horizontal (GHI)",
                    line=dict(color="#fbbf24", width=2),
                    fill="tozeroy", fillcolor="rgba(251, 191, 36, 0.2)"
                ))
        if _dni_col:
            _dni_s = pd.to_numeric(cdf[_dni_col], errors="coerce").dropna()
            if not _dni_s.empty:
                _dni_doy = pd.DataFrame({'doy': _dni_s.index.dayofyear, 'val': _dni_s.values})
                _dni_d = _dni_doy.groupby('doy')['val'].mean()
                fig_solar.add_trace(go.Scatter(
                    x=_dni_d.index, y=_dni_d.values,
                    mode="lines", name="Direct Normal (DNI)",
                    line=dict(color="#f97316", width=2, dash="dot")
                ))
        if _dhi_col:
            _dhi_s = pd.to_numeric(cdf[_dhi_col], errors="coerce").dropna()
            if not _dhi_s.empty:
                _dhi_doy = pd.DataFrame({'doy': _dhi_s.index.dayofyear, 'val': _dhi_s.values})
                _dhi_d = _dhi_doy.groupby('doy')['val'].mean()
                fig_solar.add_trace(go.Scatter(
                    x=_dhi_d.index, y=_dhi_d.values,
                    mode="lines", name="Diffuse Horizontal (DHI)",
                    line=dict(color="#60a5fa", width=2, dash="dash")
                ))

        fig_solar.update_layout(
            title="Solar Radiation — Annual Daily Means (W/m²)",
            yaxis_title="Irradiance (W/m²)", xaxis_title="",
            height=360, margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_solar.update_xaxes(showgrid=False, linecolor="rgba(255,255,255,0.2)",
                               tickmode="array",
                               tickvals=[1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
                               ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        fig_solar.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
        _st_plotly_chart(fig_solar, use_container_width=True)
        _add_manual_pdf_figure("Monthly Solar Insolation", fig_solar)
        _irr_shared_max = 0.0
        for _shared_col in [_dhi_col, _dni_col]:
            if _shared_col:
                _shared_vals = pd.to_numeric(cdf[_shared_col], errors="coerce")
                if not _shared_vals.dropna().empty:
                    _irr_shared_max = max(_irr_shared_max, float(_shared_vals.max()))

        # Individual heatmaps for DHI and DNI
        for _irr_label, _irr_col, _irr_scale, _irr_cname, _pdf_key in [
            ("DHI — Diffuse Horizontal Irradiance", _dhi_col, "Blues", "DHI (W/m²)", "DHI Irradiance Heatmap"),
            ("DNI — Direct Normal Irradiance", _dni_col, "YlOrRd", "DNI (W/m²)", "DNI Irradiance Heatmap"),
        ]:
            if _irr_col:
                _tmp_irr = pd.DataFrame({"doy": cdf.index.dayofyear, "hour": cdf.index.hour, "val": cdf[_irr_col]}).dropna()
                if not _tmp_irr.empty:
                    _mat_irr = _tmp_irr.pivot_table(index="hour", columns="doy", values="val", aggfunc="mean").sort_index()
                    _fig_irr = px.imshow(_mat_irr.values, origin="lower", aspect="auto",
                                    labels=dict(x="Day of Year", y="Hour", color=_irr_cname),
                                    title=f"Annual heatmap — {_irr_label}",
                                    height=360, color_continuous_scale=SOLAR_COLORSCALE)
                    _fig_irr.update_traces(zmin=0, zmax=_irr_shared_max if _irr_shared_max > 0 else 1.0)
                    _fig_irr.update_xaxes(tickvals=_month_days, ticktext=_month_names, side="bottom")
                    _fig_irr.update_yaxes(tickvals=[0, 6, 12, 18, 23], ticktext=["12AM", "6AM", "12PM", "6PM", "11PM"])
                    _st_plotly_chart(_fig_irr, use_container_width=True)
                    _add_manual_pdf_figure(_pdf_key, _fig_irr)

    # Cloud Coverage Frequencies 
    st.markdown("---")
    st.markdown("#### Cloud coverage")
    if "totskycvr" not in cdf:
        st.info("This EPW has no total sky cover column.")
    else:
        cc = cdf[["totskycvr"]].copy()
        cc["month"] = cc.index.month
        def bucket(x):
            if pd.isna(x): return np.nan
            if x <= 3:  return "Clear (0–3/10)"
            if x <= 7:  return "Intermediate (4–7/10)"
            return "Cloudy (8–10/10)"
        cc["category"] = cc["totskycvr"].apply(bucket)
        # SAFE aggregation → no duplicate 'month' on reset_index
        counts = (cc.value_counts(["month", "category"])
                    .rename("n")
                    .reset_index())
        counts["pct"] = 100 * counts["n"] / counts.groupby("month")["n"].transform("sum")
        freq = counts.drop(columns="n")
        cat_order = ["Clear (0–3/10)", "Intermediate (4–7/10)", "Cloudy (8–10/10)"]
        colors = ["#8FD3FF", "#7BB07B", "#F08A8A"]
        import plotly.express as px
        fig_cloud = px.bar(
            freq, x="month", y="pct", color="category",
            category_orders={"category": cat_order},
            color_discrete_sequence=colors, barmode="stack",
            labels={"month":"Month", "pct":"% of hours", "category":""},
            title="Cloud coverage by month (stacked frequency)"
        )
        _st_plotly_chart(fig_cloud, use_container_width=True)
        _add_manual_pdf_figure("Cloud Coverage", fig_cloud)
        # Hourly scatter faceted by month with smoothed mean overlay
        vcol = "totskycvr"
        vlabel = "Total sky cover (tenths)"
        scat = cdf[[vcol]].copy()
        scat["month"] = scat.index.month
        scat["hour"] = scat.index.hour

        fig_sc = px.scatter(
            scat,
            x="hour",
            y=vcol,
            facet_col="month",
            facet_col_wrap=4,
            opacity=0.35,
            labels={"hour": "Hour", vcol: vlabel},
            title=f"Daily scatter by month — {vlabel}",
            height=650,
        )
        # Build smoothed “monthly mean by hour” curve to overlay on each facet
        curve = (
            scat.groupby(["month", "hour"])[vcol]
                .mean()
                .reset_index()
                .sort_values(["month", "hour"])
        )
        curve["smooth"] = curve.groupby("month")[vcol].transform(
            lambda s: s.rolling(3, center=True, min_periods=1).mean()
        )
        # Add the smoothed line to each month facet
        for m in range(1, 13):
            sub = curve[curve["month"] == m]
            fig_sc.add_trace(
                go.Scatter(
                    x=sub["hour"],
                    y=sub["smooth"],
                    mode="lines",
                    line=dict(width=2),
                    name="Monthly mean (smoothed)",
                    showlegend=(m == 1),   # show once
                    legendgroup="meanline"
                ),
                row=((m - 1) // 4) + 1,
                col=((m - 1) % 4) + 1
            )
        # Add a dummy scatter just to expose a legend entry for the hourly points
        fig_sc.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=6, opacity=0.35),
                name="Hourly points",
                showlegend=True,
                legendgroup="points",
                hoverinfo="skip"
            )
        )
        # Replace facet labels “month=1..12” with Jan..Dec
        fig_sc.for_each_annotation(
            lambda a: a.update(
                text=pd.Timestamp(2001, int(a.text.split("=")[-1]), 1).strftime("%b")
            )
        )
        # (Optional) tidy layout
        fig_sc.update_layout(
            legend=dict(
                orientation="h",
                x=1, xanchor="right",
                y=1.02, yanchor="bottom"  # just under the top margin
            ),
            margin=dict(t=110, b=40, l=40, r=20)
        )

        _st_plotly_chart(fig_sc, use_container_width=True)
        _add_manual_pdf_figure("Cloud Coverage Scatter", fig_sc)
        # ---- Annual heatmap (day-of-year × hour) ----
        tmp = pd.DataFrame({
            "doy": cdf.index.dayofyear,
            "hour": cdf.index.hour,
            "val": cdf[vcol]
        }).dropna()
        mat = tmp.pivot_table(index="hour", columns="doy", values="val", aggfunc="mean").sort_index()
        scale = "RdYlBu_r" if ("Wh/m²" in vlabel or "Dry-bulb" in vlabel) else "Blues"
        fig_hm = px.imshow(mat.values, origin="lower", aspect="auto",
                        labels=dict(x="Day of Year", y="Hour", color=vlabel),
                        title=f"Annual heatmap — {vlabel}",
                        height=360, color_continuous_scale=scale)
        fig_hm.update_xaxes(side="bottom")
        # Add month labels on x-axis
        month_days = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        fig_hm.update_xaxes(tickvals=month_days, ticktext=month_names)
        fig_hm.update_yaxes(tickvals=[0, 6, 12, 18, 23], ticktext=["12AM", "6AM", "12PM", "6PM", "11PM"])
        _st_plotly_chart(fig_hm, use_container_width=True)
        _add_manual_pdf_figure("Cloud Coverage Heatmap", fig_hm)

        st.markdown("### Advanced Solar Diagnostics")
        st.caption("Figures generated for publication-standard reporting.")
        
        station_name = _safe_location_label(st.session_state.get("header", {}))
        epw_metadata = _location_meta(st.session_state.get("header", {}))
        df_cwec = _prepare_advanced_figure_df(cdf)
        
        # Fig A
        fig_a, cap_a = build_fig_a_longwave_heatmap(df_cwec, station_name)
        _st_plotly_chart(fig_a, use_container_width=True)
        st.caption(cap_a)
        _add_manual_pdf_figure("longwave_irradiance_heatmap", fig_a)
        
        # Fig B
        fig_b, cap_b = build_fig_b_hourly_incoming_radiation_boxwhisker(df_cwec, station_name)
        _st_plotly_chart(fig_b, use_container_width=True)
        st.caption(cap_b)
        _add_manual_pdf_figure("hourly_incoming_radiation_boxwhisker", fig_b)
        
        # Fig C
        fig_c, cap_c = build_fig_c_inclined_surface_insolation(df_cwec, station_name, epw_metadata)
        _st_plotly_chart(fig_c, use_container_width=True)
        st.caption(cap_c)
        _add_manual_pdf_figure("inclined_surface_insolation", fig_c)

def build_fig_g_seasonal_psychrometric(df, utci_baseline, station_name):
    df = _prepare_advanced_figure_df(df)
    missing = [c for c in ["drybulb_C", "rh_pct"] if c not in df.columns]
    if missing:
        return placeholder_figure("Dry-bulb and relative humidity are required for seasonal psychrometrics."), ""
    utci_baseline = np.asarray(utci_baseline, dtype=float)
    if len(utci_baseline) != len(df):
        return placeholder_figure("UTCI scenario length does not match hourly weather data."), ""

    if 'atmos_pressure' in df and df['atmos_pressure'].notna().any():
        P_kPa = float(np.nanmedian(df['atmos_pressure'])) / 1000.0
    else:
        P_kPa = 101.325
    T_pts = df['drybulb_C'].to_numpy(float)
    RH_pts = df['rh_pct'].to_numpy(float)
    Pv_pts = (RH_pts / 100.0) * psh.p_ws_kPa(T_pts)
    w_pts = psh.w_from_Pv_kPa(Pv_pts, P_kPa)
    Y_gpkg = psh.gpkg(w_pts)
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=['Winter (DJF)', 'Spring (MAM)', 'Summer (JJA)', 'Autumn (SON)'], shared_xaxes=True, shared_yaxes=True)
    seasons = [
        ('Winter (DJF)', [12, 1, 2], 1, 1),
        ('Spring (MAM)', [3, 4, 5], 1, 2),
        ('Summer (JJA)', [6, 7, 8], 2, 1),
        ('Autumn (SON)', [9, 10, 11], 2, 2)
    ]
    
    ashrae_x = [18, 26, 26, 18, 18]
    ashrae_y = [4, 4, 12, 12, 4]
    
    for name, months, r, c in seasons:
        mask = df.index.month.isin(months)
        fig.add_trace(go.Scatter(
            x=T_pts[mask], y=Y_gpkg[mask], mode='markers',
            marker=dict(size=2, opacity=0.3, color=utci_baseline[mask], colorscale='RdYlBu_r', cmin=9, cmax=32, showscale=(r==1 and c==1)),
            name=name
        ), row=r, col=c)
        fig.add_trace(go.Scatter(x=ashrae_x, y=ashrae_y, mode='lines', line=dict(color='black', dash='dash'), name='ASHRAE', showlegend=False), row=r, col=c)

    fig.update_layout(height=600, margin=dict(l=40, r=40, t=60, b=40), showlegend=False)
    fig.update_xaxes(title_text="Dry-Bulb (°C)", row=2)
    fig.update_yaxes(title_text="Humidity Ratio (g/kg)", col=1)
    
    summer_mask = df.index.month.isin([6, 7, 8])
    comfort_mask = (utci_baseline > 9) & (utci_baseline < 26)
    summer_comfort_pct = round((comfort_mask & summer_mask).sum() / summer_mask.sum() * 100, 1) if summer_mask.sum() > 0 else 0
    
    caption_template = "Seasonal psychrometric scatter plots show the joint distribution of dry-bulb temperature and humidity ratio for all hours in each season, with the ASHRAE comfort zone overlaid and points colored by UTCI thermal stress category. For {station}, {summer_comfort_pct}% of summer hours fall within the comfort zone before conditioning — with winter points concentrated below 0°C — confirming heating as the dominant seasonal mechanical load."
    caption = safe_format_caption(caption_template, {"station": station_name, "summer_comfort_pct": summer_comfort_pct})
    
    return fig, caption

def build_fig_h_hourly_psychrometric_paths(df, station_name):
    df = _prepare_advanced_figure_df(df)
    missing = [c for c in ["drybulb_C", "rh_pct"] if c not in df.columns]
    if missing:
        return placeholder_figure("Dry-bulb and relative humidity are required for hourly psychrometric paths."), ""

    if 'atmos_pressure' in df and df['atmos_pressure'].notna().any():
        P_kPa = float(np.nanmedian(df['atmos_pressure'])) / 1000.0
    else:
        P_kPa = 101.325
    T_pts = df['drybulb_C']
    RH_pts = df['rh_pct']
    Pv_pts = (RH_pts / 100.0) * psh.p_ws_kPa(T_pts)
    w_pts = psh.w_from_Pv_kPa(Pv_pts, P_kPa)
    Y_gpkg = pd.Series(psh.gpkg(w_pts), index=df.index)
    
    fig = go.Figure()
    colors = px.colors.sample_colorscale('RdYlBu_r', 12)
    
    comfort_months_list = []
    
    for i, month in enumerate(range(1, 13)):
        mask = df.index.month == month
        mean_t = T_pts[mask].groupby(df[mask].index.hour).mean()
        mean_y = Y_gpkg[mask].groupby(df[mask].index.hour).mean()
        
        if 14 in mean_t and 14 in mean_y:
            t14 = mean_t[14]
            y14 = mean_y[14]
            if 18 <= t14 <= 26 and 4 <= y14 <= 12:
                month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                comfort_months_list.append(month_names[i])
        
        fig.add_trace(go.Scatter(x=mean_t, y=mean_y, mode='lines+markers', line=dict(color=colors[i]), name=calendar.month_abbr[month]))
        if len(mean_t) > 14 and 14 in mean_t.index and 13 in mean_t.index:
            fig.add_annotation(
                x=mean_t[14], y=mean_y[14], ax=mean_t[13], ay=mean_y[13],
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor=colors[i]
            )

    ashrae_x = [18, 26, 26, 18, 18]
    ashrae_y = [4, 4, 12, 12, 4]
    fig.add_trace(go.Scatter(x=ashrae_x, y=ashrae_y, mode='lines', line=dict(color='black', dash='dash'), name='Comfort Zone'))
    
    fig.update_layout(title="Monthly Mean 24-Hour Psychrometric Paths", xaxis_title="Dry-Bulb (°C)", yaxis_title="Humidity Ratio (g/kg)", height=600)
    
    comfort_months_str = ", ".join(comfort_months_list) if comfort_months_list else "no"
    comfort_hours = "09:00–17:00"
    
    caption_template = "Monthly mean 24-hour psychrometric paths trace the typical daily cycle of temperature and humidity for each calendar month, with arrowheads at 14:00 showing the direction of the daily loop. For {station}, {comfort_months} paths pass through the comfort zone near midday, with the natural ventilation opportunity window open approximately {comfort_hours} — the primary passive cooling scheduling target."
    caption = safe_format_caption(caption_template, {"station": station_name, "comfort_months": comfort_months_str, "comfort_hours": comfort_hours})
    
    return fig, caption

def _diurnal_comfort_matrix(df, utci_a, utci_b, title, station_name, metric_name, scenario_a_name, scenario_b_name):
    df = _prepare_advanced_figure_df(df)
    if not isinstance(df.index, pd.DatetimeIndex):
        return placeholder_figure(f"{title}: datetime index required."), 0, "none"
    utci_a = np.asarray(utci_a, dtype=float)
    utci_b = np.asarray(utci_b, dtype=float)
    if len(utci_a) != len(df) or len(utci_b) != len(df):
        return placeholder_figure(f"{title}: UTCI scenario length mismatch."), 0, "none"

    rows = [('Winter (DJF)', [12,1,2]), ('Spring (MAM)', [3,4,5]), ('Summer (JJA)', [6,7,8]), ('Autumn (SON)', [9,10,11])]
    cols = [('Night (20-05h)', [20,21,22,23,0,1,2,3,4,5]), ('Morning (06-11h)', [6,7,8,9,10,11]), ('Afternoon (12-17h)', [12,13,14,15,16,17]), ('Evening (18-19h)', [18,19])]
    
    fig = make_subplots(rows=4, cols=4, subplot_titles=[c[0] for c in cols], row_titles=[r[0] for r in rows], shared_yaxes=True)
    
    def cat(u):
        if pd.isna(u): return "No Data"
        if u < 9: return "Cold Stress"
        if u <= 26: return "Comfort"
        if u <= 32: return "Mod Heat"
        if u <= 38: return "Strong Heat"
        return "Extreme Heat"
        
    cats = ["Cold Stress", "Comfort", "Mod Heat", "Strong Heat", "Extreme Heat"]
    colors = ['#313695', '#4575b4', '#fdae61', '#f46d43', '#a50026']
    
    max_benefit = 0
    best_season = "none"
    
    for r_idx, (r_name, r_months) in enumerate(rows):
        for c_idx, (c_name, c_hours) in enumerate(cols):
            mask = df.index.month.isin(r_months) & df.index.hour.isin(c_hours)
            if not mask.any(): continue
            
            ua = pd.Series(utci_a[mask]).apply(cat).value_counts(normalize=True) * 100
            ub = pd.Series(utci_b[mask]).apply(cat).value_counts(normalize=True) * 100
            
            comfort_a = ua.get("Comfort", 0)
            comfort_b = ub.get("Comfort", 0)
            diff = comfort_b - comfort_a
            if diff > max_benefit:
                max_benefit = diff
                best_season = r_name
                
            for i, c_cat in enumerate(cats):
                fig.add_trace(go.Bar(name=c_cat, x=[scenario_a_name, scenario_b_name], y=[ua.get(c_cat,0), ub.get(c_cat,0)], marker_color=colors[i], showlegend=(r_idx==0 and c_idx==0)), row=r_idx+1, col=c_idx+1)
    
    fig.update_layout(barmode='stack', title=title, height=800)
    return fig, max_benefit, best_season.split(" ")[0]

def build_fig_i_diurnal_comfort_shading(df, utci_dict, station_name):
    fig, max_benefit, best_season = _diurnal_comfort_matrix(df, utci_dict['baseline'], utci_dict['shaded'], "Diurnal Thermal Comfort Frequency — Shading Scenario", station_name, "shading", "Ambient", "Shaded (-15°C MRT)")
    cap = safe_format_caption("The diurnal thermal comfort frequency matrix compares UTCI comfort category distributions under ambient and shaded (MRT reduced by 15°C) conditions across all seasons and times of day. For {station}, shading delivers the greatest benefit in {best_shade_season} afternoons, shifting an estimated {shade_benefit}% of hours from heat-stress into the neutral comfort band — the strongest single passive intervention available at this location.", {"station": station_name, "best_shade_season": best_season, "shade_benefit": round(max_benefit, 1)})
    return fig, cap

def build_fig_j_diurnal_comfort_wind(df, utci_dict, station_name):
    df = _prepare_advanced_figure_df(df)
    fig, max_benefit, best_season = _diurnal_comfort_matrix(df, utci_dict['calm'], utci_dict['baseline'], "Diurnal Thermal Comfort Frequency — Wind Scenario", station_name, "wind", "Calm (0.5m/s)", "Measured Wind")
    wind_mean = round(df['wind_speed_ms'].mean(), 1) if 'wind_speed_ms' in df else "[data unavailable]"
    cap = safe_format_caption("The wind-scenario comfort matrix compares UTCI distributions under measured wind conditions against a calm-wind (0.5 m/s) baseline, isolating the cooling benefit of site wind exposure across seasons and times of day. For {station} with mean wind speed {wind_mean} m/s, wind-driven cooling provides the most significant comfort benefit in {wind_benefit_season}, reducing moderate heat stress hours by increasing convective heat loss from the body.", {"station": station_name, "wind_mean": wind_mean, "wind_benefit_season": best_season})
    return fig, cap

def build_fig_k_diurnal_comfort_humidity(df, utci_dict, station_name):
    df = _prepare_advanced_figure_df(df)
    fig, max_benefit, best_season = _diurnal_comfort_matrix(df, utci_dict['baseline'], utci_dict['neutral'], "Diurnal Thermal Comfort Frequency — Humidity Scenario", station_name, "humidity", "Ambient", "Neutral (RH 50%)")
    rh_mean = round(df['rh_pct'].mean(), 1) if 'rh_pct' in df else "[data unavailable]"
    
    if max_benefit > 5:
        humidity_verdict = "is a meaningful strategy"
        humidity_impact_sentence = f"dehumidification to 50% RH could shift up to {round(max_benefit,1)}% of hours into the comfort band during peak humid periods."
    else:
        humidity_verdict = "provides limited benefit at this location"
        humidity_impact_sentence = "dry-bulb temperature, not humidity, is the primary driver of discomfort here."
        
    cap = safe_format_caption("The humidity-scenario comfort matrix compares UTCI distributions under measured humidity against a neutralised RH=50% baseline, quantifying how much thermal discomfort at this location is humidity-driven. For {station} with annual mean RH {rh_mean}%, humidity control {humidity_verdict} — {humidity_impact_sentence}", {"station": station_name, "rh_mean": rh_mean, "humidity_verdict": humidity_verdict, "humidity_impact_sentence": humidity_impact_sentence})
    return fig, cap

def render_psychrometrics_page():
    cdf = st.session_state.get("cdf")
    if cdf is None:
        return
    needed = ["drybulb", "relhum"]
    if not all(k in cdf.columns for k in needed):
        st.info("This EPW is missing required fields for the psychrometric plot.")
        return

    # Pressure
    if "atmos_pressure" in cdf and cdf["atmos_pressure"].notna().any():
        P_kPa = float(np.nanmedian(cdf["atmos_pressure"].values)) / 1000.0
    else:
        P_kPa = 101.325

    location_label = get_clean_city_name()
    st.markdown(f"<h3>{location_label} – Psychrometrics</h3>", unsafe_allow_html=True)
    st.caption("Plot hourly EPW data on the classic psychrometric grid with bioclimatic strategy zones, frequency heatmap, and toggleable metric overlays.")

    # ── Controls ──
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    month_range = ctrl1.slider("Month Range", 1, 12, (1, 12), key="psy_month_range")
    overlay_choice = ctrl2.selectbox("Comfort Overlay", ["None", "Givoni Bioclimatic Chart", "ASHRAE 55 Comfort Zone"], index=1, key="psy_overlay")

    mean_outdoor_t = 20.0
    if overlay_choice == "Givoni Bioclimatic Chart":
        if "drybulb" in cdf.columns:
            daily_mean = cdf["drybulb"].resample("1D").mean()
            running_mean = daily_mean.rolling(30, min_periods=1).mean()
            mean_outdoor_t = float(running_mean.median())
        mean_outdoor_t = ctrl3.slider("Mean Outdoor Temp (°C)", 5.0, 35.0, float(round(mean_outdoor_t, 1)), step=0.5, key="psy_trm", help="Adjusts adaptive comfort zone width.")

    auto_zoom = ctrl3.checkbox("Auto zoom to EPW range", value=True, key="psy_autozoom") if overlay_choice != "Givoni Bioclimatic Chart" else st.sidebar.checkbox("Auto zoom", value=True, key="psy_autozoom")

    with st.expander("Chart Metric Lines", expanded=False):
        mc1, mc2, mc3, mc4 = st.columns(4)
        show_rh = mc1.checkbox("RH Lines", True, key="psy_rh")
        show_enthalpy = mc2.checkbox("Enthalpy", True, key="psy_enth")
        show_volume = mc3.checkbox("Spec. Volume", False, key="psy_vol")
        show_wetbulb = mc4.checkbox("Wet-Bulb", False, key="psy_twb")

    # ── Data prep ──
    dfp = cdf[["drybulb", "relhum"]].dropna().copy()
    dfp = dfp[(dfp.index.month >= month_range[0]) & (dfp.index.month <= month_range[1])]
    if dfp.empty:
        st.info("No data points in selected range.")
        return

    T_pts = dfp["drybulb"].to_numpy(float)
    RH_pts = dfp["relhum"].to_numpy(float)
    Pv_pts = (RH_pts / 100.0) * psh.p_ws_kPa(T_pts)
    w_pts = psh.w_from_Pv_kPa(Pv_pts, P_kPa)
    Y_gpkg = psh.gpkg(w_pts)
    dp_pts = psh.dew_point_C(T_pts, RH_pts)
    tw_pts = psh.wet_bulb_C(T_pts, RH_pts)
    h_pts = psh.enthalpy_kJkg(T_pts, w_pts)
    v_pts = psh.specific_vol(T_pts, w_pts, P_kPa)

    # Axis ranges
    if auto_zoom:
        x_min = max(-15, float(np.nanmin(T_pts)) - 3)
        x_max = min(50, float(np.nanmax(T_pts)) + 3)
        y_min = max(0, float(np.nanmin(Y_gpkg)) - 1)
        y_max = min(35, float(np.nanmax(Y_gpkg)) + 2)
    else:
        x_min, x_max, y_min, y_max = -10.0, 50.0, 0.0, 30.0

    T_axis = np.linspace(x_min, x_max, 500)

    # ── Build figure ──
    fig_psy = go.Figure()

    # Zone color palette — distinct solid colors matching Climate Consultant
    zone_palette = {
        "COMFORT ZONE": "#2ca02c",
        "NATURAL\nVENTILATION": "#17becf",
        "EVAPORATIVE\nCOOLING": "#1f77b4",
        "MASS\nCOOLING": "#ff7f0e",
        "NIGHT VENT\n& MASS COOL": "#d62728",
        "A/C &\nDEHUMIDIFICATION": "#9467bd",
        "PASSIVE SOLAR\nHEATING": "#bcbd22",
        "INTERNAL\nGAINS": "#8c564b",
        "ACTIVE\nSOLAR": "#e377c2",
        "HEATING": "#7f7f7f",
        "HUMIDIFICATION": "#aec7e8",
        "Unclassified": "#cccccc",
    }

    # ── Givoni zone outlines (drawn first, behind dots) ──
    zones = None
    if overlay_choice == "Givoni Bioclimatic Chart":
        zones = psh.givoni_zones(P_kPa, mean_outdoor_t)
        for zname, verts in zones.items():
            xs = [v[0] for v in verts] + [verts[0][0]]
            ys = [v[1] for v in verts] + [verts[0][1]]
            zcolor = zone_palette.get(zname, "#999999")
            # Convert hex to rgba fill
            r, g, b = int(zcolor[1:3], 16), int(zcolor[3:5], 16), int(zcolor[5:7], 16)
            fig_psy.add_trace(go.Scatter(
                x=xs, y=ys, mode="none",
                fill="toself", fillcolor=f"rgba({r},{g},{b},0.12)",
                name=zname.replace("\n", " "), showlegend=False, hoverinfo="name",
            ))
            # Label at centroid
            cx = np.mean([v[0] for v in verts])
            cy = np.mean([v[1] for v in verts])
            fig_psy.add_annotation(
                x=cx, y=cy, text=f"<b>{zname}</b>", showarrow=False,
                font=dict(size=9, color=zcolor), align="center",
                bgcolor="rgba(0,0,0,0.5)", borderpad=2,
            )

    elif overlay_choice == "ASHRAE 55 Comfort Zone":
        for label, poly, color in [
            ("Summer Comfort", [(23.5,1),(23.5,12),(26.5,12),(26.5,1)], "rgba(255,80,80,0.2)"),
            ("Winter Comfort", [(20,1),(20,12),(24,12),(24,1)], "rgba(80,80,255,0.2)"),
        ]:
            xs = [v[0] for v in poly] + [poly[0][0]]
            ys = [v[1] for v in poly] + [poly[0][1]]
            fig_psy.add_trace(go.Scatter(
                x=xs, y=ys, mode="none", fill="toself", fillcolor=color,
                name=label, showlegend=True, hoverinfo="name",
            ))

    # ── Saturation curve ──
    y_sat = psh.gpkg(psh.w_sat(T_axis, P_kPa))
    fig_psy.add_trace(go.Scatter(
        x=T_axis, y=y_sat, mode="lines",
        line=dict(width=2.5, color="#333333"),
        name="Saturation", showlegend=False,
        hovertemplate="100%% RH<br>T: %{x:.1f}°C<br>W: %{y:.2f} g/kg<extra></extra>",
    ))

    # ── RH isolines ──
    if show_rh:
        for rh_val in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
            y_rh = psh.rh_curve(T_axis, rh_val, P_kPa)
            fig_psy.add_trace(go.Scatter(
                x=T_axis, y=y_rh, mode="lines",
                line=dict(width=0.7, dash="dot", color="rgba(100,100,100,0.5)"),
                showlegend=False, hoverinfo="skip",
            ))

    # ── Enthalpy lines ──
    if show_enthalpy:
        for h_val in [10, 20, 30, 40, 50, 60, 70, 80, 100]:
            y_h = psh.enthalpy_w_line(T_axis, h_val)
            valid = (y_h >= y_min) & (y_h <= y_max * 1.1)
            if valid.any():
                fig_psy.add_trace(go.Scatter(
                    x=T_axis[valid], y=np.clip(y_h[valid], y_min, y_max), mode="lines",
                    line=dict(width=0.6, dash="dash", color="rgba(180,120,0,0.4)"),
                    showlegend=False, hoverinfo="skip",
                ))

    # ── Specific volume lines ──
    if show_volume:
        for v_val in [0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90]:
            y_v = psh.volume_w_line(T_axis, v_val, P_kPa)
            valid = (y_v >= y_min) & (y_v <= y_max)
            if valid.any():
                fig_psy.add_trace(go.Scatter(
                    x=T_axis[valid], y=y_v[valid], mode="lines",
                    line=dict(width=0.6, dash="dot", color="rgba(60,100,200,0.4)"),
                    showlegend=False, hoverinfo="skip",
                ))

    # ── Hourly dots colored by zone ──
    custom = np.c_[RH_pts, Pv_pts * 1000, h_pts, v_pts, dp_pts, tw_pts]
    hover_tpl = ("<b>%{text}</b><br>Tdb %{x:.1f}°C<br>W %{y:.2f} g/kg<br>"
                 "RH %{customdata[0]:.1f}%<br>h %{customdata[2]:.1f} kJ/kg<br>"
                 "Tdp %{customdata[4]:.1f}°C<br>Twb %{customdata[5]:.1f}°C<extra></extra>")

    if zones is not None:
        # Classify each hourly point into a zone
        labels = psh.classify_points_to_zones(T_pts, Y_gpkg, zones)
        # Plot one trace per zone (for legend)
        unique_labels = list(dict.fromkeys(labels))  # preserve order
        for zlabel in unique_labels:
            mask = labels == zlabel
            if not mask.any():
                continue
            dot_color = zone_palette.get(zlabel, "#cccccc")
            fig_psy.add_trace(go.Scatter(
                x=T_pts[mask], y=Y_gpkg[mask], mode="markers",
                marker=dict(size=4, color=dot_color, opacity=0.7),
                name=zlabel.replace("\n", " "),
                showlegend=True,
                customdata=custom[mask],
                text=[zlabel.replace("\n", " ")] * int(mask.sum()),
                hovertemplate=hover_tpl,
            ))
    else:
        # No Givoni overlay — color dots by temperature with a colorbar key
        fig_psy.add_trace(go.Scatter(
            x=T_pts, y=Y_gpkg, mode="markers",
            marker=dict(
                size=4, opacity=0.6, color=T_pts, colorscale="Turbo",
                showscale=True,
                colorbar=dict(
                    title="Dry Bulb<br>°C", len=0.6, y=0.3,
                    thickness=14, tickfont=dict(size=10),
                ),
            ),
            name="Hourly Data", showlegend=True, customdata=custom,
            text=["Hourly"] * len(T_pts),
            hovertemplate=hover_tpl,
        ))

    # ── RH labels on right margin ──
    for rh_val in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        y_at_xmax = float(psh.gpkg(psh.w_from_Pv_kPa(
            np.array([(rh_val / 100.0) * psh.p_ws_kPa(np.array([x_max]))[0]]), P_kPa))[0])
        if y_min < y_at_xmax < y_max:
            fig_psy.add_annotation(
                x=x_max, y=y_at_xmax, text=f"{rh_val}%", xanchor="left",
                showarrow=False, font=dict(size=9, color="#555555"),
            )

    # ── Layout (white background like Climate Consultant) ──
    fig_psy.update_xaxes(
        range=[x_min, x_max], dtick=5,
        title="Dry Bulb Temperature (°C)",
        showgrid=True, gridcolor="rgba(200,200,200,0.3)",
        zeroline=False, showline=True, linecolor="#aaaaaa",
        ticks="outside", ticklen=5,
    )
    fig_psy.update_yaxes(
        range=[y_min, y_max], dtick=5,
        title="Humidity Ratio (g/kg dry air)",
        showgrid=True, gridcolor="rgba(200,200,200,0.3)",
        zeroline=False, showline=True, linecolor="#aaaaaa",
        ticks="outside", ticklen=5,
    )
    fig_psy.update_layout(
        height=720, margin=dict(l=70, r=60, t=80, b=80),
        showlegend=True,
        hovermode="closest",
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Psychrometric Chart – ASHRAE Style", x=0.01, xanchor="left", yanchor="top", font=dict(size=18)),
        legend=dict(
            orientation="v", x=1.02, y=1, xanchor="left", yanchor="top",
            bgcolor="rgba(0,0,0,0)", font=dict(size=10),
            itemsizing="constant",
        ),
    )

    clean_loc = location_label.replace(" ", "_").replace(",", "").replace("__", "_")
    _st_plotly_chart(fig_psy, use_container_width=True, config={"displaylogo": False, "modeBarButtonsToRemove": ["select2d","lasso2d"], "toImageButtonOptions": {"filename": f"{clean_loc}_psychrometric_chart", "format": "png", "scale": 2}})
    _add_manual_pdf_figure("Psychrometric Chart", fig_psy)

    with st.expander("Export psychrometric chart", expanded=False):
        st.caption("Preparing SVG uses Kaleido and can take a moment, so downloads are generated only when requested.")
        if st.button("Prepare psychrometric downloads", key="prepare_psy_downloads"):
            try:
                st.session_state["psy_svg_bytes"] = fig_psy.to_image(format="svg", width=1200, height=900, scale=2)
                st.session_state["psy_svg_error"] = ""
            except Exception as e:
                st.session_state["psy_svg_error"] = str(e)
            try:
                st.session_state["psy_html_bytes"] = fig_psy.to_html(include_plotlyjs="cdn").encode("utf-8")
                st.session_state["psy_html_error"] = ""
            except Exception as e:
                st.session_state["psy_html_error"] = str(e)

        d1, d2 = st.columns(2)
        if st.session_state.get("psy_svg_bytes"):
            d1.download_button("Download Chart (SVG)", st.session_state["psy_svg_bytes"], f"{clean_loc}_psychrometric_chart.svg", "image/svg+xml", key="dl_psy_svg")
        elif st.session_state.get("psy_svg_error"):
            d1.warning(f"SVG export failed: {st.session_state['psy_svg_error']}")
        if st.session_state.get("psy_html_bytes"):
            d2.download_button("Download Chart (HTML)", st.session_state["psy_html_bytes"], f"{clean_loc}_psychrometric_chart.html", "text/html", key="dl_psy_html")
        elif st.session_state.get("psy_html_error"):
            d2.warning(f"HTML export failed: {st.session_state['psy_html_error']}")

    # Zone summary
    if overlay_choice == "Givoni Bioclimatic Chart":
        st.markdown("### Bioclimatic Zone Summary")
        st.caption("Hours in each Givoni strategy zone. Points may overlap multiple zones.")
        zones = psh.givoni_zones(P_kPa, mean_outdoor_t)
        zone_hours = psh.count_hours_in_zones(T_pts, Y_gpkg, zones)
        total = len(T_pts)
        rows = []
        for zname, hrs in sorted(zone_hours.items(), key=lambda x: -x[1]):
            pct = (hrs / total * 100) if total > 0 else 0
            rows.append({"Strategy Zone": zname.replace("\n"," "), "Hours": hrs, "% of Period": f"{pct:.1f}%"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Advanced Psychrometric & Comfort Diagnostics")
    st.caption("Publication-standard diagnostics generated by the PDF engine.")
    extra = _get_extra_figures()
    for title in [
        "Seasonal Psychrometric Points",
        "Hourly Psychrometric Paths",
        "Diurnal Thermal Comfort Frequency - Shading Scenario",
        "Diurnal Thermal Comfort Frequency - Wind Scenario",
        "Diurnal Thermal Comfort Frequency - Humidity Scenario",
    ]:
        if title in extra:
            _st_plotly_chart(extra[title], use_container_width=True)
            _add_manual_pdf_figure(title, extra[title])






WIND_DIRECTION_LABELS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def _clean_wind_frame(df: pd.DataFrame, wind_col: str, wdir_col: str) -> pd.DataFrame:
    out = df.copy()
    speed = pd.to_numeric(out[wind_col], errors="coerce")
    speed = speed.mask((speed < 0) | (speed >= 999))
    direction = _normalize_wind_dir(out[wdir_col])
    out[wind_col] = speed
    out[wdir_col] = direction
    return out.dropna(subset=[wind_col, wdir_col])


def _wind_direction_categories(direction: pd.Series) -> pd.Categorical:
    direction = pd.to_numeric(direction, errors="coerce") % 360
    sector_idx = np.floor(((direction + 11.25) % 360) / 22.5).fillna(-1).astype(int).clip(-1, 15)
    return pd.Categorical.from_codes(
        sector_idx.to_numpy(dtype=int),
        categories=WIND_DIRECTION_LABELS,
        ordered=True,
    )


def _wind_speed_bin_spec(speed_mps: pd.Series, max_bins: int = 6) -> tuple[list[float], list[str]]:
    clean = pd.to_numeric(speed_mps, errors="coerce").dropna()
    clean = clean[(clean >= 0) & (clean < 999)]
    if clean.empty:
        return [0.0, 0.5], ["0.0-0.5"]

    max_speed = float(clean.max())
    if not np.isfinite(max_speed) or max_speed <= 0:
        return [0.0, 0.5], ["0.0-0.5"]

    upper = max(max_speed, 0.5)
    raw_bins = np.linspace(0.0, upper, max_bins + 1)
    decimals = 2 if upper < 2 else 1
    bins = np.unique(np.round(raw_bins, decimals))
    if len(bins) < 2:
        bins = np.array([0.0, upper])
    bins = bins.astype(float)
    bins[-1] = max(bins[-1], max_speed) + max(upper * 1e-6, 1e-9)
    labels = [f"{bins[i]:g}-{bins[i + 1]:g}" for i in range(len(bins) - 1)]
    return bins.tolist(), labels


WIND_ROSE_DIR_LABELS = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]


def _wind_rose_direction_categories(direction: pd.Series) -> pd.Categorical:
    direction = _normalize_wind_dir(direction)
    sector_idx = np.floor(((direction + 11.25) % 360) / 22.5).fillna(-1).astype(int).clip(-1, 15)
    return pd.Categorical.from_codes(
        sector_idx.to_numpy(dtype=int),
        categories=WIND_ROSE_DIR_LABELS,
        ordered=True,
    )


def _wind_rose_speed_bins_mph(speed_mps: pd.Series, num_bins: int = 6) -> tuple[list[float], list[str]]:
    speed_mph = pd.to_numeric(speed_mps, errors="coerce") * 2.23694
    speed_mph = speed_mph.mask((speed_mph < 0) | (speed_mph >= 999 * 2.23694)).dropna()
    if speed_mph.empty:
        return [0.0, 1.0], ["0.0-1.0"]

    max_speed = float(speed_mph.max())
    if pd.isna(max_speed) or max_speed <= 0:
        max_speed = 1.0

    speed_bins = np.linspace(0, max_speed, num_bins + 1)
    speed_bins[-1] = max(speed_bins[-1], max_speed + 1e-9)
    step = max_speed / max(num_bins, 1)
    decimals = 1 if step >= 0.1 else min(4, int(np.ceil(-np.log10(max(step, 1e-6)))) + 1)
    speed_labels = [
        f"{speed_bins[i]:.{decimals}f}-{speed_bins[i + 1]:.{decimals}f}"
        for i in range(len(speed_bins) - 1)
    ]
    return speed_bins.tolist(), speed_labels


def create_wind_rose(df):
    df = df.copy()

    # Determine wind speed and direction columns dynamically
    wind_spd_col = None
    if "2dSpeed_m_s" in df.columns:
        wind_spd_col = "2dSpeed_m_s"
    else:
        wind_spd_col = get_metric_column(df, ["wind_speed", "windspeed", "windspd", "ws", "wspd"])

    wind_dir_col = None
    if "Azimuth_deg" in df.columns:
        wind_dir_col = "Azimuth_deg"
    else:
        wind_dir_col = get_metric_column(df, ["wind_direction", "winddir", "wd", "wdir", "wind_dir", "HourlyWindDirection"])

    if not wind_spd_col or not wind_dir_col:
        return None

    df[wind_spd_col] = pd.to_numeric(df[wind_spd_col], errors="coerce")
    df[wind_dir_col] = _normalize_wind_dir(df[wind_dir_col])

    df = df.dropna(subset=[wind_spd_col, wind_dir_col])
    # Convert wind speed from m/s to mph
    df.loc[:, "2dSpeed_mph"] = df[wind_spd_col] * 2.23694

    df = df[(df["2dSpeed_mph"] >= 0) & (df["2dSpeed_mph"] < 999)]
    if df.empty:
        return None

    # Define direction bins
    dir_bins = np.arange(0, 361, 22.5)
    dir_labels = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
    ]

    # Dynamically create speed bins based on data
    max_speed = df["2dSpeed_mph"].max()
    num_bins = 6
    speed_bins = np.linspace(0, max_speed, num_bins + 1)
    speed_labels = [
        f"{speed_bins[i]:.1f}-{speed_bins[i + 1]:.1f}"
        for i in range(len(speed_bins) - 1)
    ]

    # Categorize data
    df.loc[:, "dir_cat"] = pd.cut(
        df[wind_dir_col],
        bins=dir_bins,
        labels=dir_labels,
        include_lowest=True,
        ordered=False,
    )
    df.loc[:, "speed_cat"] = pd.cut(
        df["2dSpeed_mph"],
        bins=speed_bins,
        labels=speed_labels,
        include_lowest=True,
        ordered=False,
    )

    # Count occurrences and calculate percentages
    wind_data = (
        df.groupby(["dir_cat", "speed_cat"], observed=True).size().unstack(fill_value=0)
    )
    # Convert index and columns to string lists to prevent any CategoricalIndex reindexing issues
    wind_data.index = [str(i) for i in wind_data.index]
    wind_data.columns = [str(c) for c in wind_data.columns]
    
    wind_data = wind_data.reindex(index=dir_labels, columns=speed_labels, fill_value=0)
    
    total_count = wind_data.sum().sum()
    if total_count == 0:
        return None
    wind_percentages = wind_data / total_count * 100

    # Create wind rose
    fig = go.Figure()

    # RdYlBu (Blue through yellow to red, very distinct)
    import plotly.express as px
    colors = px.colors.diverging.RdYlBu[::-1][: len(speed_labels)]

    for i, speed_cat in enumerate(speed_labels):
        fig.add_trace(
            go.Barpolar(
                r=wind_percentages[speed_cat],
                theta=dir_labels,
                name=f"{speed_cat} mph",
                marker_color=colors[i],
                marker_line_width=1,
                opacity=0.8,
                hovertemplate="Direction: %{theta}<br>"
                + "Speed: "
                + speed_cat
                + " mph<br>"
                + "Percentage: %{r:.1f}%<extra></extra>",
            )
        )

    fig.update_layout(
        title={
            "text": "Wind Rose Diagram",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        font_size=12,
        legend_font_size=10,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, wind_percentages.max().max()],
                ticksuffix="%",
                tickmode="array",
                tickvals=np.arange(0, wind_percentages.max().max(), 5),
                ticktext=[
                    f"{i}%" for i in range(0, int(wind_percentages.max().max()), 5)
                ],
            ),
            angularaxis=dict(direction="clockwise", rotation=90),
            bgcolor="rgba(0,0,0,0)",  # Set polar area background to transparent
        ),
        width=620,
        height=620,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5
        ),
    )

    return fig


def build_fig_l(df, station_name):
    """Wind Speed by Hour and Day - Heatmap."""
    df = _prepare_advanced_figure_df(df)
    wind_col = next((c for c in ["wind_speed_ms", "windspd", "windspd_ms"] if c in df.columns), None)
    if not wind_col:
        return placeholder_figure("Wind data not available in source EPW file."), ""
    wind = pd.to_numeric(df[wind_col], errors="coerce")
    if wind.dropna().empty:
        return placeholder_figure("Wind speed data not available in source EPW file."), ""
    mat = advance_day_hour_matrix(df, wind)
    if mat.empty:
        return placeholder_figure("Not enough wind data to build heatmap."), ""

        # Ensure pivot aligns with our 365-day date range
    if mat.shape[1] > 365:
        mat_plot = mat.iloc[:, :365]
    else:
        mat_plot = mat.reindex(columns=range(1, 366))
    
    dates_2021 = pd.date_range(start="2021-01-01", periods=365, freq="D")
    month_starts = pd.date_range(start="2021-01-01", end="2021-12-01", freq="MS")

    fig = go.Figure(data=go.Heatmap(
        z=mat_plot.values,
        x=dates_2021,
        y=mat_plot.index,
        colorscale="Blues",
        colorbar=dict(title="m/s"),
        hovertemplate="Date: %{x|%b %d}<br>Hour: %{y}:00<br>Wind Speed: %{z:.2f} m/s<extra></extra>",
    ))
    fig.update_layout(
        title="Wind Speed by Hour and Day",
        xaxis=dict(
            title="Calendar day",
            tickmode="array",
            tickvals=month_starts,
            ticktext=[d.strftime("%b") for d in month_starts],
            showgrid=False,
        ),
        yaxis_title="Hour of Day",
        yaxis=dict(
            tickmode="array",
            tickvals=[0, 6, 12, 18, 23],
            ticktext=["12AM", "6AM", "12PM", "6PM", "11PM"],
            autorange="reversed"
        ),
        height=400,
        margin=dict(l=55, r=40, t=40, b=55),
    )

    if not isinstance(df.index, pd.DatetimeIndex):
        cap = safe_format_caption(
            "Wind speed is mapped by day of year and hour of day, revealing seasonal and diurnal patterns in site wind exposure relevant to natural ventilation scheduling, outdoor comfort, and wind-driven infiltration. For {station}, detailed month and hour statistics are unavailable because the source index is not datetime-based.",
            dict(station=station_name),
        )
        return fig, cap

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_wind = wind.groupby(df.index.month).mean().dropna()
    hourly_wind = wind.groupby(df.index.hour).mean().dropna()
    spread = monthly_wind.std()
    spread = 0.0 if pd.isna(spread) else float(spread)
    high_wind_months = ", ".join(
        month_names[int(m) - 1]
        for m in monthly_wind.index
        if monthly_wind.loc[m] > monthly_wind.mean() + 0.5 * spread
    ) or "none"
    peak_wind_hour = int(hourly_wind.idxmax()) if not hourly_wind.empty else "[unavailable]"
    low_wind_period = ", ".join(
        month_names[int(m) - 1]
        for m in monthly_wind.index
        if monthly_wind.loc[m] < monthly_wind.quantile(0.25)
    ) or "none"
    cap = safe_format_caption(
        "Wind speed is mapped by day of year and hour of day, revealing seasonal and diurnal patterns in site wind exposure relevant to natural ventilation scheduling, outdoor comfort, and wind-driven infiltration. For {station}, mean wind is highest in {high_wind_months} with a diurnal peak near {peak_wind_hour}:00 local time, while {low_wind_period} represents the calmest sustained window - ideal for heat-flush night ventilation scheduling.",
        dict(station=station_name, high_wind_months=high_wind_months, peak_wind_hour=peak_wind_hour, low_wind_period=low_wind_period),
    )
    return fig, cap


def build_fig_m(df, station_name):
    """Seasonal Wind Roses - 2x2 subplot."""
    df = _prepare_advanced_figure_df(df)
    wind_col = next((c for c in ["wind_speed_ms", "windspd", "windspd_ms"] if c in df.columns), None)
    wdir_col = next((c for c in ["wind_dir_deg", "winddir", "winddir_deg"] if c in df.columns), None)
    if not wind_col or not wdir_col:
        return placeholder_figure("Wind data not available for seasonal wind roses."), ""
    df = _clean_wind_frame(df, wind_col, wdir_col)
    if df.empty:
        return placeholder_figure("Wind speed data not available for seasonal wind roses."), ""
    if not isinstance(df.index, pd.DatetimeIndex):
        return placeholder_figure("Wind data not available for seasonal wind roses."), ""

    seasons = [
        ("Winter (DJF)", [12, 1, 2], 1, 1),
        ("Spring (MAM)", [3, 4, 5], 1, 2),
        ("Summer (JJA)", [6, 7, 8], 2, 1),
        ("Autumn (SON)", [9, 10, 11], 2, 2),
    ]
    beaufort_bins = [0, 1.6, 3.4, 5.5, 8.0, 10.8, np.inf]
    beaufort_labels = ["Calm <1.6", "Light 1.6-3.4", "Breeze 3.4-5.5", "Fresh 5.5-8.0", "Strong 8.0-10.8", "Storm >10.8"]
    beaufort_colors = ["#aec7e8", "#1f77b4", "#ffbb78", "#ff7f0e", "#d62728", "#7f0000"]
    dir_labels = WIND_DIRECTION_LABELS

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "polar"}] * 2] * 2,
        subplot_titles=[name for name, *_ in seasons],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    prevailing = {}
    for sname, months, row, col in seasons:
        mask = df.index.month.isin(months)
        spd = pd.to_numeric(df.loc[mask, wind_col], errors="coerce").to_numpy(float)
        dirs = pd.to_numeric(df.loc[mask, wdir_col], errors="coerce").to_numpy(float) % 360
        valid = ~(np.isnan(spd) | np.isnan(dirs))
        spd, dirs = spd[valid], dirs[valid]
        if len(spd) == 0:
            continue
        dir_cats = np.floor(((dirs + 11.25) % 360) / 22.5).astype(int)
        spd_cats = pd.cut(spd, bins=beaufort_bins, labels=False, include_lowest=True)
        total = len(spd)
        for bi, (blabel, bcolor) in enumerate(zip(beaufort_labels, beaufort_colors)):
            freqs = [float(((dir_cats == di) & (spd_cats == bi)).sum()) / total * 100 for di in range(16)]
            fig.add_trace(
                go.Barpolar(r=freqs, theta=dir_labels, name=blabel, marker_color=bcolor, showlegend=(row == 1 and col == 1)),
                row=row,
                col=col,
            )
        counts = np.bincount(dir_cats, minlength=16)
        prevailing[sname] = dir_labels[int(counts.argmax())]

    polar_updates = {}
    for i in range(1, 5):
        key = "polar" if i == 1 else f"polar{i}"
        polar_updates[key] = dict(
            radialaxis=dict(tickfont=dict(size=7)),
            angularaxis=dict(direction="clockwise", rotation=90),
        )

    fig.update_layout(
        height=800,
        margin=dict(l=50, r=50, t=100, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
        **polar_updates,
    )
    # Push subplot titles above polar plots
    fig.update_annotations(font_size=12, yshift=10)
    winter_prevailing = prevailing.get("Winter (DJF)", "[unavailable]")
    summer_prevailing = prevailing.get("Summer (JJA)", "[unavailable]")
    cap = safe_format_caption(
        "Seasonal wind roses show the frequency and Beaufort-scale speed distribution of wind direction for each season, revealing how prevailing winds shift through the year. For {station}, the dominant wind direction shifts from {winter_prevailing} in winter to {summer_prevailing} in summer - a seasonal rotation that should inform cross-ventilation axis selection, windbreak placement, and winter wind exposure on the building envelope.",
        dict(station=station_name, winter_prevailing=winter_prevailing, summer_prevailing=summer_prevailing),
    )
    return fig, cap


def build_fig_n(df, station_name):
    """Diurnal Wind Roses - 4x4 polar matrix."""
    df = _prepare_advanced_figure_df(df)
    wind_col = next((c for c in ["wind_speed_ms", "windspd", "windspd_ms"] if c in df.columns), None)
    wdir_col = next((c for c in ["wind_dir_deg", "winddir", "winddir_deg"] if c in df.columns), None)
    if not wind_col or not wdir_col:
        return placeholder_figure("Wind data not available for diurnal wind roses."), ""
    df = _clean_wind_frame(df, wind_col, wdir_col)
    if df.empty:
        return placeholder_figure("Wind speed data not available for diurnal wind roses."), ""
    if not isinstance(df.index, pd.DatetimeIndex):
        return placeholder_figure("Wind data not available for diurnal wind roses."), ""

    seasons = [("Winter", [12, 1, 2]), ("Spring", [3, 4, 5]), ("Summer", [6, 7, 8]), ("Autumn", [9, 10, 11])]
    tod_slots = [
        ("Night", list(range(20, 24)) + list(range(0, 6))),
        ("Morning", list(range(6, 12))),
        ("Afternoon", list(range(12, 18))),
        ("Evening", [18, 19]),
    ]
    dir_labels = WIND_DIRECTION_LABELS
    fig = make_subplots(
        rows=4,
        cols=4,
        specs=[[{"type": "polar"}] * 4] * 4,
        subplot_titles=[f"{sname} · {tname}" for sname, _ in seasons for tname, _ in tod_slots],
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    for ri, (sname, smonths) in enumerate(seasons):
        for ci, (tname, thours) in enumerate(tod_slots):
            mask = df.index.month.isin(smonths) & df.index.hour.isin(thours)
            if not mask.any():
                continue
            dirs = pd.to_numeric(df.loc[mask, wdir_col], errors="coerce").dropna().to_numpy(float) % 360
            if len(dirs) == 0:
                continue
            dir_cats = np.floor(((dirs + 11.25) % 360) / 22.5).astype(int)
            counts = np.bincount(dir_cats, minlength=16)
            if counts.sum() == 0:
                continue
            freqs = counts / counts.sum() * 100
            fig.add_trace(
                go.Barpolar(r=freqs, theta=dir_labels, marker_color="steelblue", showlegend=False),
                row=ri + 1,
                col=ci + 1,
            )

    polar_updates = {}
    for i in range(1, 17):
        key = "polar" if i == 1 else f"polar{i}"
        polar_updates[key] = dict(
            radialaxis=dict(showticklabels=False, ticks=""),
            angularaxis=dict(direction="clockwise", rotation=90, tickfont=dict(size=8)),
        )
    fig.update_layout(width=1400, height=1600, margin=dict(l=60, r=60, t=120, b=60), **polar_updates)
    fig.update_annotations(font_size=8, yshift=8)  # Shrink font + push above polar plots
    cap = safe_format_caption(
        "The diurnal wind rose matrix shows how prevailing wind direction and speed vary across 16 sub-periods defined by season and time of day, providing the most granular wind characterisation available from the hourly dataset. For {station}, this matrix identifies the periods of most consistent directionality for cross-ventilation axis design and outdoor comfort planning.",
        dict(station=station_name),
    )
    return fig, cap


def build_fig_o(df, station_name):
    """Annual Directional Wind Power - 4-panel."""
    df = _prepare_advanced_figure_df(df)
    wind_col = next((c for c in ["wind_speed_ms", "windspd", "windspd_ms"] if c in df.columns), None)
    wdir_col = next((c for c in ["wind_dir_deg", "winddir", "winddir_deg"] if c in df.columns), None)
    if not wind_col or not wdir_col:
        return placeholder_figure("Wind data not available for directional wind power."), ""
    df = _clean_wind_frame(df, wind_col, wdir_col)
    if df.empty:
        return placeholder_figure("Wind data not available for directional wind power."), ""
    speed = pd.to_numeric(df[wind_col], errors="coerce")
    direction = pd.to_numeric(df[wdir_col], errors="coerce") % 360
    valid = speed.notna() & direction.notna()
    if not valid.any():
        return placeholder_figure("Wind data not available for directional wind power."), ""

    dir_labels = WIND_DIRECTION_LABELS
    v50 = speed * (50 / 10) ** 0.143
    power = 0.5 * 1.225 * np.power(v50, 3)
    dir_cats = pd.Series(np.floor(((direction[valid] + 11.25) % 360) / 22.5).astype(int), index=direction[valid].index)
    total = int(valid.sum())
    mean_speed = []
    power_density = []
    frequency = []
    energy_density = []
    for di in range(16):
        mask = valid & (dir_cats.reindex(df.index) == di)
        count = int(mask.sum())
        mean_speed.append(float(v50[mask].mean()) if count else 0)
        power_density.append(float(power[mask].mean()) if count else 0)
        frequency.append(count / total * 100 if total else 0)
        energy_density.append(float(power[mask].sum()) / total if count and total else 0)

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "polar"}] * 2] * 2,
        subplot_titles=[
            "Mean Wind Speed (m/s) @ 50m",
            "Wind Power Density (W/m\u00B2)",
            "Directional Frequency (%)",
            "Wind Energy Density (Wh/m\u00B2)",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    for row, col, vals, color in [
        (1, 1, mean_speed, "#1f77b4"),
        (1, 2, power_density, "#ff7f0e"),
        (2, 1, frequency, "#2ca02c"),
        (2, 2, energy_density, "#9467bd"),
    ]:
        fig.add_trace(go.Barpolar(r=vals, theta=dir_labels, marker_color=color, showlegend=False), row=row, col=col)
    polar_updates = {}
    for i in range(1, 5):
        key = "polar" if i == 1 else f"polar{i}"
        polar_updates[key] = dict(
            radialaxis=dict(tickfont=dict(size=7)),
            angularaxis=dict(direction="clockwise", rotation=90),
        )

    fig.update_layout(
        height=800,
        margin=dict(l=50, r=50, t=90, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
        **polar_updates,
    )
    # Push subplot titles above polar plots
    fig.update_annotations(font_size=12, yshift=6)

    best_dir = dir_labels[int(np.argmax(power_density))]
    peak_pd = round(max(power_density), 1)
    cap = safe_format_caption(
        "The four-panel directional wind power chart decomposes the annual wind resource by direction into mean speed, power density, directional frequency, and energy density at 50m elevation using a wind shear exponent of 0.143. For {station}, the {dominant_power_direction} direction carries the highest power density at {peak_power_density} W/m2 - the primary axis for wind energy siting and the critical exposure direction for building envelope structural wind-load design.",
        dict(station=station_name, dominant_power_direction=best_dir, peak_power_density=peak_pd),
    )
    return fig, cap


def _render_advanced_wind_diagnostics(cdf: Optional[pd.DataFrame]) -> None:
    st.markdown("---")
    st.markdown("#### Advanced Wind Diagnostics")
    st.caption("Figures generated for publication-standard reporting.")
    if cdf is None or cdf.empty:
        return
    df_cwec = _prepare_advanced_figure_df(cdf)
    station_name = _safe_location_label(st.session_state.get("header", {}))
    for key, fig_builder in [
        ("Monthly Wind Speed", _monthly_wind_speed_dashboard_fig),
        ("Wind Speed Frequency Distribution", _wind_speed_frequency_dashboard_fig),
    ]:
        fig = fig_builder(cdf)
        _st_plotly_chart(fig, use_container_width=True)
        _add_manual_pdf_figure(key, fig)

    for key, builder in [
        ("wind_speed_heatmap", build_fig_l),
        ("seasonal_wind_roses", build_fig_m),
        ("diurnal_wind_roses", build_fig_n),
        ("directional_wind_power", build_fig_o),
    ]:
        fig, cap = builder(df_cwec, station_name)
        _st_plotly_chart(fig, use_container_width=True)
        if cap:
            st.caption(cap)
            _add_manual_pdf_caption(key, cap)
        _add_manual_pdf_figure(key, fig)


def render_wind_page():
    cdf = st.session_state.get("cdf")
    if cdf is None or cdf.empty:
        st.info("No weather data available to construct the Wind dashboard.")
        return

    location_label = get_clean_city_name()
    st.markdown(f"<h3>{location_label} – Wind Analysis</h3>", unsafe_allow_html=True)
    st.caption("Understand prevalent wind patterns, magnitude, and directional distribution across the selected period.")
    
    wind_spd_col = get_metric_column(cdf, WIND_SPEED_ALIASES)
    wind_dir_col = get_metric_column(cdf, ["wind_direction", "winddir", "wd", "wdir", "wind_dir", "HourlyWindDirection"])
    if not wind_spd_col or not wind_dir_col:
        st.warning(f"This EPW file is missing required wind columns (Speed or Direction). Cannot generate Wind Rose.")
        _render_advanced_wind_diagnostics(cdf)
        return

    df_clean = _clean_wind_frame(cdf, wind_spd_col, wind_dir_col)

    # Debug: show column info to help diagnose wind data issues
    if not df_clean.empty:
        spd_vals = pd.to_numeric(df_clean[wind_spd_col], errors='coerce')
        dir_vals = pd.to_numeric(df_clean[wind_dir_col], errors='coerce')
        st.caption(f"Wind data: speed col='{wind_spd_col}' (range {spd_vals.min():.1f}–{spd_vals.max():.1f} m/s), dir col='{wind_dir_col}' (range {dir_vals.min():.0f}–{dir_vals.max():.0f}°), {len(df_clean)} records")

    if df_clean.empty:
        st.warning("All wind records are empty or invalid.")
        _render_advanced_wind_diagnostics(cdf)
        return

    fig = None
    try:
        fig = create_wind_rose(df_clean)
    except Exception as _wr_err:
        st.warning(f"Wind Rose failed to render: {_wr_err}")

    if fig is not None:
        _st_plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": True})
        st.caption(
            "Wind-rose sectors show prevailing direction and speed-class frequency. "
            "The legend keys show the speed categories (in m/s). Read the longest sectors first, "
            "then compare color distribution to understand whether wind is frequent, strong, or diffuse."
        )
        _add_manual_pdf_figure("Annual Wind Rose", fig)
        clean_loc = location_label.replace(" ", "_").replace(",", "").replace("__", "_")
        with st.expander("Export wind rose", expanded=False):
            st.caption("SVG export is prepared on demand to keep the Wind view responsive.")
            if st.button("Prepare wind downloads", key="prepare_wind_downloads"):
                try:
                    st.session_state["wind_svg_bytes"] = fig.to_image(format="svg", width=800, height=800, scale=2)
                    st.session_state["wind_svg_error"] = ""
                except Exception as e:
                    st.session_state["wind_svg_error"] = str(e)
                try:
                    st.session_state["wind_html_bytes"] = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
                    st.session_state["wind_html_error"] = ""
                except Exception as e:
                    st.session_state["wind_html_error"] = str(e)

            d1, d2 = st.columns(2)
            if st.session_state.get("wind_svg_bytes"):
                d1.download_button("Download Wind Rose (SVG)", st.session_state["wind_svg_bytes"], f"{clean_loc}_wind_rose.svg", "image/svg+xml")
            elif st.session_state.get("wind_svg_error"):
                d1.warning(f"SVG export failed: {st.session_state['wind_svg_error']}")
            if st.session_state.get("wind_html_bytes"):
                d2.download_button("Download Wind Rose (HTML)", st.session_state["wind_html_bytes"], f"{clean_loc}_wind_rose.html", "text/html")
            elif st.session_state.get("wind_html_error"):
                d2.warning(f"HTML export failed: {st.session_state['wind_html_error']}")
    else:
        st.warning("Not enough valid points to construct a Wind Rose.")
        
    _render_advanced_wind_diagnostics(cdf)


def render_live_data_page():
    page = st.session_state.get("nav_page")
    cdf = st.session_state.get("cdf")
    # Live Data might not strictly need cdf if just showing sensors? 
    # But it accesses epw_df from cdf mainly via state or passed args?
    # Original code accessed `epw_df` which... wait.
    # Where does `epw_df` come from? 
    # It must be defined global or from cdf?
    # Let's check context. 
    # Lines 6361: `if epw_df is None...`
    # I need to ensure epw_df is available. 
    # It is likely `cdf` or a derived df.
    # I'll check if I need to define epw_df = cdf.
    epw_df = cdf 
    


    # Brute-force distinguishable styles for multi-sensor plots
    SENSOR_STYLES = [
        {"color": "#f97316", "dash": "solid"},    # vivid orange
        {"color": "#22c55e", "dash": "dot"},      # green dotted
        {"color": "#3b82f6", "dash": "dash"},     # blue dashed
        {"color": "#e11d48", "dash": "dashdot"},  # magenta dash-dot
        {"color": "#a855f7", "dash": "longdash"}, # violet long dash
        {"color": "#facc15", "dash": "solid"},    # yellow solid
        {"color": "#06b6d4", "dash": "dot"},      # cyan dotted
        {"color": "#ef4444", "dash": "dash"},     # red dashed
    ]

    # ---------- State + helpers ----------
    focus_threshold = float(st.session_state.get("custom_overheat_threshold", 30))
    st.session_state.setdefault("sensor_df", pd.DataFrame())
    st.session_state.setdefault("sensor_history", [])
    st.session_state.setdefault("sensors", {})  # sensor_id -> dataframe
    st.session_state.setdefault("sensor_meta", {})  # sensor_id -> metadata dict
    st.session_state.setdefault("active_sensor_id", None)

    def render_ingested_sensors_panel():
        """Reusable panel for the Ingested Sensors table and selection."""
        sensors_store = st.session_state.get("sensors", {})
        sensor_meta = st.session_state.get("sensor_meta", {})
        
        # Consistent card styling
        with st.container(border=True):
            st.markdown("#### Ingested sensors")
            st.caption("Overview of all data currently loaded into memory.")
            
            if sensors_store:
                rows = []
                for sid, df_val in sensors_store.items():
                    meta = sensor_meta.get(sid, {})
                    rows.append({
                        "sensor_id": sid,
                        "label": meta.get("label", sid),
                        "source": meta.get("source", ""),
                        "records": meta.get("records", len(df_val)),
                        "ingested_at": meta.get("ingested_at"),
                        "date_min": meta.get("date_min"),
                        "date_max": meta.get("date_max"),
                    })

                table_df = pd.DataFrame(rows)
                table_df = table_df.sort_values("ingested_at", ascending=False, na_position="last")
                
                st.dataframe(
                    table_df,
                    use_container_width=True,
                    height=220,
                    column_config={
                        "sensor_id": st.column_config.TextColumn("ID", width="small"),
                        "label": st.column_config.TextColumn("Label", width="medium"),
                        "records": st.column_config.NumberColumn("Records", format="%d"),
                        "date_min": st.column_config.DatetimeColumn("Start", format="D MMM, HH:mm"),
                        "date_max": st.column_config.DatetimeColumn("End", format="D MMM, HH:mm"),
                    }, 
                    hide_index=True
                )

                st.divider()

                c_sel, _ = st.columns([1, 1])
                with c_sel:
                    st.markdown("**Active analysis target**")
                    chosen = st.selectbox(
                        "Select sensor",
                        options=list(sensors_store.keys()),
                        index=list(sensors_store.keys()).index(st.session_state.get("active_sensor_id")) if st.session_state.get("active_sensor_id") in sensors_store else 0,
                        label_visibility="collapsed",
                        key="active_sensor_picker_panel_unique"
                    )
                    st.session_state["active_sensor_id"] = chosen
                    st.caption(f"Currently analyzing: `{chosen}` vs EPW.")
            else:
                st.info("No ingests yet. Upload a file or fetch from an API to get started.")



    def _calc_abs_humidity(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
        """Compute absolute humidity (g/m³) from temperature (°C) and RH (%)."""
        temp_c = pd.to_numeric(temp_c, errors="coerce")
        rh_pct = pd.to_numeric(rh_pct, errors="coerce")
        es = 610.94 * np.exp(17.625 * temp_c / (temp_c + 243.04))
        vap = es * (rh_pct / 100.0)
        return 216.7 * vap / (temp_c + 273.15)

    def _normalize_sensor_columns(df: pd.DataFrame, tz_assumed) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["timestamp", "T_db", "RH"])
        cols = {c.lower(): c for c in df.columns}
        rename_map = {}
        # Timestamp aliases
        if "timestamp" in cols:
            rename_map[cols["timestamp"]] = "timestamp"
        else:
            for key, orig in cols.items():
                if "date" in key or "time" in key or "datetime" in key:
                    rename_map[orig] = "timestamp"
                    break

        # Temperature aliases
        temp_aliases = [
            "t_db", "dry_bulb", "ta", "temp", "temperature", "tair", "t",
            "air_temperature", "ambient_temperature", "drybulb"
        ]
        for alias in temp_aliases:
            if alias in cols:
                rename_map[cols[alias]] = "T_db"
                break

        # Humidity aliases
        rh_aliases = ["rh", "relative_humidity", "humidity", "rh_percent", "rh_%"]
        for alias in rh_aliases:
            if alias in cols:
                rename_map[cols[alias]] = "RH"
                break

        ghi_aliases = [
            "ghi", "glohorrad", "solar", "solar_radiation", "global_horizontal_irradiance",
            "global_horizontal", "solar_wm2", "irradiance"
        ]
        for alias in ghi_aliases:
            if alias in cols:
                rename_map[cols[alias]] = "GHI"
                break

        windspd_aliases = ["windspd", "wind_speed", "windspeed", "wind", "ws"]
        for alias in windspd_aliases:
            if alias in cols:
                rename_map[cols[alias]] = "windspd"
                break

        winddir_aliases = ["winddir", "wind_dir", "wdir", "wd", "winddirection", "wind_direction"]
        for alias in winddir_aliases:
            if alias in cols:
                rename_map[cols[alias]] = "winddir"
                break

        # Fallback: pick first likely temperature column if none mapped
        if "T_db" not in rename_map.values():
            for key, orig in cols.items():
                if "temp" in key or key in {"tair", "t"}:
                    rename_map[orig] = "T_db"
                    break

        # Fallback: pick first likely humidity column if none mapped
        if "RH" not in rename_map.values():
            for key, orig in cols.items():
                if "hum" in key or key.startswith("rh"):
                    rename_map[orig] = "RH"
                    break
        df = df.rename(columns=rename_map)
        # Wider column inference for temperature / humidity
        if "timestamp" not in df.columns:
            df = df.assign(timestamp=df.index)
        ts_raw = df.get("timestamp")
        ts = pd.to_datetime(ts_raw, errors="coerce")
        # Fallbacks: Excel serials and fixed format mm/dd/YYYY HH:MM:SS
        if ts.isna().all() and pd.api.types.is_numeric_dtype(ts_raw):
            ts = pd.to_datetime(ts_raw, unit="d", origin="1899-12-30", errors="coerce")
        if ts.isna().all():
            ts = pd.to_datetime(ts_raw.astype(str), format="%m/%d/%Y %H:%M:%S", errors="coerce")

        if ts.dt.tz is None:
            try:
                ts = ts.dt.tz_localize(tz_assumed)
            except Exception:
                ts = ts.dt.tz_localize("US/Eastern")
        else:
            ts = ts.dt.tz_convert(tz_assumed)
        out = df.assign(timestamp=ts)
        # ensure float cols
        for col in ["T_db", "RH", "GHI", "windspd", "winddir"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        if "abs_hum" not in out.columns and {"T_db", "RH"}.issubset(out.columns):
            out["abs_hum"] = _calc_abs_humidity(out["T_db"], out["RH"])
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
        return out

    from datetime import timezone as dt_timezone, timedelta as dt_timedelta

    header = st.session_state.get("header") if isinstance(st.session_state.get("header"), dict) else {}
    loc = header.get("location", {}) if isinstance(header, dict) else {}
    try:
        tz_hours = float(loc.get("timezone") or 0.0)
    except Exception:
        tz_hours = 0.0
    tzinfo = dt_timezone(dt_timedelta(hours=tz_hours))

    def _append_history(label: str, source: str, count: int, sensor_id: Optional[str] = None, date_min=None, date_max=None):
        hist = st.session_state.get("sensor_history", [])
        ts_str = pd.Timestamp.now(tz=tzinfo or "US/Eastern").strftime("%Y-%m-%d %H:%M")
        hist.append({
            "ingested_at": ts_str,
            "source": source,
            "records": int(count),
            "label": label,
            "sensor_id": sensor_id,
            "date_min": date_min,
            "date_max": date_max,
        })
        st.session_state["sensor_history"] = hist

    def _fetch_live_api_df(url: str) -> pd.DataFrame:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return pd.DataFrame(data)
        except Exception as exc:
            st.error(f"API fetch failed: {exc}")
            return pd.DataFrame()

    # ---------- Data ingest ----------
    st.markdown("#### Data input & coverage")
    left, right = st.columns(2)

    # Upload path
    with left:
        st.write("**Upload CSV/XLSX**")
        uploaded_files = st.file_uploader(
            "Sensor file(s)",
            type=["csv", "xlsx"],
            accept_multiple_files=True,
            key="live_sensor_file",
            help="Upload one or more sensor files (max 200MB each)",
        )
        ingest_click = st.button("Ingest Uploaded Data", use_container_width=True)
        if ingest_click:
            if not uploaded_files:
                st.warning("Attach at least one file before ingesting.")
            else:
                ingested = []
                active_set = False
                for uploaded_file in uploaded_files:
                    try:
                        with st.spinner(f"Reading {uploaded_file.name}…"):
                            if uploaded_file.name.lower().endswith(".xlsx"):
                                try:
                                    raw = pd.read_excel(uploaded_file, engine="openpyxl")
                                except ImportError:
                                    st.error("Excel ingest requires the 'openpyxl' package. Install it and try again.")
                                    raw = pd.DataFrame()
                            else:
                                import io, csv
                                if hasattr(uploaded_file, "seek"): uploaded_file.seek(0)
                                content = uploaded_file.read()
                                if isinstance(content, bytes): content = content.decode("utf-8", errors="replace")
                                try:
                                    raw = pd.read_csv(io.StringIO(content))
                                except Exception as e:
                                    if "saw" in str(e).lower() and "expected" in str(e).lower():
                                        lines = content.splitlines()
                                        max_cols = 0
                                        best_skip = 0
                                        for i, line in enumerate(lines[:50]):
                                            if not line.strip(): continue
                                            try:
                                                cols = len(next(csv.reader([line])))
                                                if cols > max_cols:
                                                    max_cols = cols
                                                    best_skip = i
                                            except Exception: pass
                                        raw = pd.read_csv(io.StringIO(content), skiprows=best_skip)
                                    else:
                                        raise
                            norm = _normalize_sensor_columns(raw, tzinfo or "US/Eastern")
                        if norm.empty:
                            st.warning(f"No rows after parsing: {uploaded_file.name}")
                            continue
                        norm = norm.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
                        base_id = Path(uploaded_file.name).stem or "sensor_upload"
                        sensor_id = base_id
                        suffix = 1
                        while sensor_id in st.session_state["sensors"]:
                            sensor_id = f"{base_id}_{suffix}"
                            suffix += 1

                        st.session_state["sensors"][sensor_id] = norm
                        if not active_set:
                            st.session_state["sensor_df"] = norm
                            st.session_state["active_sensor_id"] = sensor_id
                            active_set = True
                        ingested_at = pd.Timestamp.now(tz=tzinfo or "US/Eastern").strftime("%Y-%m-%d %H:%M")
                        st.session_state.setdefault("sensor_meta", {})[sensor_id] = {
                            "label": uploaded_file.name,
                            "source": "upload",
                            "records": len(norm),
                            "date_min": norm["timestamp"].min(),
                            "date_max": norm["timestamp"].max(),
                            "ingested_at": ingested_at,
                        }
                        _append_history(uploaded_file.name, "Upload", len(norm), sensor_id=sensor_id, date_min=norm["timestamp"].min(), date_max=norm["timestamp"].max())
                        ingested.append({"file": uploaded_file.name, "sensor_id": sensor_id, "records": len(norm)})
                    except Exception as exc:
                        st.error(f"Failed to ingest {uploaded_file.name}: {exc}")
                if ingested:
                    summary_df = pd.DataFrame(ingested)
                    st.success(f"Ingested {len(ingested)} file(s).")
                    st.dataframe(summary_df, use_container_width=True, height=200)

    # API path
    with right:
        st.write("**Fetch from API**")
        api_url = st.text_input("API endpoint", value="", placeholder="https://.../sensors")
        fetch_click = st.button("Fetch Live Data", use_container_width=True)
        if fetch_click:
            if not api_url:
                st.warning("Enter an API URL to fetch.")
            else:
                with st.spinner("Fetching…"):
                    api_df = _fetch_live_api_df(api_url)
                if not api_df.empty:
                    norm = _normalize_sensor_columns(api_df, tzinfo or "US/Eastern")
                    if norm.empty:
                        st.warning("API returned no usable rows.")
                    else:
                        # derive sensor id from API hostname or timestamp
                        try:
                            from urllib.parse import urlparse
                            host = urlparse(api_url).netloc or "api"
                        except Exception:
                            host = "api"
                        base_id = host.replace(":", "_") or "api"
                        sensor_id = base_id
                        suffix = 1
                        while sensor_id in st.session_state["sensors"]:
                            sensor_id = f"{base_id}_{suffix}"
                            suffix += 1

                        ingested_at = pd.Timestamp.now(tz=tzinfo or "US/Eastern").strftime("%Y-%m-%d %H:%M")
                        st.session_state["sensor_df"] = norm
                        st.session_state["sensors"][sensor_id] = norm
                        st.session_state["active_sensor_id"] = sensor_id
                        st.session_state.setdefault("sensor_meta", {})[sensor_id] = {
                            "label": api_url,
                            "source": "api",
                            "records": len(norm),
                            "date_min": norm["timestamp"].min(),
                            "date_max": norm["timestamp"].max(),
                            "ingested_at": ingested_at,
                        }
                        _append_history(api_url, "API", len(norm), sensor_id=sensor_id, date_min=norm["timestamp"].min(), date_max=norm["timestamp"].max())
                        st.success(f"Fetched {len(norm):,} rows into sensor '{sensor_id}'")

    # respect active sensor selection if available
    active_sensor_id = st.session_state.get("active_sensor_id")
    sensors_store = st.session_state.get("sensors", {})
    sensor_df = sensors_store.get(active_sensor_id, st.session_state.get("sensor_df", pd.DataFrame()))
    epw_df = st.session_state.get("epw_df") or st.session_state.get("cdf")

    # ---------- Ingested sensors (unified view) ----------
    # ---------- Ingested sensors (unified view) ----------
    # Card-like container for the data table and selection
    # ---------- Ingested sensors (unified view) ----------
    render_ingested_sensors_panel()

    st.divider()

    location_label = get_clean_city_name()
    st.markdown(f"<h3>{location_label} – Local Sensors vs Climate Baseline (EPW)</h3>", unsafe_allow_html=True)
    st.caption(
        "Compare on-site sensor readings to a long-term climate baseline (EnergyPlus Weather 'typical year'). Comparisons are statistical, not timestamp-based."
    )

    # ---------- Observed Site Conditions (sensor-only) ----------
    st.markdown("#### Observed Site Conditions")
    if sensor_df.empty:
        st.info("Upload sensor data to see on-site conditions.")
    elif "T_db" not in sensor_df.columns or "timestamp" not in sensor_df.columns:
        st.info("Sensor data needs 'timestamp' and 'T_db' columns to summarize site conditions.")
    else:
        sensor = sensor_df.copy()
        sensor["hour"] = sensor["timestamp"].dt.hour
        sensor["month"] = sensor["timestamp"].dt.month_name()

        def _temp_stats(df: pd.DataFrame) -> dict:
            diurnal = df.set_index("timestamp")["T_db"].resample("D").apply(lambda s: s.max() - s.min())
            return {
                "min": df["T_db"].min(),
                "mean": df["T_db"].mean(),
                "max": df["T_db"].max(),
                "diurnal": diurnal.mean(),
                "pct_comfort": (df["T_db"].between(18, 26)).mean() * 100,
                "pct_hot30": (df["T_db"] > 30).mean() * 100,
            }

        def _rh_stats(df: pd.DataFrame) -> dict:
            if "RH" not in df.columns:
                return {}
            return {
                "min": df["RH"].min(),
                "mean": df["RH"].mean(),
                "max": df["RH"].max(),
                "pct_rh70": (df["RH"] > 70).mean() * 100,
            }

        temp_stats = _temp_stats(sensor)
        rh_stats = _rh_stats(sensor)

        # Temperature KPIs
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Min temp (°C)", f"{temp_stats['min']:.1f}")
        c2.metric("Mean temp (°C)", f"{temp_stats['mean']:.1f}")
        c3.metric("Max temp (°C)", f"{temp_stats['max']:.1f}")
        c4.metric("Diurnal range (°C)", f"{temp_stats['diurnal']:.1f}")
        c5.metric("Hours 18–26°C", f"{temp_stats['pct_comfort']:.0f} %")
        c6.metric("Hours >30°C", f"{temp_stats['pct_hot30']:.0f} %")
        st.caption("Observed temperature range and comfort share on site. Higher diurnal range signals larger day–night swings; more hours >30°C indicate local overheating risk.")

        # Humidity KPIs (if available)
        if rh_stats:
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Min RH (%)", f"{rh_stats['min']:.0f}")
            h2.metric("Mean RH (%)", f"{rh_stats['mean']:.0f}")
            h3.metric("Max RH (%)", f"{rh_stats['max']:.0f}")
            h4.metric("Hours >70% RH", f"{rh_stats['pct_rh70']:.0f} %")
            st.caption("Humidity context: prolonged hours above 70% RH can feel muggy and reduce nighttime cooling.")

        # Sensor-only bar charts
        fig_temp_bar = go.Figure([
            go.Bar(x=["Min", "Mean", "Max"], y=[temp_stats[k] for k in ["min", "mean", "max"]], marker_color="#1f78b4")
        ])
        fig_temp_bar.update_layout(title="Observed sensor temperature (entire period)", yaxis_title="Temperature (°C)", height=280, margin=dict(l=10,r=10,t=32,b=16))
        cname = get_clean_city_name().replace(" ", "_")
        _st_plotly_chart(fig_temp_bar, use_container_width=True, config={"toImageButtonOptions": {"filename": f"{cname}_sensor_temp_bar", "format": "png", "scale": 2}, "displayModeBar": True})
        st.caption("Sensor temperatures over the selected period. Shows the observed range and average conditions on site.")

        if rh_stats:
            fig_rh_bar = go.Figure([
                go.Bar(x=["Min", "Mean", "Max"], y=[rh_stats[k] for k in ["min", "mean", "max"]], marker_color="#4c78a8")
            ])
            fig_rh_bar.update_layout(title="Observed sensor humidity (entire period)", yaxis_title="Relative Humidity (%)", height=280, margin=dict(l=10,r=10,t=32,b=16))
            _st_plotly_chart(fig_rh_bar, use_container_width=True, config={"toImageButtonOptions": {"filename": f"{cname}_sensor_rh_bar", "format": "png", "scale": 2}, "displayModeBar": True})
            st.caption("Sensor humidity over the selected period.")

    st.divider()

    # ---------- Climate Baseline Comparison (distribution-based) ----------
    st.markdown("#### Climate Baseline Comparison")
    st.caption("EPW represents a long-term typical climate. Comparisons are statistical, not timestamp-based.")
    if epw_df is None or len(epw_df) == 0:
        st.info("Load a climate baseline (EPW) to see how your site differs from typical conditions.")
    elif sensor_df.empty:
        st.info("Upload sensor data to compare against the climate baseline.")
    else:
        epw_work = epw_df.copy()
        if "datetime" in epw_work.columns:
            epw_work.rename(columns={"datetime": "timestamp"}, inplace=True)
        epw_work["timestamp"] = pd.to_datetime(epw_work.get("timestamp", epw_work.index), errors="coerce")
        if epw_work["timestamp"].dt.tz is None:
            epw_work["timestamp"] = epw_work["timestamp"].dt.tz_localize(tzinfo or "US/Eastern", nonexistent="shift_forward")
        epw_work = epw_work.rename(columns={"drybulb": "T_db", "T_db": "T_db"})

        # Require temperature columns
        if "T_db" not in epw_work.columns:
            st.info("EPW data is missing 'T_db'. Reload or check the EPW file.")
        elif "T_db" not in sensor_df.columns:
            st.info("Sensor data is missing 'T_db'. Ensure the ingest file has a temperature column.")
        else:
            sensor = sensor_df.copy()
            sensor["hour"] = sensor["timestamp"].dt.hour
            sensor["month"] = sensor["timestamp"].dt.month_name()
            epw_work["hour"] = epw_work["timestamp"].dt.hour
            epw_work["month"] = epw_work["timestamp"].dt.month_name()

            mean_bias = sensor["T_db"].mean() - epw_work["T_db"].mean()
            day_bias = sensor[sensor["hour"].between(8, 18)]["T_db"].mean() - epw_work[epw_work["hour"].between(8, 18)]["T_db"].mean()
            night_bias = sensor[(sensor["hour"] >= 18) | (sensor["hour"] < 8)]["T_db"].mean() - epw_work[(epw_work["hour"] >= 18) | (epw_work["hour"] < 8)]["T_db"].mean()

            pct_hot30_sensor = (sensor["T_db"] > 30).mean() * 100
            pct_hot30_epw = (epw_work["T_db"] > 30).mean() * 100
            pct_comfort_sensor = (sensor["T_db"].between(18, 26)).mean() * 100
            pct_comfort_epw = (epw_work["T_db"].between(18, 26)).mean() * 100

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Mean bias (°C)", f"{mean_bias:+.1f}")
            m2.metric("Daytime bias (°C)", f"{day_bias:+.1f}")
            m3.metric("Nighttime bias (°C) 🌙", f"{night_bias:+.1f}")
            m4.metric("Overheating vs baseline", f"{pct_hot30_sensor - pct_hot30_epw:+.1f} pp")
            st.caption(
                "Mean bias: positive values mean the site is warmer than the long-term climate baseline. "
                "Daytime bias highlights warm work/school hours. Nighttime bias 🌙 is a key urban heat island signal. "
                "Overheating compares share of hours above 30°C versus typical climate."
            )

            st.info(f"🌙 Nighttime bias {night_bias:+.1f}°C: warmer nights reduce cooling relief and are a hallmark of urban heat islands.")

            # Hour-of-day bias line
            hour_bias = sensor.groupby("hour")["T_db"].mean() - epw_work.groupby("hour")["T_db"].mean()
            fig_hour = go.Figure()
            fig_hour.add_trace(go.Scatter(
                x=hour_bias.index, y=hour_bias.values,
                mode="lines+markers", line=dict(color="#1f78b4", width=2.4),
                name="Bias (Sensor − Climate)",
                hovertemplate="Hour %{x}:00<br>Bias %{y:+.2f}°C<extra></extra>",
            ))
            fig_hour.update_layout(
                title="Hour-of-day temperature bias (Sensor − Climate baseline)",
                xaxis_title="Hour of Day",
                yaxis_title="Bias (°C)",
                height=340,
                margin=dict(l=10, r=10, t=46, b=20),
            )
            _st_plotly_chart(fig_hour, use_container_width=True, config={"toImageButtonOptions": {"filename": f"{cname}_hourly_bias", "format": "png", "scale": 2}, "displayModeBar": True})
            st.caption("Positive values mean the site is warmer than the climate baseline at that hour; nighttime bias suggests urban heat island effects.")

            # Monthly bias bar
            month_order = pd.date_range("2000-01-01", periods=12, freq="MS").strftime("%B")
            month_bias = (sensor.groupby("month")["T_db"].mean() - epw_work.groupby("month")["T_db"].mean()).reindex(month_order)
            fig_month = go.Figure([go.Bar(x=month_bias.index, y=month_bias.values, marker_color="#4c78a8")])
            fig_month.update_layout(
                title="Monthly mean temperature bias",
                xaxis_title="Month",
                yaxis_title="Bias (°C)",
                height=340,
                margin=dict(l=10, r=10, t=46, b=36),
            )
            _st_plotly_chart(fig_month, use_container_width=True, config={"toImageButtonOptions": {"filename": f"{cname}_monthly_bias", "format": "png", "scale": 2}, "displayModeBar": True})
            st.caption("Positive = warmer than the typical climate month; Negative = cooler. Monthly bias is shown only where sensor data exists; blank months mean no observations, not zero bias.")

            # Overheating comparison
            fig_hot = go.Figure(data=[
                go.Bar(name="Sensor >30°C", x=[">30°C"], y=[pct_hot30_sensor], marker_color="#e45756"),
                go.Bar(name="Baseline >30°C", x=[">30°C"], y=[pct_hot30_epw], marker_color="#b2b2b2"),
            ])
            fig_hot.update_layout(
                title="Overheating hours compared to climate baseline",
                xaxis_title="Threshold",
                yaxis_title="Percent of hours",
                barmode="group",
                height=300,
                legend=dict(orientation="h"),
                margin=dict(l=10, r=10, t=50, b=30)
            )
            cname = get_clean_city_name().replace(" ", "_")
            _st_plotly_chart(fig_hot, use_container_width=True, config={"toImageButtonOptions": {"filename": f"{cname}_overheating_comparison", "format": "png", "scale": 2}, "displayModeBar": True})
            st.caption("Higher values mean more hours above 30°C. Overheating elevates heat stress risk.")

            # Comfort comparison
            fig_comfort = go.Figure(data=[
                go.Bar(name="Sensor 18–26°C", x=["18–26°C"], y=[pct_comfort_sensor], marker_color="#1f78b4"),
                go.Bar(name="Baseline 18–26°C", x=["18–26°C"], y=[pct_comfort_epw], marker_color="#b2b2b2"),
            ])
            fig_comfort.update_layout(
                title="Comfort band hours compared to climate baseline",
                xaxis_title="Comfort band",
                yaxis_title="Percent of hours",
                barmode="group",
                height=300,
                legend=dict(orientation="h"),
                margin=dict(l=10, r=10, t=50, b=30)
            )
            _st_plotly_chart(fig_comfort, use_container_width=True, config={"toImageButtonOptions": {"filename": f"{cname}_comfort_comparison", "format": "png", "scale": 2}, "displayModeBar": True})
            st.caption("Comfort band (18–26°C) is a typical indoor comfort target. Higher share indicates more comfortable conditions.")

    st.divider()

    # ---------- Seasonal climatology diagnostics ----------
    st.markdown("#### Seasonal climatology: Live sensors vs EPW")
    st.caption("Day-of-year climatology (mean by hour) lets you see how the on-site sensors track the long-term EPW seasonality across key variables.")

    if epw_df is None or len(epw_df) == 0:
        st.info("Load an EPW to build the climatology baseline.")
    elif sensor_df.empty:
        st.info("Upload or fetch sensor data to compare against the EPW climatology.")
    else:
        # Prepare EPW climatology with timezone alignment
        epw_src = epw_df.copy()
        if not isinstance(epw_src.index, pd.DatetimeIndex):
            if "timestamp" in epw_src.columns:
                epw_src = epw_src.set_index(pd.to_datetime(epw_src["timestamp"], errors="coerce"))
            else:
                epw_src.index = pd.to_datetime(epw_src.index, errors="coerce")
        if epw_src.index.tz is None:
            epw_src.index = epw_src.index.tz_localize(tzinfo or "US/Eastern", nonexistent="shift_forward")
        else:
            epw_src.index = epw_src.index.tz_convert(tzinfo or "US/Eastern")

        # Sensor smoothing (fixed: hourly resample + 1h centered rolling mean)
        st.markdown("##### Sensor smoothing & focus")
        show_raw_trace = st.checkbox("Show raw sensor trace (faint)", value=False)
        focus_month_only = st.checkbox("Auto-zoom to months with sensor data (calendar axis)", value=True)
        st.caption("Sensor shown as hourly resampled + 1-hour centered rolling mean to reduce high-frequency noise.")

        def _smooth_sensor(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty or "timestamp" not in df.columns:
                return df
            work = df.copy()
            work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
            work = work.dropna(subset=["timestamp"]).sort_values("timestamp")
            work = work.set_index("timestamp")
            numeric_cols = [c for c in ["T_db", "RH", "GHI", "abs_hum", "windspd", "winddir", "temperature", "relative_humidity", "ghi", "wind_speed", "wind_dir"] if c in work.columns]
            
            # Select only numeric columns to prevent TypeError during aggregation
            work_num = work.select_dtypes(include=["number"])
            resampled = work_num.resample("1h").mean()
            
            if numeric_cols:
                valid_cols = [c for c in numeric_cols if c in resampled.columns]
                if valid_cols:
                    resampled[valid_cols] = resampled[valid_cols].interpolate(method="time", limit_direction="both")
                    resampled[valid_cols] = resampled[valid_cols].rolling(window=1, center=True, min_periods=1).mean()
            return resampled.reset_index().dropna(subset=["timestamp"])

        # Sensor timezone normalization + smoothing pipeline (plotting only; raw store unchanged)
        sensor_ts = sensor_df.copy()
        sensor_ts["timestamp"] = pd.to_datetime(sensor_ts.get("timestamp", sensor_ts.index), errors="coerce")
        if sensor_ts["timestamp"].dt.tz is None:
            sensor_ts["timestamp"] = sensor_ts["timestamp"].dt.tz_localize(tzinfo or "US/Eastern", nonexistent="shift_forward")
        else:
            sensor_ts["timestamp"] = sensor_ts["timestamp"].dt.tz_convert(tzinfo or "US/Eastern")
        sensor_ts = sensor_ts.dropna(subset=["timestamp"]).sort_values("timestamp")

        sensor_smoothed = _smooth_sensor(sensor_ts)

        # Month detection and selection (auto-zoom to sensor months)
        available_months = sorted(sensor_smoothed["timestamp"].dt.month.dropna().unique().tolist()) if not sensor_smoothed.empty else []
        selected_month = None  # kept for compatibility with below helpers
        sensor_start = sensor_ts["timestamp"].min() if not sensor_ts.empty else None
        sensor_end = sensor_ts["timestamp"].max() if not sensor_ts.empty else None
        if focus_month_only and not available_months:
            st.info("No sensor months available yet to focus.")

        # Prepare climatologies (smoothed for plotting, raw kept for optional overlay)
        epw_clim = ls.build_epw_climatology(epw_src)

        # Rename columns defensively - only rename columns that exist
        rename_map = {
            "T_db": "temperature",
            "RH": "relative_humidity",
            "GHI": "ghi",
            "windspd": "wind_speed",
            "winddir": "wind_dir",
        }
        # Filter to only columns that exist
        rename_map_raw = {k: v for k, v in rename_map.items() if k in sensor_ts.columns}
        rename_map_smoothed = {k: v for k, v in rename_map.items() if k in sensor_smoothed.columns}
    
        sensor_for_clim_raw = sensor_ts.rename(columns=rename_map_raw).copy()
        sensor_for_clim_smoothed = sensor_smoothed.rename(columns=rename_map_smoothed).copy()

        sensor_clim_raw = ls.build_sensor_climatology(sensor_for_clim_raw)
        sensor_clim_smoothed = ls.build_sensor_climatology(sensor_for_clim_smoothed)

        merged_clim = ls.compare_epw_vs_sensor(epw_clim, sensor_clim_smoothed)
        if merged_clim.empty:
            st.info("Not enough overlap to build climatology curves yet.")
        else:
            merged_clim = merged_clim.sort_values(["doy", "hour"])
            merged_clim["doy_hr"] = merged_clim["doy"] + merged_clim["hour"] / 24.0
            if not sensor_clim_raw.empty:
                sensor_clim_raw = sensor_clim_raw.sort_values(["doy", "hour"])
                sensor_clim_raw["doy_hr"] = sensor_clim_raw["doy"] + sensor_clim_raw["hour"] / 24.0

            month_ticks = pd.date_range("2001-01-01", periods=12, freq="MS")
            tick_vals = [d.dayofyear for d in month_ticks]
            tick_text = [d.strftime("%b") for d in month_ticks]
            season_bands = [
                (80, 171, "Spring"),
                (172, 263, "Summer"),
                (264, 354, "Fall"),
            ]

            def _add_season_shading(fig: go.Figure):
                for x0, x1, label in season_bands:
                    fig.add_vrect(
                        x0=x0, x1=x1,
                        fillcolor="rgba(77,214,255,0.06)", line_width=0,
                        layer="below",
                    )
                fig.update_xaxes(tickmode="array", tickvals=tick_vals, ticktext=tick_text, title="Day of Year")
                return fig

            def safe_replace_year(ts: pd.Timestamp, year: int) -> pd.Timestamp:
                """Safely replace year in timestamp, handling Feb 29 in non-leap years.
                
                If ts is Feb 29 and target year is not a leap year, shift to Feb 28.
                Preserves timezone awareness.
                """
                try:
                    return ts.replace(year=year)
                except ValueError:
                    # Feb 29 in non-leap year - shift to Feb 28
                    if ts.month == 2 and ts.day == 29:
                        return ts.replace(year=year, month=2, day=28)
                    raise
            
            def _align_epw_to_range(epw_df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
                if epw_df.empty or start_ts is None or end_ts is None:
                    return pd.DataFrame(columns=epw_df.columns)
                base = epw_df.copy()
                frames = []
                years = sorted({start_ts.year, end_ts.year})
                for yr in years:
                    temp = base.copy()
                    # Use safe_replace_year to handle Feb 29 in non-leap years
                    temp.index = temp.index.map(lambda ts: safe_replace_year(ts, yr))
                    frames.append(temp)
                aligned = pd.concat(frames).sort_index()
                return aligned[(aligned.index >= start_ts.floor("h")) & (aligned.index <= end_ts.ceil("h"))]

            def _plot_calendar_overlay(epw_col: str, sensor_col: str, title: str, units: str, y_range=None, thresholds: Optional[list] = None):
                if not focus_month_only or sensor_start is None or sensor_end is None:
                    return False
                epw_aligned = _align_epw_to_range(epw_src, sensor_start, sensor_end)
                if epw_aligned.empty and sensor_smoothed.empty:
                    st.info(f"No data to plot for {title} in the sensor window.")
                    return True

                epw_source_col = {
                    "epw_temp": "drybulb",
                    "epw_rh": "relhum",
                    "epw_ghi": "glohorrad",
                    "epw_abs_hum": "abs_hum",
                    "epw_windspd": "windspd",
                    "epw_winddir": "winddir",
                }.get(epw_col, epw_col)

                sensor_source_col = {
                    "sensor_temp": "T_db",
                    "sensor_rh": "RH",
                    "sensor_ghi": "GHI",
                    "sensor_abs_hum": "abs_hum",
                    "sensor_windspd": "windspd",
                    "sensor_winddir": "winddir",
                }.get(sensor_col, sensor_col)

                epw_plot = epw_aligned[[epw_source_col]].dropna() if epw_source_col in epw_aligned else pd.DataFrame()
                sensor_plot = sensor_smoothed.set_index("timestamp")[[sensor_source_col]].dropna() if sensor_source_col in sensor_smoothed else pd.DataFrame()
                raw_plot = sensor_ts.set_index("timestamp")[[sensor_source_col]].dropna() if show_raw_trace and sensor_source_col in sensor_ts else pd.DataFrame()

                if epw_plot.empty and sensor_plot.empty and raw_plot.empty:
                    st.info(f"No data to plot for {title} in the sensor window.")
                    return True

                fig = go.Figure()
                if not epw_plot.empty:
                    fig.add_trace(go.Scatter(
                        x=epw_plot.index,
                        y=epw_plot[epw_source_col],
                        mode="lines",
                        line=dict(color="rgba(108,117,125,0.9)", width=2.2, dash="dash"),
                        name="EPW (typical year)",
                        hovertemplate="%{x|%b %d, %H:%M}<br>EPW %{y:.2f} " + units + "<extra></extra>",
                    ))

                if not sensor_plot.empty:
                    fig.add_trace(go.Scatter(
                        x=sensor_plot.index,
                        y=sensor_plot[sensor_source_col],
                        mode="lines",
                        line=dict(color="#1f78b4", width=3.4),
                        name="Sensor (hourly avg)",
                        hovertemplate="%{x|%b %d, %H:%M}<br>Sensor %{y:.2f} " + units + "<extra></extra>",
                    ))

                if not raw_plot.empty:
                    fig.add_trace(go.Scatter(
                        x=raw_plot.index,
                        y=raw_plot[sensor_source_col],
                        mode="markers",
                        marker=dict(color="rgba(31,120,180,0.25)", size=4),
                        name="Sensor raw",
                        hovertemplate="%{x|%b %d, %H:%M}<br>Raw %{y:.2f} " + units + "<extra></extra>",
                    ))

                if y_range:
                    fig.update_yaxes(range=y_range)

                thresholds = thresholds or []
                for thr, desc, color in thresholds:
                    hline_kwargs = {"y": thr, "line_dash": "dot", "line_color": color}
                    if desc:
                        hline_kwargs.update({"annotation_text": desc, "annotation_position": "top left"})
                    fig.add_hline(**hline_kwargs)

                if sensor_col == "sensor_temp" or epw_col == "epw_temp":
                    fig.add_hrect(y0=18, y1=26, fillcolor="rgba(31,120,180,0.08)", line_width=0, layer="below")

                fig.update_xaxes(title="Date", tickformat="%b %d", showgrid=True)
                fig.update_layout(
                    title=dict(text=title, x=0.01, xanchor="left", yanchor="top", pad=dict(t=6, b=6)),
                    yaxis_title=f"{title} ({units})",
                    height=460,
                    margin=dict(t=64, b=96, l=60, r=44),
                    autosize=True,
                    legend=dict(
                        orientation="h",
                        yanchor="top", y=-0.22,
                        xanchor="left", x=0.0,
                        bgcolor="rgba(255,255,255,0.0)",
                        title=None,
                    ),
                )
                cname = get_clean_city_name().replace(" ", "_")
                _st_plotly_chart(fig, use_container_width=True, config={"toImageButtonOptions": {"filename": f"{cname}_{title.replace(' ', '_')}_calendar", "format": "png", "scale": 2}, "displayModeBar": True})
                st.caption("🛈 Sensor shown as hourly resampled + 1h centered rolling mean. EPW is typical-year baseline.")
                return True

            def _plot_clim(epw_col: str, sensor_col: str, title: str, units: str, thresholds: Optional[list] = None, y_range=None):
                # Calendar-axis focus aligned to sensor months
                if focus_month_only:
                    plotted = _plot_calendar_overlay(epw_col, sensor_col, title, units, y_range, thresholds)
                    if plotted:
                        return

                present_cols = [c for c in [epw_col, sensor_col] if c in merged_clim.columns]
                if not present_cols:
                    st.info(f"Missing data for {title}. Ensure both EPW and sensor fields are present.")
                    return
                data_slice = merged_clim[present_cols]
                if data_slice.dropna(how="all").empty:
                    st.info(f"No overlapping data for {title} yet.")
                    return

                epw_series = merged_clim[epw_col].rolling(24, center=True, min_periods=1).mean() if epw_col in merged_clim else pd.Series(dtype=float)
                fig = go.Figure()
                if epw_col in merged_clim and epw_series.notna().any():
                    fig.add_trace(go.Scatter(
                        x=merged_clim["doy_hr"],
                        y=epw_series,
                        mode="lines",
                        line=dict(color="rgba(108,117,125,0.9)", width=2.2, dash="dash"),
                        name="EPW (typical year)",
                        hovertemplate="DOY %{x:.1f}<br>EPW %{y:.2f} " + units + "<extra></extra>",
                    ))

                sensor_vals = merged_clim[sensor_col] if sensor_col in merged_clim else pd.Series(dtype=float)
                if sensor_vals.notna().any():
                    fig.add_trace(go.Scatter(
                        x=merged_clim["doy_hr"],
                        y=sensor_vals,
                        mode="lines",
                        line=dict(color="#1f78b4", width=3.2),
                        name="Sensor (hourly avg)",
                        hovertemplate="DOY %{x:.1f}<br>Sensor %{y:.2f} " + units + "<extra></extra>",
                    ))

                if show_raw_trace and sensor_col in sensor_clim_raw.columns:
                    raw_slice = sensor_clim_raw.dropna(subset=[sensor_col])
                    if not raw_slice.empty:
                        fig.add_trace(go.Scatter(
                            x=raw_slice["doy_hr"],
                            y=raw_slice[sensor_col],
                            mode="lines",
                            line=dict(color="rgba(31,120,180,0.35)", width=1.2, dash="dot"),
                            name="Sensor raw (hourly)",
                            hovertemplate="DOY %{x:.1f}<br>Raw %{y:.2f} " + units + "<extra></extra>",
                        ))

                thresholds = thresholds or []
                for thr, desc, color in thresholds:
                    hline_kwargs = {"y": thr, "line_dash": "dot", "line_color": color}
                    if desc:
                        hline_kwargs.update({"annotation_text": desc, "annotation_position": "top left"})
                    fig.add_hline(**hline_kwargs)

                # Comfort band shading for temperature plots
                if sensor_col == "sensor_temp" or epw_col == "epw_temp":
                    fig.add_hrect(y0=18, y1=26, fillcolor="rgba(31,120,180,0.08)", line_width=0, layer="below")

                if y_range:
                    fig.update_yaxes(range=y_range)

                fig.update_layout(
                    title=dict(text=title, x=0.01, xanchor="left", yanchor="top", pad=dict(t=6, b=6)),
                    yaxis_title=f"{title} ({units})",
                    height=450,
                    margin=dict(t=60, b=80, l=60, r=40),
                    autosize=True,
                    legend=dict(
                        orientation="h",
                        yanchor="top", y=-0.18,
                        xanchor="left", x=0.0,
                        bgcolor="rgba(255,255,255,0.0)",
                        title=None,
                    ),
                )
                _add_season_shading(fig)
                cname = get_clean_city_name().replace(" ", "_")
                _st_plotly_chart(fig, use_container_width=True, config={"toImageButtonOptions": {"filename": f"{cname}_{title.replace(' ', '_')}_climatology", "format": "png", "scale": 2}, "displayModeBar": True})
                st.caption("🛈 Sensor shown as hourly resampled + 1h centered rolling mean. EPW is typical-year baseline.")

            st.markdown("### 🌡️ Dry-Bulb Temperature")
            _plot_clim(
                "epw_temp",
                "sensor_temp",
                "Dry-bulb temperature",
                "°C",
                thresholds=[(30, ">30°C overheating", "#e45756"), (18, "18–26°C comfort band", "#1f78b4"), (26, None, "#1f78b4")],
            )

            st.markdown("### 💧 Relative Humidity")
            _plot_clim("epw_rh", "sensor_rh", "Relative humidity", "%", y_range=[0, 100])

            st.markdown("### ☀️ Global Solar Radiation")
            _plot_clim("epw_ghi", "sensor_ghi", "Global solar radiation", "W/m²")

            st.markdown("### 🫧 Absolute Humidity")
            _plot_clim("epw_abs_hum", "sensor_abs_hum", "Absolute humidity", "g/m³")

            st.markdown("### 💨 Wind Speed")
            _plot_clim("epw_windspd", "sensor_windspd", "Wind speed", "m/s")

            st.markdown("### 🧭 Wind Direction")
            _plot_clim("epw_winddir", "sensor_winddir", "Wind direction", "deg", y_range=[0, 360])

    # ---------- Comfort & UHI snapshot ----------
    st.markdown("#### Urban Heat Island & Thermal Comfort")
    if sensor_df.empty:
        st.info("Load sensor data first. EPW baseline missing. Load a weather file in the main EPW tab to enable comparison." if epw_df is None else "Need sensor data to compute comfort and overheating metrics.")
    elif epw_df is None or len(epw_df) == 0:
        st.info("EPW baseline missing. Load a weather file in the main EPW tab to enable comparison.")
    else:
        epw_work = epw_df.copy()
        if "datetime" in epw_work.columns:
            epw_work.rename(columns={"datetime": "timestamp"}, inplace=True)
        epw_work["timestamp"] = pd.to_datetime(epw_work.get("timestamp", epw_work.index), errors="coerce")
        if epw_work["timestamp"].dt.tz is None:
            epw_work["timestamp"] = epw_work["timestamp"].dt.tz_localize(tzinfo or "US/Eastern", nonexistent="shift_forward")
        epw_work = epw_work.rename(columns={"drybulb": "T_db", "T_db": "T_db"})

        sensor_ts = sensor_df.copy()

        # Guard against missing temperature column
        if "T_db" not in sensor_ts.columns:
            st.info("Sensor data is missing 'T_db'. Ensure the ingest file has a dry-bulb/temperature column.")
        elif "T_db" not in epw_work.columns:
            st.info("EPW data is missing 'T_db'. Reload or check the EPW file.")
        else:
            date_min = min(sensor_ts["timestamp"].min(), epw_work["timestamp"].min())
            date_max = max(sensor_ts["timestamp"].max(), epw_work["timestamp"].max())
            start, end = st.date_input("Date range", value=(date_min.date(), date_max.date()))

            if isinstance(start, datetime.date) and isinstance(end, datetime.date):
                start_ts = pd.Timestamp(start, tz=tzinfo or "US/Eastern")
                end_ts = pd.Timestamp(end, tz=tzinfo or "US/Eastern") + pd.Timedelta(days=1)
                epw_slice = epw_work[(epw_work["timestamp"] >= start_ts) & (epw_work["timestamp"] < end_ts)]
                sensor_slice = sensor_ts[(sensor_ts["timestamp"] >= start_ts) & (sensor_ts["timestamp"] < end_ts)]
            else:
                epw_slice, sensor_slice = epw_work, sensor_ts

            epw_hot = {thr: int((epw_slice.set_index("timestamp")["T_db"] > thr).resample("1h").sum().sum()) for thr in [26, 28, 30]}
            sensor_hot = {thr: int((sensor_slice.set_index("timestamp")["T_db"] > thr).resample("1h").sum().sum()) for thr in [26, 28, 30]}

            delta30 = sensor_hot[30] - epw_hot[30]
            st.caption(f"As shown above, your site experienced {delta30:+d} more hours above 30 °C than the climate baseline in the selected window.")

            with st.expander("Advanced thresholds (>26°C, >28°C, >30°C)"):
                bars = []
                for thr in [26, 28, 30]:
                    bars.append(go.Bar(name=f"Sensor >{thr}°C", x=[f">{thr}°C"], y=[sensor_hot[thr]], marker_color="#e45756"))
                    bars.append(go.Bar(name=f"Baseline >{thr}°C", x=[f">{thr}°C"], y=[epw_hot[thr]], marker_color="#b2b2b2"))
                fig_hot = go.Figure(data=bars)
                fig_hot.update_layout(
                    title="Overheating Hours Compared to Climate Baseline (detailed thresholds)",
                    xaxis_title="Threshold",
                    yaxis_title="Hours above threshold",
                    barmode="group",
                    height=320,
                    legend=dict(orientation="h"),
                    margin=dict(l=10, r=10, t=40, b=20)
                )
                cname = get_clean_city_name().replace(" ", "_")
                _st_plotly_chart(fig_hot, use_container_width=True, config={"toImageButtonOptions": {"filename": f"{cname}_overheating_detailed", "format": "png", "scale": 2}, "displayModeBar": True})
                st.caption("Detailed thresholds let you inspect moderate heat (>26°C) versus high heat (>30°C) without crowding the main view.")

    st.divider()

    # ---------- Outputs & Next Steps ----------
    st.markdown("#### Outputs & Next Steps")
    st.info("Use the calibrated EPW download when sufficient coverage exists. Key insights above: mean and nighttime biases, monthly patterns, and overheating vs climate baseline.")

    calibrated_epw_bytes = st.session_state.get("calibrated_epw_bytes")
    calibrated_cdf = st.session_state.get("calibrated_cdf")
    calibrated_fields = st.session_state.get("calibrated_fields", [])

    st.markdown("#### Outputs")
    st.caption("Download a calibrated EnergyPlus Weather file once enough overlap exists to adjust the baseline.")
    if calibrated_epw_bytes is None or calibrated_cdf is None:
        st.info("Once enough bias coverage exists, you can download a calibrated EPW here.")
    else:
        field_map = {
            "drybulb": "Dry-bulb",
            "relhum": "Relative humidity",
            "glohorrad": "Global horizontal irradiance",
        }
        applied = ", ".join(field_map.get(f, f) for f in calibrated_fields)
        st.success(f"Applied DOY × hour corrections to {applied}.")

        loc_meta = (st.session_state.get("header", {}) or {}).get("location", {})
        slug_source = loc_meta.get("city") or loc_meta.get("state_province") or "site"
        safe_slug = re.sub(r"[^A-Za-z0-9_-]", "_", slug_source).strip("_") or "site"
        file_name = f"{safe_slug}_calibrated.epw"

        st.download_button(
            "Download calibrated EPW",
            data=calibrated_epw_bytes,
            file_name=file_name,
            mime="text/plain",
        )

        with st.expander("Preview calibrated columns", expanded=False):
            preview_cols = ["drybulb", "drybulb_calibrated"]
            if "relhum_calibrated" in calibrated_cdf.columns:
                preview_cols.append("relhum_calibrated")
            if "glohorrad_calibrated" in calibrated_cdf.columns:
                preview_cols.append("glohorrad_calibrated")
            st.dataframe(
                calibrated_cdf[preview_cols].tail(168),
                use_container_width=True,
                height=260,
            )

        st.caption("This EPW keeps the original metadata but swaps the corrected hourly columns, ready for EnergyPlus/Ladybug workflows.")

    st.markdown("#### What does this mean?")
    st.info(
        "Overall, the site appears warmer than the long-term climate baseline, especially at night, which aligns with urban heat island patterns. "
        "Warm nights reduce cooling relief, and higher shares of hours above 30°C elevate discomfort and heat risk. "
        "Monthly bias and hour-of-day charts above show when the differences are strongest, guiding mitigation or scheduling decisions."
    )

    st.markdown("#### Future climate roadmap")
    with st.expander("Design blueprint", expanded=False):
        st.markdown(
            "- Placement: dedicated tab 'Future Scenarios' that lives to the right of the Live Data view.\n"
            "- Controls: scenario (SSP1-2.6, SSP2-4.5, SSP5-8.5), horizon year (2050/2080), and a toggle to morph either the raw TMY or the calibrated EPW baseline.\n"
            "- Outputs: temperature shift summary (delta mean/delta extremes), future comfort/overheating metrics, and a download button for each morphed EPW.\n"
            "- Flow: user selects scenario -> app clones the calibrated EPW -> applies morphing deltas -> refreshes plots and download links."
        )

    st.markdown("#### Paper-ready notes")
    st.markdown(
        "- Data ingestion: CSV uploads and authenticated API pulls normalize fields into a local Parquet store with timezone-aware timestamps.\n"
        "- Storage & processing: sensor histories persist under data/sensors.parquet, deduplicated by timestamp + sensor ID for reproducible analyses.\n"
        "- Climatological alignment: both EPW and sensors collapse to DOY x hour means, enabling direct bias computation and smoothing of sparse data.\n"
        "- Bias metrics: dashboards report average, day/night, and RH biases plus comfort coverage and overheating hours tied to UHI framing.\n"
        "- Calibration pipeline: DOY x hour biases map back onto every EPW hour to emit a site-specific EPW download for EnergyPlus/Ladybug."
    )
    st.info("Capture screenshots of the Live Data tab, 7-day overlay, bias heatmap, and scatter plot for the Methods section.")


def render_sensor_comparison_page():
    page = st.session_state.get("nav_page")
    
    # Brute-force distinguishable styles for multi-sensor plots
    SENSOR_STYLES = [
        {"color": "#f97316", "dash": "solid"},    # vivid orange
        {"color": "#22c55e", "dash": "dot"},      # green dotted
        {"color": "#3b82f6", "dash": "dash"},     # blue dashed
        {"color": "#e11d48", "dash": "dashdot"},  # magenta dash-dot
        {"color": "#a855f7", "dash": "longdash"}, # violet long dash
        {"color": "#facc15", "dash": "solid"},    # yellow solid
        {"color": "#06b6d4", "dash": "dot"},      # cyan dotted
        {"color": "#ef4444", "dash": "dash"},     # red dashed
    ]

    # Helper to clean up verbose sensor names for the legend
    import re
    def clean_sensor_name(name):
        # Matches patterns like " 2025-12-18 00_30_58 EST (Data EST)" and removes them
        return re.sub(r'\s+\d{4}-\d{2}-\d{2}\s+\d{2}_\d{2}_\d{2}.*$', '', str(name))

    # ========== PROOF MARKER ==========
    st.success("Sensor Comparison UI loaded")
    
    # ========== HELPER FUNCTIONS ==========
    @st.cache_data
    def detect_timestamp_col(df: pd.DataFrame) -> Optional[str]:
        """Detect timestamp column name."""
        candidates = ["timestamp", "time", "datetime", "date", "Date", "Timestamp", "Time"]
        for col in candidates:
            if col in df.columns:
                return col
        # fallback: pick the first datetime-like column
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        return None
    
    @st.cache_data
    def detect_sensor_id_col(df: pd.DataFrame) -> Optional[str]:
        """Detect sensor identifier column name."""
        candidates = ["sensor_id", "device_id", "location", "sensor", "device"]
        for col in candidates:
            if col in df.columns:
                return col
        # fallback: first object/string column that's not timestamp
        for col in df.columns:
            if df[col].dtype == object:
                return col
        return None
    
    def get_numeric_metric_columns(df: pd.DataFrame, ts_col: Optional[str], sensor_col: Optional[str]) -> list[str]:
        """Get numeric columns that are NOT index-like and have sufficient variance.

        Excludes:
        - Column name exactly "#"
        - Names in {"index","idx","row","row_id"}
        - Names starting with "Unnamed"
        - Timestamp or sensor id columns
        - Numeric columns that look like sequential indices (mostly unique + monotonic + step ~1)

        Requires:
        - >= 20 non-null values
        - >= 5 unique values
        """
        exclude_names = {"#", "index", "idx", "row", "row_id"}
        exclude_cols = {ts_col, sensor_col}

        metric_cols: List[str] = []
        for col in df.columns:
            if col is None:
                continue
            if col in exclude_cols or col in exclude_names or str(col).startswith("Unnamed"):
                continue

            series = df[col]

            # Attempt to coerce to numeric where sensible
            if not pd.api.types.is_numeric_dtype(series):
                coerced = pd.to_numeric(series, errors="coerce")
                num_values = coerced.notna().sum()
                unique_count = coerced.nunique()
                numeric_fraction = num_values / max(1, len(series))
                if numeric_fraction < 0.6:
                    continue
                non_null = num_values
                uniq = unique_count
            else:
                non_null = series.notna().sum()
                uniq = series.nunique()

            if non_null < 20 or uniq < 5:
                continue

            # Detect index-like numeric series: mostly unique, monotonic and step close to 1
            try:
                numeric_series = pd.to_numeric(series.dropna(), errors="coerce").astype(float)
                if len(numeric_series) > 50:
                    # proportion of unique values
                    prop_unique = numeric_series.nunique() / len(numeric_series)
                    diffs = np.diff(np.sort(numeric_series.values))
                    if len(diffs) > 0:
                        median_step = float(np.median(diffs))
                        # fraction of diffs approximately equal to 1 (within 5%)
                        near_one = np.isclose(diffs, 1.0, rtol=0.05, atol=1e-6).sum() / len(diffs)
                    else:
                        near_one = 0.0
                        median_step = 0.0

                    # If it's mostly unique and stepping by ~1, treat as index-like
                    if prop_unique > 0.9 and (near_one > 0.8 or np.isclose(median_step, 1.0, rtol=0.1)):
                        continue
            except Exception as e:
                st.warning(f"Section failed: {e}")

            metric_cols.append(col)

        return metric_cols
    
    def choose_default_metrics(metric_cols: list[str]) -> list[str]:
        """Choose up to 4 default metrics based on priority."""
        priority = ["co2", "humidity", "temperature", "pm25", "voc", "noise"]
        chosen = []
        
        for p in priority:
            if p in metric_cols and len(chosen) < 4:
                chosen.append(p)
        
        # Fill remaining slots with other metrics
        for col in metric_cols:
            if col not in chosen and len(chosen) < 4:
                chosen.append(col)
        
        return chosen[:4]
    
    def get_unit(col_name: str) -> str:
        """Get unit string for a metric column."""
        units_map = {
            "co2": "ppm",
            "humidity": "%",
            "rh": "%",
            "dew_point": "°C",
            "temperature": "°C",
            "temp": "°C",
            "pm25": "µg/m³",
        }
        col_lower = col_name.lower()
        for key, unit in units_map.items():
            if key in col_lower:
                return unit
        return ""

    def pretty_label(col_name: str) -> str:
        """Return display label for a column (used with selectbox format_func)."""
        unit = get_unit(col_name)
        base = str(col_name).replace("_", " ")
        if unit:
            return f"{base} ({unit})"
        return base
    
    @st.cache_data
    def computed_prev_window_df(df: pd.DataFrame, col: str, sensor_ids: list[str], sensor_col_name: str, frac: float = 0.2) -> dict:
        """Return latest and previous segments for a given column and sensors.

        Cached to avoid recomputing window splits repeatedly.
        Returns a dict with keys: 'latest' and 'prev' mapping to pd.Series (concatenated across sensors).
        """
        if df.empty or col not in df.columns:
            return {"latest": pd.Series(dtype=float), "prev": pd.Series(dtype=float)}

        filtered = df[df[sensor_col_name].isin(sensor_ids)][col].dropna()
        if filtered.empty:
            return {"latest": filtered, "prev": pd.Series(dtype=float)}

        # Cap to avoid slowness
        max_rows = min(len(filtered), 5000 * max(1, len(sensor_ids)))
        filtered = filtered.tail(max_rows)

        n = len(filtered)
        latest_n = max(1, int(n * frac))
        prev_n = max(1, int(n * frac))

        latest_segment = filtered.tail(latest_n)
        prev_segment = filtered.iloc[-(latest_n + prev_n):-latest_n] if latest_n + prev_n <= n else filtered.head(prev_n)

        return {"latest": latest_segment, "prev": prev_segment}

    def compute_window_metrics(df: pd.DataFrame, col: str, sensor_ids: list[str], sensor_col_name: str) -> dict:
        """Compute max and delta vs previous segment (latest 20% vs previous 20%). Uses cached splitter."""
        segs = computed_prev_window_df(df, col, sensor_ids, sensor_col_name, frac=0.2)
        latest_segment = segs.get("latest")
        prev_segment = segs.get("prev")

        if latest_segment is None or latest_segment.empty:
            return {"max": np.nan, "delta": None}

        latest_max = latest_segment.max()
        prev_max = prev_segment.max() if prev_segment is not None and not prev_segment.empty else np.nan

        delta = latest_max - prev_max if not pd.isna(latest_max) and not pd.isna(prev_max) else None
        return {"max": latest_max, "delta": delta}

    @st.cache_data
    def filtered_df_by_window(df: pd.DataFrame, ts_col: Optional[str], days: Optional[int] = None) -> pd.DataFrame:
        """Return a copy of df optionally filtered to the last `days` days (by ts_col).

        Cached to speed repeated plotting operations.
        """
        if ts_col is None or days is None:
            return df.copy()
        try:
            end = df[ts_col].max()
            start = end - pd.Timedelta(days=days)
            return df[df[ts_col] >= start].copy()
        except Exception:
            return df.copy()
    
    # ========== DATA LOADING ==========
    st.markdown("### Live Sensor Data Comparison")
    st.caption("Compare live environmental sensor readings across locations and time windows.")
    
    sensors_store = st.session_state.get("sensors", {})
    sensor_meta = st.session_state.get("sensor_meta", {})
    
    if not sensors_store:
        st.info("No stored sensor data found. Ingest multiple sensor locations in the Live Data vs EPW tab to enable comparison.")
        st.stop()
    
    # Build combined dataframe
    frames = []
    for sid, df_val in sensors_store.items():
        df_local = df_val.copy()
        if "timestamp" not in df_local.columns:
            continue
        df_local["timestamp"] = pd.to_datetime(df_local["timestamp"], errors="coerce")
        df_local["sensor_id"] = sid
        frames.append(df_local)
    
    if not frames:
        st.info("No sensor data with timestamps found.")
        st.stop()
    
    df = pd.concat(frames, ignore_index=True)
    
    # Defensive: drop index-like columns
    df = df.drop(columns=["#"], errors="ignore")

    # Detect columns
    ts_col = detect_timestamp_col(df)
    sensor_col = detect_sensor_id_col(df) or "sensor_id"

    # Timestamp handling and normalized plotting timestamp (_ts)
    if ts_col:
        # Robust timestamp parsing: try several strategies and pick the one
        # that yields the most non-null datetimes. Keep timezone-aware UTC
        # values so downstream tz-conversion logic remains unchanged.
        raw_ts = df[ts_col]
        parse_candidates = {}

        # 1) Default (fast) parse
        try:
            parse_candidates["default"] = pd.to_datetime(raw_ts, errors="coerce", utc=True)
        except Exception:
            parse_candidates["default"] = pd.Series(pd.NaT, index=raw_ts.index)

        # 2) Day-first parse (handles DD/MM ambiguity)
        try:
            parse_candidates["dayfirst"] = pd.to_datetime(raw_ts, errors="coerce", dayfirst=True, utc=True)
        except Exception:
            parse_candidates["dayfirst"] = pd.Series(pd.NaT, index=raw_ts.index)

        # 3) A few explicit common formats
        explicit_formats = [
            "%m/%d/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M",
            "%d/%m/%Y %H:%M",
        ]
        for fmt in explicit_formats:
            try:
                parsed = pd.to_datetime(raw_ts.astype(str), format=fmt, errors="coerce")
                # ensure timezone-aware in UTC so later tz_convert works predictably
                parsed = parsed.dt.tz_localize("UTC")
                parse_candidates[fmt] = parsed
            except Exception:
                parse_candidates[fmt] = pd.Series(pd.NaT, index=raw_ts.index)

        # Choose the candidate with the most non-null parses
        best_key = max(parse_candidates.keys(), key=lambda k: parse_candidates[k].notna().sum())
        df[ts_col] = parse_candidates[best_key]
        df = df.dropna(subset=[ts_col]).copy()

        # Normalize to a single timezone for display and produce naive datetimes in _ts
        try:
            df["_ts"] = df[ts_col].dt.tz_convert("America/New_York").dt.tz_localize(None)
        except Exception:
            try:
                # If timestamps are tz-naive, localize to UTC first then convert
                df[ts_col] = df[ts_col].dt.tz_localize("UTC")
                df["_ts"] = df[ts_col].dt.tz_convert("America/New_York").dt.tz_localize(None)
            except Exception:
                # Fallback: drop tz info
                df["_ts"] = df[ts_col].dt.tz_localize(None)

        # Sort by plotting ts for charts (ascending)
        df = df.sort_values(by="_ts", ascending=True)
    else:
        # Ensure _ts exists even if no timestamp column detected
        df["_ts"] = pd.NaT
    
    # Get metric columns (initial candidates)
    metric_cols = get_numeric_metric_columns(df, ts_col, sensor_col)

    # Prune metric options that have effectively no numeric data (avoid confusing dropdowns)
    pruned_metrics = []
    min_absolute = 10
    min_fraction = 0.005
    min_fraction_count = max(min_absolute, int(min_fraction * len(df))) if len(df) > 0 else min_absolute
    for col in metric_cols:
        try:
            non_na = pd.to_numeric(df[col], errors="coerce").notna().sum()
        except Exception:
            non_na = 0
        if non_na >= min_fraction_count:
            pruned_metrics.append(col)

    metric_cols = pruned_metrics

    if not metric_cols:
        st.warning("No numeric metric columns with sufficient data were found in the data.")
        st.stop()
    
    # Sensor selection with persistence
    available_sensors = sorted(df[sensor_col].dropna().unique().tolist())
    available_sig = tuple(map(str, available_sensors))

    if (
        "selected_sensors" not in st.session_state
        or st.session_state.get("_sensor_available_sig") != available_sig
    ):
        st.session_state["selected_sensors"] = available_sensors
        st.session_state["_sensor_available_sig"] = available_sig

    # keep only sensors still present
    st.session_state["selected_sensors"] = [s for s in st.session_state["selected_sensors"] if s in available_sensors]
    if not st.session_state["selected_sensors"]:
        st.session_state["selected_sensors"] = available_sensors

    # Top Card: Compare Different Sensors
    with st.container(border=True):
        st.markdown("#### Compare Different Sensors")
        st.caption("Select sensors to compare")
        csel, cclear = st.columns([1, 1])
        with csel:
            if st.button("Select all loaded sensors", use_container_width=True):
                st.session_state["selected_sensors"] = available_sensors
                _rerun()
        with cclear:
            st.caption(f"{len(available_sensors)} loaded sensor(s)")
        st.multiselect(
            "Select sensors to compare",
            options=available_sensors,
            default=st.session_state["selected_sensors"],
            key="selected_sensors",
            help="Select at least one sensor to compare."
        )

    selected_sensors = st.session_state.get("selected_sensors", [])
    if not selected_sensors:
        st.warning("Select at least one sensor to compare.")
        st.stop()
    
    # Filter dataframe
    df_filtered = df[df[sensor_col].isin(selected_sensors)].copy()
    
    # Choose default metrics
    default_metrics = choose_default_metrics(metric_cols)
    
    # ========== SUMMARY METRIC CARDS ==========
    st.markdown("#### Current Snapshot Summary")
    
    selected_metrics = default_metrics[:4]
    
    cols = st.columns(4)
    for idx, metric in enumerate(selected_metrics):
        with cols[idx]:
            metrics_data = compute_window_metrics(df_filtered, metric, selected_sensors, sensor_col)
            unit = get_unit(metric)
            unit_str = f" {unit}" if unit else ""
            
            delta_value = metrics_data["delta"]
            delta_label = None
            if delta_value is not None and not pd.isna(delta_value):
                delta_label = f"{delta_value:+.2f}{unit_str}"
            
            st.metric(
                label=metric.replace("_", " ").title(),
                value=f"{metrics_data['max']:.2f}{unit_str}" if not pd.isna(metrics_data['max']) else "—",
                delta=delta_label
            )
    
    # ========== STACKED FULL-WIDTH LAYOUT ==========

    # Primary Metric Trend (full width)
    with st.container(border=True):
        st.markdown("#### Primary Metric Trend")

        primary_metric = st.selectbox(
            "Primary metric",
            options=[m for m in metric_cols if m not in ["#"]],
            index=0 if default_metrics and default_metrics[0] in metric_cols else 0,
            key="primary_metric",
            format_func=lambda c: pretty_label(c)
        )

        unit = get_unit(primary_metric)
        unit_str = f" ({unit})" if unit else ""

        # Use cached filtered df for plotting (no UI for time bucket)
        plot_df = filtered_df_by_window(df_filtered, "_ts", days=None)

            # Coerce metric to numeric into a safe plotting column _metric
        metric_series = pd.to_numeric(plot_df.get(primary_metric, pd.Series(dtype=float)), errors="coerce")
        plot_df = plot_df.assign(_metric=metric_series)

        # debug expander removed

        non_na = int(plot_df["_metric"].notna().sum())
        if non_na == 0:
            st.error(f"Selected primary metric '{primary_metric}' cannot be coerced to numeric — skipping plot.")
        else:
            # Downsample for plotting only when very large using _ts index
            if "_ts" in plot_df.columns and len(plot_df) > 20000:
                plot_df = plot_df.set_index("_ts")
                numeric_cols = [c for c in plot_df.columns if c != sensor_col and c != "_metric"]
                if numeric_cols:
                    plot_df[numeric_cols] = plot_df[numeric_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
                if sensor_col in plot_df.columns:
                    cols_to_agg = ["_metric"]
                    plot_df_resampled = plot_df.groupby(sensor_col)[cols_to_agg].resample('5min').mean(numeric_only=True)
                    plot_df_resampled = plot_df_resampled.reset_index()
                else:
                    plot_df_resampled = plot_df[["_metric"]].resample('5min').mean(numeric_only=True).reset_index()
                plot_df = plot_df_resampled

            fig_primary = go.Figure()
            for i, sensor in enumerate(selected_sensors):
                style = SENSOR_STYLES[i % len(SENSOR_STYLES)]
                sensor_data = plot_df[plot_df[sensor_col] == sensor]
                if not sensor_data.empty and "_metric" in sensor_data.columns and "_ts" in sensor_data.columns:
                    display_name = clean_sensor_name(sensor)
                    fig_primary.add_trace(go.Scatter(
                        x=sensor_data["_ts"],
                        y=sensor_data["_metric"],
                        mode="lines",
                        name=display_name,
                        line=dict(
                            color=style["color"],
                            width=2.3,
                            dash=style["dash"],
                        ),
                        opacity=1.0,
                        showlegend=True,
                        hovertemplate=f"{display_name}<br>%{{x}}<br>%{{y:.2f}}{unit_str}<extra></extra>"
                    ))

            fig_primary.update_layout(
                title=f"{primary_metric.replace('_', ' ').title()}{unit_str}",
                xaxis_title="Time",
                yaxis_title=f"{primary_metric.replace('_', ' ').title()}{unit_str}",
                height=420,
                hovermode="x unified",
                legend=dict(y=0.5, x=1.02, xanchor='left')
            )

            # Ensure x-axis is date type and zoom to real data range safely
            fig_primary.update_xaxes(type="date")
            try:
                # Zoom only to timestamps where the metric has values
                metric_ts = plot_df.loc[plot_df["_metric"].notna(), "_ts"]
                x_min = metric_ts.min() if not metric_ts.empty else None
                x_max = metric_ts.max() if not metric_ts.empty else None
                if pd.notna(x_min) and pd.notna(x_max) and x_min < x_max:
                    pad = (x_max - x_min) * 0.02
                    fig_primary.update_xaxes(range=[x_min - pad, x_max + pad], autorange=False)
                else:
                    fig_primary.update_xaxes(autorange=True)
            except Exception:
                fig_primary.update_xaxes(autorange=True)

            _st_plotly_chart(fig_primary, use_container_width=True)

    # Distribution (full width) — histogram for numeric metrics
    with st.container(border=True):
        st.markdown("#### Distribution")

        fig_dist = go.Figure()
        if primary_metric in df_filtered.columns:
            plot_df_dist = df_filtered.copy()
            metric_series = pd.to_numeric(plot_df_dist.get(primary_metric, pd.Series(dtype=float)), errors="coerce")
            plot_df_dist = plot_df_dist.assign(_metric=metric_series)

            numeric_count = int(plot_df_dist["_metric"].notna().sum())
            unique_numeric = int(plot_df_dist["_metric"].nunique(dropna=True))

            if numeric_count == 0:
                st.info("Metric is non-numeric; distribution histogram unavailable.")
            else:
                if unique_numeric > 20 or pd.api.types.is_numeric_dtype(plot_df_dist["_metric"]):
                    for i, sensor in enumerate(selected_sensors):
                        style = SENSOR_STYLES[i % len(SENSOR_STYLES)]
                        sensor_data = plot_df_dist[plot_df_dist[sensor_col] == sensor]["_metric"].dropna()
                        if not sensor_data.empty:
                            display_name = clean_sensor_name(sensor)
                            fig_dist.add_trace(go.Histogram(
                                x=sensor_data, 
                                name=display_name, 
                                opacity=0.75, 
                                nbinsx=40,
                                marker_color=style["color"]
                            ))
                    fig_dist.update_layout(title=f"{primary_metric.replace('_', ' ').title()} Distribution", xaxis_title=f"{primary_metric.replace('_', ' ').title()}{unit_str}", yaxis_title="Frequency", barmode="overlay", height=380)
                else:
                    st.info("Metric appears categorical/low-cardinality; distribution histogram unavailable.")

        _st_plotly_chart(fig_dist, use_container_width=True)

    # Secondary Metric Trend (full width)
    with st.container(border=True):
        st.markdown("#### Secondary Metric Trend")

        secondary_options = [m for m in metric_cols if m != primary_metric and m not in ["#"]]
        if not secondary_options:
            st.info("No secondary metric available.")
        else:
            secondary_metric = st.selectbox(
                "Secondary metric",
                options=secondary_options,
                index=0,
                key="secondary_metric",
                format_func=lambda c: pretty_label(c)
            )

            unit = get_unit(secondary_metric)
            unit_str = f" ({unit})" if unit else ""

            plot_df = filtered_df_by_window(df_filtered, "_ts", days=None)
            metric_series = pd.to_numeric(plot_df.get(secondary_metric, pd.Series(dtype=float)), errors="coerce")
            plot_df = plot_df.assign(_metric=metric_series)

            non_na = int(plot_df["_metric"].notna().sum())
            if non_na == 0:
                st.error(f"Selected secondary metric '{secondary_metric}' cannot be coerced to numeric — skipping plot.")
            else:
                if "_ts" in plot_df.columns and len(plot_df) > 20000:
                    plot_df = plot_df.set_index("_ts")
                    numeric_cols = [c for c in plot_df.columns if c != sensor_col and c != "_metric"]
                    if numeric_cols:
                        plot_df[numeric_cols] = plot_df[numeric_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
                    if sensor_col in plot_df.columns:
                        cols_to_agg = ["_metric"]
                        plot_df_resampled = plot_df.groupby(sensor_col)[cols_to_agg].resample('5min').mean(numeric_only=True)
                        plot_df_resampled = plot_df_resampled.reset_index()
                    else:
                        plot_df_resampled = plot_df[["_metric"]].resample('5min').mean(numeric_only=True).reset_index()
                    plot_df = plot_df_resampled

                fig_secondary = go.Figure()
                for i, sensor in enumerate(selected_sensors):
                    style = SENSOR_STYLES[i % len(SENSOR_STYLES)]
                    sensor_data = plot_df[plot_df[sensor_col] == sensor]
                    if not sensor_data.empty and "_metric" in sensor_data.columns and "_ts" in sensor_data.columns:
                        display_name = clean_sensor_name(sensor)
                        fig_secondary.add_trace(go.Scatter(
                            x=sensor_data["_ts"],
                            y=sensor_data["_metric"],
                            mode="lines",
                            name=display_name,
                            line=dict(
                                color=style["color"],
                                width=2.3,
                                dash=style["dash"],
                            ),
                            opacity=1.0,
                            showlegend=True,
                            hovertemplate=f"{display_name}<br>%{{x}}<br>%{{y:.2f}}{unit_str}<extra></extra>"
                        ))

                fig_secondary.update_layout(
                    title=f"{secondary_metric.replace('_', ' ').title()}{unit_str}",
                    xaxis_title="Time",
                    yaxis_title=f"{secondary_metric.replace('_', ' ').title()}{unit_str}",
                    height=420,
                    hovermode="x unified",
                    legend=dict(y=0.5, x=1.02, xanchor='left')
                )

                fig_secondary.update_xaxes(type="date")
                try:
                    # Zoom only to timestamps where the metric has values
                    metric_ts = plot_df.loc[plot_df["_metric"].notna(), "_ts"]
                    x_min = metric_ts.min() if not metric_ts.empty else None
                    x_max = metric_ts.max() if not metric_ts.empty else None
                    if pd.notna(x_min) and pd.notna(x_max) and x_min < x_max:
                        pad = (x_max - x_min) * 0.02
                        fig_secondary.update_xaxes(range=[x_min - pad, x_max + pad], autorange=False)
                    else:
                        fig_secondary.update_xaxes(autorange=True)
                except Exception:
                    fig_secondary.update_xaxes(autorange=True)

                _st_plotly_chart(fig_secondary, use_container_width=True)
    
    # Row 3: Raw Data Table (Full Width)
    
    
    # ========== RAW DATA TABLE ==========
    with st.container(border=True):
        st.markdown("#### Raw Data")
        
        # Column selection
        default_display_cols = []
        if ts_col:
            default_display_cols.append(ts_col)
        default_display_cols.append(sensor_col)
        # prefer primary/secondary if present
        if 'primary_metric' in st.session_state:
            default_display_cols.append(st.session_state['primary_metric'])
        if 'secondary_metric' in st.session_state:
            default_display_cols.append(st.session_state['secondary_metric'])

        available_display_cols = [c for c in df_filtered.columns if c not in ["#"]]

        display_cols = st.multiselect(
            "Columns to display",
            options=available_display_cols,
            default=[c for c in default_display_cols if c in available_display_cols],
            key="display_cols"
        )
        
        if display_cols:
            display_df = df_filtered[display_cols].copy()
            
            # Sort by plotting timestamp desc if available
            if "_ts" in display_df.columns:
                display_df = display_df.sort_values(by="_ts", ascending=False)
            
            render_virtualized_table(display_df, height=400, key="sensor_raw_aggrid", page_size=50)
            
            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="Download filtered CSV",
                data=csv,
                file_name="sensor_data_filtered.csv",
                mime="text/csv"
            )
        else:
            st.info("Select at least one column to display.")


def render_raw_data_page():
    page = st.session_state.get("nav_page")
    # Raw Data page relies on st.session_state.cdf which should be valid if nav is allowed
    
    # ====================== RAW DATA ======================
    # ====================== RAW DATA ======================
    st.markdown("### 📁 Raw Data & Export")
    st.caption(
        "Filter the underlying EPW table, spot-check any column, and pull exactly the rows you need "
        "before exporting. Handy when someone asks for the source numbers behind a chart."
    )
    with st.expander("Show raw EPW data (filter and export)"):
        # sensible defaults
        default_cols = [c for c in ["timestamp","drybulb","relhum","glohorrad","difhorrad","dirnorrad","windspd","winddir"]
            if (c == "timestamp") or (c in st.session_state.cdf.columns)]
        options = ["timestamp"] + [c for c in st.session_state.cdf.columns if c != "timestamp"]
        show_cols = st.multiselect(
            "Columns to show",
            options=options,
            default=default_cols
        )

        # ensure timestamp visible even if user unchecks it later
        if "timestamp" in show_cols:
            df_for_view = st.session_state.cdf.copy()
        else:
            df_for_view = st.session_state.cdf.copy()

        # date range
        tmin = pd.to_datetime(st.session_state.cdf.index.min())
        tmax = pd.to_datetime(st.session_state.cdf.index.max())
        d1, d2 = st.date_input(
            "Date range",
            value=(tmin.date(), tmax.date()),
            min_value=tmin.date(),
            max_value=tmax.date()
        )
        idx_tz = st.session_state.cdf.index.tz
        d1_ts = pd.Timestamp(d1)
        d2_ts = pd.Timestamp(d2) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        if idx_tz is not None:
            d1_ts = d1_ts.tz_localize(idx_tz, nonexistent="shift_forward")
            d2_ts = d2_ts.tz_localize(idx_tz, nonexistent="shift_forward")

        # filter and slice
        view = df_for_view.loc[(st.session_state.cdf.index >= d1_ts) & (st.session_state.cdf.index <= d2_ts)]
        if "timestamp" in show_cols:
            # show index as a column
            view = view.reset_index().rename(columns={"index": "timestamp"})
        view = view[show_cols] if show_cols else view

        # downsample large selections for responsiveness
        if len(view) > 20000:
            st.caption("Large selection — showing every 3rd row for responsiveness.")
            view = view.iloc[::3]

        render_virtualized_table(view, height=360, key="epw_raw_aggrid", page_size=75)

        # download filtered CSV
        csv_bytes = view.to_csv(index=False).replace("\r\n", "\n").encode("utf-8")
        st.download_button("⬇️ Download filtered CSV", data=csv_bytes,
                           file_name="climate_filtered.csv", mime="text/csv")

    # ---------- EPW HEADER (optional) ----------
    with st.expander("EPW header (metadata)"):
        st.json(st.session_state.header, expanded=False)



def render_short_term_prediction_page():
    import models.forecasting as fc
    import importlib
    importlib.reload(fc)
    page = st.session_state.get("nav_page")
    cdf = st.session_state.get("cdf")
    
    st.markdown("### 📈 Short-Term Prediction (10-Day NWP)")
    st.caption(
        "Fetch a 10-day deterministic hourly weather forecast from the Open-Meteo external NWP API "
        "(based on ECMWF/GFS models). You can evaluate upcoming heat events or download the forecast as an EPW file for immediate building simulations."
    )

    focus_threshold = float(st.session_state.get("custom_overheat_threshold", 30))

    base_header = st.session_state.get("header")
    if base_header is None:
        st.info("Please load an EPW file first to establish the geographic location for the forecast.")
        return
        
    loc = base_header.get("location", {}) if isinstance(base_header, dict) else {}
    lat = loc.get("latitude")
    lon = loc.get("longitude")
    if lat is None or lon is None:
        st.error("EPW header is missing latitude/longitude coordinates required for the NWP lookup.")
        return

    st.session_state.setdefault("short_forecast", None)
    st.session_state.setdefault("short_forecast_bias", None)
    st.session_state.setdefault("short_forecast_daily", None)

    if st.button("Fetch 10-Day NWP Forecast", type="primary"):
        with st.spinner(f"Querying Open-Meteo API for {float(lat):.3f}, {float(lon):.3f}…"):
            try:
                forecast_df, daily_df = fc.fetch_openmeteo_10day_forecast(float(lat), float(lon))
                epw_clim_short = fc.load_epw_climatology(cdf)
                bias_df = fc.compare_forecast_to_epw(forecast_df, epw_clim_short)
                st.session_state["short_forecast"] = forecast_df
                st.session_state["short_forecast_bias"] = bias_df
                st.session_state["short_forecast_daily"] = daily_df
                st.success("Successfully loaded 10-Day Forecast.")
            except Exception as exc:
                st.error(f"Failed to fetch forecast from Open-Meteo API: {exc}")

    forecast_df = st.session_state.get("short_forecast")
    bias_df = st.session_state.get("short_forecast_bias")
    daily_df = st.session_state.get("short_forecast_daily")
    
    if forecast_df is None or forecast_df.empty:
        st.info("Click the button above to generate a 10-day forecast.")
    else:
        temp_series = pd.Series(forecast_df["temp_forecast"])
        max_temp = float(temp_series.max()) if not temp_series.empty else float("nan")
        overheating_hours = int((temp_series >= focus_threshold).sum())
        delta_series = bias_df["epw_temp_bias_forecast"] if bias_df is not None and not bias_df.empty else pd.Series(dtype=float)
        delta_mean = float(delta_series.mean()) if not delta_series.empty else np.nan

        # Optionally show recent sensor history if loaded
        sensor_hourly = ls.load_sensor_data()
        history_series = None
        if not sensor_hourly.empty and "temperature" in sensor_hourly.columns:
            recent = sensor_hourly["temperature"].dropna().tail(24*7) # Last 7 days
            if not recent.empty:
                history_series = recent

        m1, m2, m3 = st.columns(3)
        m1.metric("10-Day max temperature", format_temperature(max_temp))
        m2.metric(f"{format_threshold_label(focus_threshold)} hours", f"{overheating_hours}")
        m3.metric("Mean Δ forecast vs EPW", format_temperature_delta(delta_mean) if not np.isnan(delta_mean) else "—")

        if daily_df is not None and not daily_df.empty:
            hdd, cdd = fc.calculate_degree_days(daily_df, base_temp=18.0)
            e1, e2 = st.columns(2)
            e1.metric("Heating Degree Days (Base 18°C)", f"{hdd:.1f} °C-days")
            e2.metric("Cooling Degree Days (Base 18°C)", f"{cdd:.1f} °C-days")

        if daily_df is not None and not daily_df.empty:
            st.markdown("#### 10-Day Outlook")
            cards_html = f"""
            <style>
            .weather-scroller {{
                display: flex;
                overflow-x: auto;
                gap: 12px;
                padding: 10px 4px 20px 4px;
                scrollbar-width: thin;
            }}
            .weather-scroller::-webkit-scrollbar {{
                height: 6px;
            }}
            .weather-scroller::-webkit-scrollbar-thumb {{
                background-color: rgba(156, 163, 175, 0.5);
                border-radius: 4px;
            }}
            .weather-card {{
                flex: 0 0 auto;
                width: 100px;
                background: rgba(30, 41, 59, 0.5);
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 12px;
                padding: 12px 8px;
                text-align: center;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: space-between;
                min-height: 140px;
            }}
            .wc-day {{ font-size: 0.9rem; font-weight: 600; color: #e2e8f0; margin-bottom: 4px; }}
            .wc-icon {{ font-size: 2rem; margin: 4px 0; }}
            .wc-temps {{ font-size: 0.95rem; font-weight: 700; color: #f8fafc; margin-top: 4px; }}
            .wc-temps span {{ color: #94a3b8; font-weight: 500; font-size: 0.85rem; margin-left: 6px; }}
            .wc-precip {{ font-size: 0.75rem; color: #60a5fa; margin-top: 6px; font-weight: 600; display: flex; align-items: center; justify-content: center; gap: 2px; }}
            </style>
            <div class="weather-scroller">
            """
            import datetime
            for _, row in daily_df.iterrows():
                dt = pd.to_datetime(row['date'])
                day_name = dt.strftime('%a') if dt.date() != datetime.datetime.now().date() else "Today"
                emoji = row['emoji']
                tmax = int(round(row['temp_max']))
                tmin = int(round(row['temp_min']))
                precip = int(row['precip_prob'])
                precip_html = f"<div class='wc-precip'>💧 {precip}%</div>" if precip > 0 else "<div class='wc-precip' style='opacity:0'>💧 0%</div>"
                cards_html += f"""
                <div class="weather-card">
                    <div class="wc-day">{day_name}</div>
                    <div class="wc-icon">{emoji}</div>
                    <div class="wc-temps">{tmax}°<span>{tmin}°</span></div>
                    {precip_html}
                </div>
                """
            cards_html += "</div>"
            try:
                st.html(cards_html)
            except AttributeError:
                st.markdown(cards_html, unsafe_allow_html=True)

        st.markdown("#### Hourly Detail")
        _st_plotly_chart(
            fc.plot_forecast(forecast_df, recent_history=history_series),
            use_container_width=True,
        )

        peak = fc.summarize_peak_event(forecast_df)
        if peak and peak.get("temp") is not None and not np.isnan(peak.get("temp", np.nan)):
            ts = peak.get("timestamp")
            ts_label = "Peak hour" if pd.isna(ts) else pd.Timestamp(ts).strftime("%b %d %H:%M")
            st.caption(f"Peak around {ts_label}: {format_temperature(peak['temp'])}.")

        st.info("Forecasts are sourced from Open-Meteo leveraging deterministic ECMWF/GFS outputs.")

        st.markdown("#### Solar Potential & Irradiance")
        _st_plotly_chart(fc.plot_solar_potential(forecast_df), use_container_width=True)

        st.markdown("#### EPW vs forecast bias")
        _st_plotly_chart(fc.plot_bias(bias_df if bias_df is not None else pd.DataFrame()), use_container_width=True)

        st.markdown("#### Overheating flags")
        _st_plotly_chart(fc.plot_overheating(forecast_df), use_container_width=True)

        st.markdown("#### Forecast Data & EPW Export")

        forecast_epw_df = fc.build_forecast_epw_dataframe(forecast_df)
        epw_blob = compose_epw_text(base_header, forecast_epw_df)

        st.download_button(
            label="⬇️ Download Forecast EPW",
            data=epw_blob,
            file_name="forecast_10day.epw",
            mime="text/plain",
            type="secondary"
        )

        with st.expander("Forecast table (Raw data)", expanded=False):
            st.dataframe(forecast_df.set_index("timestamp"), use_container_width=True, height=260)


def render_future_climate_page():
    import models.future_epw as fepw

    page = st.session_state.get("nav_page")
    cdf = st.session_state.get("cdf")

    st.markdown("### 🌍 Future Climate Scenarios")
    st.caption(
        "Blend today's EPW (optionally bias-corrected by your sensors) with CMIP6 deltas to sketch how typical "
        "years shift under SSP scenarios. Pick a pathway and horizon below to see the temperature/comfort "
        "impacts and download morphed EPWs."
    )
    focus_threshold = float(st.session_state.get("custom_overheat_threshold", 30))

    base_df = st.session_state.get("df")
    header_meta = st.session_state.get("header")
    if base_df is None or base_df.empty or header_meta is None:
        st.info("Load an EPW file on the main tabs to unlock future morphing.")
    else:
        loc_meta = (header_meta or {}).get("location", {})
        scenario_label = st.selectbox(
            "Scenario",
            list(fepw.SCENARIO_MAP.keys()),
            format_func=lambda key: f"{key} · {fepw.SCENARIO_DESCRIPTIONS.get(key, '')}"
        )
        target_year = st.selectbox("Reporting year", fepw.TARGET_YEARS, format_func=lambda y: f"{y}")
        use_sensor_baseline = st.toggle("Use sensor-calibrated baseline", value=True)
        temp_only = st.toggle("Apply CMIP6 deltas to temperature only", value=False)
        st.caption(
            "CMIP6 deltas are monthly climate-change adjustments from global model ensembles. "
            "This feature keeps your EPW's hour-by-hour local weather pattern, then shifts temperature, "
            "humidity, wind, and solar values for the selected SSP pathway and year so you can test future "
            "comfort and load risk."
        )
        baseline_name = "Current EPW"
        baseline_frame = st.session_state.get("cdf")
        if baseline_frame is None or baseline_frame.empty:
            baseline_frame = base_df

        current_lat = loc_meta.get("latitude")
        cmip6_table = st.session_state.get("cmip6_deltas")
        stored_lat = st.session_state.get("cmip6_latitude")
        is_custom_deltas = st.session_state.get("cmip6_is_custom", False)
        if (
            cmip6_table is None
            or (
                not is_custom_deltas
                and current_lat is not None
                and stored_lat is not None
                and abs(float(current_lat) - float(stored_lat)) > 5
            )
        ):
            cmip6_table = fepw.load_cmip6_deltas(current_lat)
            st.session_state["cmip6_deltas"] = cmip6_table
            st.session_state["cmip6_latitude"] = current_lat
            st.session_state["cmip6_is_custom"] = False

        # Show CMIP6 Deltas table
        if cmip6_table is not None:
            s_map = fepw.SCENARIO_MAP.get(scenario_label, scenario_label)
            active_deltas = cmip6_table[(cmip6_table["scenario"] == s_map) & (cmip6_table["year"] == target_year)]
            if not active_deltas.empty:
                with st.expander("ℹ️ Show applied monthly CMIP6 delta values", expanded=False):
                    st.markdown(f"**Monthly Climate Deltas for {scenario_label} ({target_year})**")
                    cols_to_show = ["month", "delta_temp", "delta_rh", "delta_wind", "delta_ghi"]
                    available_cols = [c for c in cols_to_show if c in active_deltas.columns]
                    display_deltas = active_deltas[available_cols].copy()
                    
                    rename_dict = {
                        "month": "Month",
                        "delta_temp": "Temp Delta (°C)",
                        "delta_rh": "RH Delta (%)",
                        "delta_wind": "Wind Delta (m/s)",
                        "delta_ghi": "GHI Delta (W/m²)"
                    }
                    display_deltas = display_deltas.rename(columns=rename_dict)
                    if "Month" in display_deltas.columns:
                        import calendar
                        display_deltas["Month"] = display_deltas["Month"].apply(lambda m: calendar.month_name[int(m)] if pd.notna(m) and 1 <= int(m) <= 12 else str(m))
                    st.dataframe(display_deltas, use_container_width=True, hide_index=True)

        with st.expander("Use custom CMIP6 delta CSV", expanded=False):
            st.caption(
                "Upload a CSV with columns scenario, year, month, delta_temp, delta_rh, "
                "delta_wind, and delta_ghi, with optional lat_band. Each row tells the app how much to "
                "shift one month for a scenario and reporting year."
            )
            custom_deltas = st.file_uploader("Custom delta table", type=["csv"], key="custom_delta_upload")
            if custom_deltas is not None:
                try:
                    cmip6_table = pd.read_csv(custom_deltas)
                    st.session_state["cmip6_deltas"] = cmip6_table
                    st.session_state["cmip6_is_custom"] = True
                    st.success("Loaded custom CMIP6 deltas.")
                except Exception as exc:
                    st.error(f"Failed to parse delta CSV: {exc}")

        sensor_df_full = ls.load_sensor_data()
        sensor_clim = ls.build_sensor_climatology(sensor_df_full)
        epw_clim_future = ls.build_epw_climatology(cdf)
        bias_table, bias_coverage = fepw.compute_sensor_bias(sensor_clim, epw_clim_future)
        bias_payload = bias_table
        if use_sensor_baseline:
            st.caption(f"Bias coverage across the year: {bias_coverage:.0%}")
            if bias_coverage < 0.4:
                st.warning("Need at least ~40% hour coverage across the year for reliable bias correction. Using raw EPW instead.")
                bias_payload = None
        else:
            bias_payload = None

        st.info("These scenarios morph a typical year by applying average climate deltas for planning and design-weather review; they are not weather forecasts for a specific future date.")

        with st.spinner("Morphing EPW datasets with CMIP6 deltas…"):
            payloads = fepw.generate_download_payloads(
                base_df,
                header_meta,
                scenario_label,
                cmip6_table,
                bias_payload if bias_payload is not None and not bias_payload.empty else None,
                use_sensor_baseline,
                temp_only,
            )

        active_bundle = payloads.get(target_year)
        if not active_bundle:
            st.error("Unable to build the requested future EPW.")
        else:
            future_df = active_bundle["df"]
            delta_temp = future_df["drybulb"].mean() - base_df["drybulb"].mean()
            if "relhum" in future_df and "relhum" in base_df:
                delta_rh = future_df["relhum"].mean() - base_df["relhum"].mean()
            else:
                delta_rh = np.nan
            overheating_future = int((future_df["drybulb"] > focus_threshold).sum())

            c1, c2, c3 = st.columns(3)
            c1.metric("ΔT annual mean", format_temperature_delta(delta_temp))
            c2.metric("ΔRH annual mean", f"{delta_rh:+.1f} %" if not np.isnan(delta_rh) else "—")
            c3.metric(f"Future {format_threshold_label(focus_threshold)} hours", f"{overheating_future}")

            def _monthly_curve(df: pd.DataFrame) -> pd.Series:
                return df["drybulb"].groupby(df.index.month).mean()

            baseline_curve = _monthly_curve(base_df)
            curve_2050 = payloads[2050]["df"]["drybulb"].groupby(payloads[2050]["df"].index.month).mean()
            curve_2080 = payloads[2080]["df"]["drybulb"].groupby(payloads[2080]["df"].index.month).mean()

            months = list(range(1, 13))
            fig_future = go.Figure()
            fig_future.add_trace(go.Scatter(x=months, y=baseline_curve.reindex(months), mode="lines", name="Current", line=dict(color="#94a3b8")))
            fig_future.add_trace(go.Scatter(x=months, y=curve_2050.reindex(months), mode="lines", name="2050", line=dict(color="#60a5fa")))
            fig_future.add_trace(go.Scatter(x=months, y=curve_2080.reindex(months), mode="lines", name="2080", line=dict(color="#f97316")))
            fig_future.update_layout(
                height=360,
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis=dict(tickmode="array", tickvals=months, ticktext=[pd.Timestamp(2001, m, 1).strftime("%b") for m in months]),
                yaxis_title=f"Monthly mean ({'°F' if _temp_unit() == 'F' else '°C'})",
                template=PLOTLY_TEMPLATE
            )
            st.markdown("#### Climate shift (monthly means)")
            _st_plotly_chart(fig_future, use_container_width=True)

            # ----- Comfort & load comparison -----
            st.markdown("#### Comfort & load outlook")
            scenario_frames: Dict[str, pd.DataFrame] = {}
            scenario_frames[baseline_name] = baseline_frame.copy()
            for year, bundle in payloads.items():
                frame = bundle.get("df")
                if frame is None or frame.empty:
                    continue
                try:
                    scenario_frames[f"{year} · {scenario_label}"] = build_clima_dataframe(frame)
                except Exception:
                    scenario_frames[f"{year} · {scenario_label}"] = frame

            comfort_compare = ce.compare_comfort_across_scenarios(scenario_frames)

            if comfort_compare.empty:
                st.info("Need dry-bulb, humidity, and wind speed fields to quantify comfort deltas.")
            else:
                if "error" in comfort_compare.columns:
                    errors = comfort_compare["error"].dropna()
                    if not errors.empty:
                        st.warning("Unable to compute comfort metrics for some scenarios. Showing available data only.")
                        st.write(errors)
                display_map = {
                    "fraction_in_comfort_band": "Comfort %",
                    "overheating_hours_28C": f"{format_threshold_label(28)} h",
                    "overheating_hours_30C": f"{format_threshold_label(30)} h",
                    "hours_utci_heat_stress": "UTCI heat h",
                    "hours_utci_cold_stress": "UTCI cold h",
                    "hours_di_discomfort": "DI discomfort h",
                    "heating_degree_days": "HDD",
                    "cooling_degree_days": "CDD",
                }
                focus_col = f"overheating_hours_{int(focus_threshold)}C"
                display_map.setdefault(focus_col, f"{format_threshold_label(focus_threshold)} h (focus)")
                available_cols = [c for c in display_map if c in comfort_compare.columns]
                if available_cols:
                    table = comfort_compare[available_cols].copy()
                    if "fraction_in_comfort_band" in table:
                        table["fraction_in_comfort_band"] = table["fraction_in_comfort_band"] * 100.0
                    st.dataframe(
                        table.rename(columns=display_map).round(1),
                        use_container_width=True,
                        height=240,
                    )

                target_label = f"{target_year} · {scenario_label}"
                if baseline_name in comfort_compare.index and target_label in comfort_compare.index:
                    base_row = comfort_compare.loc[baseline_name]
                    target_row = comfort_compare.loc[target_label]

                    def _numeric_metric(series, key) -> float:
                        val = series.get(key, np.nan)
                        return np.nan if pd.isna(val) else float(val)

                    def _metric_value(series, key, pct=False, suffix=" h"):
                        val = _numeric_metric(series, key)
                        if pd.isna(val):
                            return "—"
                        return f"{val * 100:.1f} %" if pct else f"{val:.0f}{suffix}"

                    def _delta(target, base, key, pct=False, suffix=" h"):
                        tv, bv = _numeric_metric(target, key), _numeric_metric(base, key)
                        if pd.isna(tv) or pd.isna(bv):
                            return None
                        delta = tv - bv
                        if pct:
                            return f"{delta * 100:+.1f} ppt"
                        return f"{delta:+.0f}{suffix}"

                    def _metric_has_signal(key: str) -> bool:
                        vals = [_numeric_metric(target_row, key), _numeric_metric(base_row, key)]
                        return any(pd.notna(v) and abs(v) > 1e-9 for v in vals)

                    focus_display_key = focus_col if focus_col in comfort_compare.columns else "overheating_hours_28C"
                    focus_label = display_map.get(focus_display_key, f"{format_threshold_label(focus_threshold)} h")

                    stress_key = "hours_utci_heat_stress"
                    stress_label = "UTCI heat stress"
                    if not _metric_has_signal(stress_key) and "hours_utci_cold_stress" in comfort_compare.columns:
                        stress_key = "hours_utci_cold_stress"
                        stress_label = "UTCI cold stress"

                    load_key = "cooling_degree_days"
                    load_label = "Cooling degree days"
                    if not _metric_has_signal(load_key) and "heating_degree_days" in comfort_compare.columns:
                        load_key = "heating_degree_days"
                        load_label = "Heating degree days"

                    third_key = stress_key if _metric_has_signal(stress_key) else load_key
                    third_label = stress_label if third_key == stress_key else load_label
                    third_suffix = " h" if third_key == stress_key else " DD"

                    k1, k2, k3 = st.columns(3)
                    k1.metric(
                        "Comfort compliance",
                        _metric_value(target_row, "fraction_in_comfort_band", pct=True),
                        delta=_delta(target_row, base_row, "fraction_in_comfort_band", pct=True),
                    )
                    k2.metric(
                        focus_label.replace(" h", " hours"),
                        _metric_value(target_row, focus_display_key),
                        delta=_delta(target_row, base_row, focus_display_key),
                    )
                    k3.metric(
                        third_label,
                        _metric_value(target_row, third_key, suffix=third_suffix),
                        delta=_delta(target_row, base_row, third_key, suffix=third_suffix),
                    )

                    def _safe_val(v):
                        return None if pd.isna(v) else float(v)

                    candidate_bars = [
                        ("Comfort band %", "fraction_in_comfort_band", True),
                        (focus_label, focus_display_key, False),
                        ("UTCI heat h", "hours_utci_heat_stress", False),
                        ("UTCI cold h", "hours_utci_cold_stress", False),
                        ("CDD", "cooling_degree_days", False),
                        ("HDD", "heating_degree_days", False),
                    ]
                    bars = []
                    seen_bar_keys = set()
                    for label, key, pct in candidate_bars:
                        if key in seen_bar_keys or key not in comfort_compare.columns:
                            continue
                        seen_bar_keys.add(key)
                        tv = _numeric_metric(target_row, key)
                        bv = _numeric_metric(base_row, key)
                        if pd.isna(tv) and pd.isna(bv):
                            continue
                        tv_plot = tv * 100.0 if pct and pd.notna(tv) else tv
                        bv_plot = bv * 100.0 if pct and pd.notna(bv) else bv
                        has_signal = any(pd.notna(v) and abs(v) > 1e-9 for v in (tv_plot, bv_plot))
                        if not has_signal:
                            continue
                        bars.append((label, tv_plot, bv_plot))

                    if bars:
                        bar_categories = [x[0] for x in bars]
                        target_vals = [_safe_val(x[1]) for x in bars]
                        base_vals = [_safe_val(x[2]) for x in bars]
                        fig_compare = go.Figure()
                        fig_compare.add_bar(name=target_label, x=bar_categories, y=target_vals, marker_color="#f97316")
                        fig_compare.add_bar(name=baseline_name, x=bar_categories, y=base_vals, marker_color="#94a3b8")
                        fig_compare.update_layout(
                            barmode="group",
                            height=360,
                            margin=dict(l=0, r=0, t=30, b=0),
                            yaxis_title="Metric value",
                            hovermode="x unified",
                        )
                        _st_plotly_chart(fig_compare, use_container_width=True)
                    else:
                        st.info("The selected future scenario does not change the available comfort/load metrics enough to plot a comparison.")

            # ── DIURNAL HEATMAPS ──────────────────────────────────
            st.markdown(f"#### Diurnal Heatmap — Future Temperature ({target_year})")
            _hour_labels = [f"{h:02d}:00" for h in range(24)]

            def _date_label_from_key(day_key: object) -> str:
                try:
                    return pd.Timestamp(f"2000-{day_key}").strftime("%b %d")
                except Exception:
                    return str(day_key)

            def _date_axis_labels(grid: Optional[pd.DataFrame]) -> List[str]:
                if grid is None:
                    return []
                return [_date_label_from_key(day_key) for day_key in grid.columns]

            def _monthly_date_ticks(date_labels: List[str]) -> List[str]:
                ticks = [label for label in date_labels if label.endswith("01")]
                if ticks:
                    return ticks
                step = max(len(date_labels) // 12, 1)
                return date_labels[::step]

            def _build_diurnal_grid(df_src, col="drybulb"):
                if col not in df_src.columns:
                    return None
                tmp = df_src[[col]].copy()
                idx = pd.DatetimeIndex(tmp.index)
                tmp["date_key"] = idx.strftime("%m-%d")
                tmp["hour"] = tmp.index.hour
                grid = tmp.pivot_table(values=col, index="hour", columns="date_key", aggfunc="mean")
                return grid.reindex(index=range(24), columns=sorted(grid.columns))

            grid_future = _build_diurnal_grid(future_df)
            grid_base = _build_diurnal_grid(base_df)

            if grid_future is not None:
                _date_labels = _date_axis_labels(grid_future)
                _date_ticks = _monthly_date_ticks(_date_labels)
                h1, h2 = st.columns(2)
                with h1:
                    st.caption(f"{target_year} · {scenario_label}")
                    fig_heat_f = go.Figure(go.Heatmap(
                        z=grid_future.values, x=_date_labels, y=_hour_labels,
                        colorscale="RdYlBu_r", colorbar=dict(title="°C"),
                        hovertemplate=f"Date: %{{x}}, {target_year}<br>Hour: %{{y}}<br>Temp: %{{z:.1f}} °C<extra></extra>",
                    ))
                    fig_heat_f.update_layout(height=380, margin=dict(l=60, r=10, t=30, b=40),
                                             xaxis=dict(title=f"Date ({target_year})", tickmode="array", tickvals=_date_ticks, ticktext=_date_ticks, showgrid=False),
                                             yaxis=dict(autorange="reversed", showgrid=False), template=PLOTLY_TEMPLATE,
                                             paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font=dict(color="#e2e8f0"))
                    _st_plotly_chart(fig_heat_f, use_container_width=True)
                with h2:
                    st.caption("ΔT (Future – Baseline)")
                    if grid_base is not None:
                        delta_grid = grid_future - grid_base.reindex(index=grid_future.index, columns=grid_future.columns)
                        delta_values = delta_grid.to_numpy(dtype=float)
                        abs_max = np.nanmax(np.abs(delta_values)) if np.isfinite(delta_values).any() else 0.1
                        abs_max = max(abs_max, 0.1)
                        fig_heat_d = go.Figure(go.Heatmap(
                            z=delta_grid.values, x=_date_labels, y=_hour_labels,
                            colorscale="RdBu_r", zmid=0, zmin=-abs_max, zmax=abs_max,
                            colorbar=dict(title="Δ°C"),
                            hovertemplate=f"Date: %{{x}}, {target_year}<br>Hour: %{{y}}<br>ΔT: %{{z:+.2f}} °C<extra></extra>",
                        ))
                        fig_heat_d.update_layout(height=380, margin=dict(l=60, r=10, t=30, b=40),
                                                 xaxis=dict(title=f"Date ({target_year})", tickmode="array", tickvals=_date_ticks, ticktext=_date_ticks, showgrid=False),
                                                 yaxis=dict(autorange="reversed", showgrid=False), template=PLOTLY_TEMPLATE,
                                                 paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font=dict(color="#e2e8f0"))
                        _st_plotly_chart(fig_heat_d, use_container_width=True)

            # UTCI diurnal heatmap
            try:
                utci_future = ce.compute_utci_approx(build_clima_dataframe(future_df))
                utci_base = ce.compute_utci_approx(baseline_frame)
                tmp_uf = pd.DataFrame({"UTCI": utci_future}, index=future_df.index)
                grid_utci = _build_diurnal_grid(tmp_uf, "UTCI")
                st.markdown(f"#### Diurnal Heatmap — UTCI (Future {target_year})")
                if grid_utci is not None:
                    _utci_date_labels = _date_axis_labels(grid_utci)
                    _utci_date_ticks = _monthly_date_ticks(_utci_date_labels)
                    fig_utci_h = go.Figure(go.Heatmap(
                        z=grid_utci.values, x=_utci_date_labels, y=_hour_labels,
                        colorscale=[[0, "#2166ac"], [0.35, "#67a9cf"], [0.5, "#fddbc7"],
                                    [0.7, "#ef8a62"], [1, "#b2182b"]],
                        colorbar=dict(title="UTCI °C"),
                        hovertemplate=f"Date: %{{x}}, {target_year}<br>Hour: %{{y}}<br>UTCI: %{{z:.1f}} °C<extra></extra>",
                    ))
                    fig_utci_h.update_layout(height=380, margin=dict(l=60, r=10, t=30, b=40),
                                             xaxis=dict(title=f"Date ({target_year})", tickmode="array", tickvals=_utci_date_ticks, ticktext=_utci_date_ticks, showgrid=False),
                                             yaxis=dict(autorange="reversed", showgrid=False), template=PLOTLY_TEMPLATE,
                                             paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font=dict(color="#e2e8f0"))
                    _st_plotly_chart(fig_utci_h, use_container_width=True)
            except Exception:
                utci_future = None
                utci_base = None

            # ── ANNUAL DISTRIBUTION ───────────────────────────────
            st.markdown("#### Annual Temperature Distribution")
            ad1, ad2 = st.columns(2)
            with ad1:
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=base_df["drybulb"], nbinsx=60, name="Baseline",
                    marker_color="rgba(148,163,184,0.5)", opacity=0.7,
                ))
                fig_dist.add_trace(go.Histogram(
                    x=future_df["drybulb"], nbinsx=60, name=f"{target_year}",
                    marker_color="rgba(249,115,22,0.5)", opacity=0.7,
                ))
                base_mean = base_df["drybulb"].mean()
                future_mean = future_df["drybulb"].mean()
                base_p95 = base_df["drybulb"].quantile(0.95)
                future_p95 = future_df["drybulb"].quantile(0.95)
                for val, clr, nm in [(base_mean, "#94a3b8", "Base mean"),
                                      (future_mean, "#f97316", f"{target_year} mean")]:
                    fig_dist.add_vline(x=val, line_dash="dash", line_color=clr,
                                       annotation_text=f"{nm}: {val:.1f}°C",
                                       annotation_position="top right")
                fig_dist.update_layout(barmode="overlay", height=340,
                                       margin=dict(l=0, r=0, t=30, b=0),
                                       xaxis_title="Dry-Bulb Temperature (°C)",
                                       yaxis_title="Hours", template=PLOTLY_TEMPLATE,
                                       paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font=dict(color="#e2e8f0"))
                _st_plotly_chart(fig_dist, use_container_width=True)
            with ad2:
                if utci_future is not None and utci_base is not None:
                    fig_utci_dist = go.Figure()
                    fig_utci_dist.add_trace(go.Histogram(
                        x=utci_base.values, nbinsx=60, name="Baseline UTCI",
                        marker_color="rgba(148,163,184,0.5)", opacity=0.7,
                    ))
                    fig_utci_dist.add_trace(go.Histogram(
                        x=utci_future.values, nbinsx=60, name=f"{target_year} UTCI",
                        marker_color="rgba(239,68,68,0.5)", opacity=0.7,
                    ))
                    fig_utci_dist.update_layout(barmode="overlay", height=340,
                                                margin=dict(l=0, r=0, t=30, b=0),
                                                xaxis_title="UTCI (°C)",
                                                yaxis_title="Hours", template=PLOTLY_TEMPLATE,
                                                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font=dict(color="#e2e8f0"))
                    _st_plotly_chart(fig_utci_dist, use_container_width=True)
                else:
                    st.caption("UTCI distribution unavailable — missing wind speed data.")

            # ── SHIFT PLOT (MONTHLY BOX) ──────────────────────────
            st.markdown("#### Monthly Shift Plot — Temperature")
            shift_data = []
            for lbl, src_df in [("Baseline", base_df),
                                ("2050", payloads[2050]["df"]),
                                ("2080", payloads[2080]["df"])]:
                tmp_s = src_df[["drybulb"]].copy()
                tmp_s["month"] = tmp_s.index.month
                tmp_s["scenario"] = lbl
                shift_data.append(tmp_s)
            shift_all = pd.concat(shift_data, ignore_index=True)
            shift_all["month_name"] = shift_all["month"].map(
                lambda m: pd.Timestamp(2001, m, 1).strftime("%b"))
            colors_shift = {"Baseline": "#94a3b8", "2050": "#60a5fa", "2080": "#f97316"}
            fig_shift = go.Figure()
            for sc_name in ["Baseline", "2050", "2080"]:
                subset = shift_all[shift_all["scenario"] == sc_name]
                fig_shift.add_trace(go.Box(
                    x=subset["month_name"], y=subset["drybulb"],
                    name=sc_name, marker_color=colors_shift[sc_name],
                    boxmean=True, line_width=1.2,
                ))
            fig_shift.update_layout(
                boxmode="group", height=400,
                margin=dict(l=0, r=0, t=30, b=0),
                yaxis_title=f"Dry-Bulb Temperature ({'°F' if _temp_unit() == 'F' else '°C'})",
                template=PLOTLY_TEMPLATE, legend=dict(orientation="h", y=1.05),
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font=dict(color="#e2e8f0"),
            )
            _st_plotly_chart(fig_shift, use_container_width=True)

            # ── UTCI / DI STRESS BREAKDOWN ────────────────────────
            st.markdown("#### Thermal Stress Breakdown")
            try:
                cdf_future = build_clima_dataframe(future_df)
                di_future = ce.compute_di(cdf_future)
                di_cats_f = ce.classify_di(di_future)
                utci_cats_f = ce.classify_utci(utci_future) if utci_future is not None else None

                di_base = ce.compute_di(baseline_frame)
                di_cats_b = ce.classify_di(di_base)
                utci_cats_b = ce.classify_utci(utci_base) if utci_base is not None else None

                sb1, sb2 = st.columns(2)

                # DI breakdown
                with sb1:
                    st.caption("Discomfort Index (DI)")
                    di_order = ["Comfortable", "Slight Discomfort", "Discomfort",
                                "Strong Discomfort", "Medical Emergency"]
                    di_colors = {"Comfortable": "#22c55e", "Slight Discomfort": "#facc15",
                                 "Discomfort": "#f97316", "Strong Discomfort": "#ef4444",
                                 "Medical Emergency": "#991b1b"}
                    fig_di = go.Figure()
                    for period_lbl, cats in [(f"{target_year}", di_cats_f), ("Baseline", di_cats_b)]:
                        vc = cats.value_counts(normalize=True).reindex(di_order, fill_value=0) * 100
                        fig_di.add_trace(go.Bar(
                            y=[period_lbl] * len(vc), x=vc.values, orientation="h",
                            name=period_lbl, marker_color=[di_colors.get(c, "#888") for c in vc.index],
                            text=[f"{c}: {v:.1f}%" for c, v in zip(vc.index, vc.values)],
                            textposition="inside", showlegend=False,
                        ))
                    fig_di.update_layout(barmode="stack", height=200,
                                         margin=dict(l=0, r=0, t=10, b=0),
                                         xaxis_title="% of hours", template=PLOTLY_TEMPLATE,
                                         paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font=dict(color="#e2e8f0"))
                    _st_plotly_chart(fig_di, use_container_width=True)

                # UTCI breakdown
                with sb2:
                    if utci_cats_f is not None and utci_cats_b is not None:
                        st.caption("UTCI Stress Categories")
                        utci_order = ["Extreme cold stress", "Very strong cold stress",
                                      "Strong cold stress", "Moderate cold stress",
                                      "No thermal stress", "Moderate heat stress",
                                      "Strong heat stress", "Very strong heat stress",
                                      "Extreme heat stress"]
                        utci_colors_map = {
                            "Extreme cold stress": "#08306b", "Very strong cold stress": "#2171b5",
                            "Strong cold stress": "#4292c6", "Moderate cold stress": "#6baed6",
                            "No thermal stress": "#22c55e", "Moderate heat stress": "#facc15",
                            "Strong heat stress": "#f97316", "Very strong heat stress": "#ef4444",
                            "Extreme heat stress": "#991b1b",
                        }
                        fig_utci_bar = go.Figure()
                        for period_lbl, cats in [(f"{target_year}", utci_cats_f), ("Baseline", utci_cats_b)]:
                            vc = cats.value_counts(normalize=True).reindex(utci_order, fill_value=0) * 100
                            fig_utci_bar.add_trace(go.Bar(
                                y=[period_lbl] * len(vc), x=vc.values, orientation="h",
                                name=period_lbl,
                                marker_color=[utci_colors_map.get(c, "#888") for c in vc.index],
                                text=[f"{v:.1f}%" if v > 3 else "" for v in vc.values],
                                textposition="inside", showlegend=False,
                            ))
                        fig_utci_bar.update_layout(barmode="stack", height=200,
                                                   margin=dict(l=0, r=0, t=10, b=0),
                                                   xaxis_title="% of hours", template=PLOTLY_TEMPLATE,
                                                   paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font=dict(color="#e2e8f0"))
                        _st_plotly_chart(fig_utci_bar, use_container_width=True)
                    else:
                        st.caption("UTCI breakdown unavailable.")
            except Exception as _stress_exc:
                st.warning(f"Could not compute stress breakdown: {_stress_exc}")

            # ── DOWNLOAD BUTTONS ──────────────────────────────────
            st.markdown("---")
            d1, d2 = st.columns(2)
            for year in fepw.TARGET_YEARS:
                bundle = payloads.get(year)
                if not bundle:
                    continue
                label = f"Download {year} EPW"
                col = d1 if year == 2050 else d2
                col.download_button(
                    label,
                    data=bundle["bytes"],
                    file_name=bundle["file_name"],
                    mime="text/plain",
                    use_container_width=True,
                )

            with st.expander("Preview future EPW (selected year)", expanded=False):
                preview_cols = ["drybulb"]
                if "relhum" in future_df:
                    preview_cols.append("relhum")
                st.dataframe(future_df[preview_cols].tail(168), use_container_width=True, height=260)


# Module level footer removed as it rendered before page content.


def main():
    """Main execution loop for the Weather Analysis App."""
    
    # 1. Setup session state & CSS
    st.markdown(PREMIUM_CSS, unsafe_allow_html=True)
    st.markdown(SECONDARY_CSS, unsafe_allow_html=True)
    st.markdown(CLIMATE_INTELLIGENCE_CSS, unsafe_allow_html=True)
    _install_plotly_capture_hook()

    # 2. Render Header (Globally Fixed Sticky UI Bar)
    render_header()

    # 3. Render Sidebar (handles navigation state)
    render_sidebar()

    # 4. Controller Logic (Load EPW, etc.)
    show_epw_status()
    setup_cdf()

    # 5. Evaluate Effective Routing Page State (AFTER sidebar so nav_page is current)
    effective_page = st.session_state.get("nav_page", DEFAULT_PAGE)

    # 6. Page Routing Execution
    
    #    - Dispatch to appropriate render function
    #    - Some pages might be restricted if no EPW is loaded (handled inside functions or via sidebar disabling)
    
    if effective_page == "Select weather file":
        render_select_station_page()
    
    elif effective_page == "Dashboard":
        render_dashboard_page()

    elif effective_page == "Overview":
        render_overview_page()

    elif effective_page == "Climate":
        render_climate_page()

    elif effective_page == "Comfort":
        render_comfort_page()

    elif effective_page == "Solar":
        render_solar_page()

    elif effective_page == "Psychrometrics":
        try:
            render_psychrometrics_page()
        except Exception as e:
            st.error(f"Psychrometrics error: {str(e)}")

    elif effective_page == "Wind":
        render_wind_page()

    elif effective_page == "Raw Data":
        render_raw_data_workspace_page()

    elif effective_page == "Export":
        render_export_page()
        
    elif effective_page == "Live Data vs EPW":
        render_live_data_page()
        
    elif effective_page == "Sensor Comparison":
        render_sensor_comparison_page()
        
    elif effective_page == "Short-Term Prediction (24–72h)":
        render_short_term_prediction_page()
        
    elif effective_page == "Future Climate (2050 / 2080 SSP)":
        render_future_climate_page()
        
    else:
        st.error(f"Page '{effective_page}' not found.")

    _finalize_dashboard_pdf_if_pending(effective_page)

    # 6. Footer
    st.markdown(
        """
        <div class="bevl-footer">
            <strong>BEVL Lab</strong> &bull; UB School of Architecture & Planning
            <br/>
            <span style="opacity:0.6">Research-grade tools for the built environment.</span>
            &nbsp;
            <a href="https://archplan.buffalo.edu/research/research-centers/bevl.html" target="_blank">Visit Lab</a> &bull;
            <a href="https://github.com/UB-BEVL/climateclock/blob/main/README.md" target="_blank">Docs</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
