from __future__ import annotations
# standard libs next
import os
import calendar
from pathlib import Path
import io, zipfile, csv, math, argparse, datetime
import requests
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional, Union
import re
from streamlit_plotly_events import plotly_events
# third-party
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import pvlib
# Patch platform processor to avoid Windows WMI KeyError during h5py/pvlib import
import platform as _platform
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
        pass


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


# ========== IMPROVED LAYOUT ==========
st.set_page_config(
    page_title="Climate Analysis Pro",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.session_state.setdefault("temperature_unit", "C")

# Navigation definitions
NAV_ITEMS = [
    ("Select weather file", "Select weather file"),
    ("Dashboard", "📊 Dashboard"),
    ("Raw Data", "📁 Raw Data"),
    ("Temperature & Humidity", "🌡️ Temperature & Humidity"),
    ("Solar Analysis", "☀️ Solar Analysis"),
    ("Psychrometrics", "📈 Psychrometrics"),
    ("Live Data vs EPW", "📡 Live Data vs EPW"),
    ("Sensor Comparison", "Sensor Comparison"),
    ("Short-Term Prediction (24–72h)", "📈 Short-Term Prediction (24–72h)"),
    ("Future Climate (2050 / 2080 SSP)", "🌍 Future Climate (2050 / 2080 SSP)"),
]

FROZEN_NAV_LABELS = {
    "Short-Term Prediction (24–72h)",
    "Future Climate (2050 / 2080 SSP)",
}
FROZEN_PAGES = {
    "📈 Short-Term Prediction (24–72h)",
    "🌍 Future Climate (2050 / 2080 SSP)",
}

LABEL_TO_PAGE = {label: page for label, page in NAV_ITEMS}
PAGE_TO_LABEL = {page: label for label, page in NAV_ITEMS}
ALLOWED_PAGES = list(PAGE_TO_LABEL.keys())
DEFAULT_PAGE = "Select weather file"

st.session_state.setdefault("active_page", DEFAULT_PAGE)

THEME_BASE = "light"
try:
    theme_option = st.get_option("theme.base")
    if isinstance(theme_option, str):
        THEME_BASE = theme_option.lower()
except Exception:
    pass

PLOTLY_TEMPLATE = "plotly_dark" if THEME_BASE == "dark" else "plotly"
px.defaults.template = PLOTLY_TEMPLATE
pio.templates.default = PLOTLY_TEMPLATE

# Default station source fallback (can be overridden)
STATION_SOURCE = "https://raw.githubusercontent.com/CenterForTheBuiltEnvironment/clima/main/assets/data/epw_location.json"

# Streamlit cache decorator (data) compatible alias
try:
    CACHE = st.cache_data
except Exception:
    CACHE = lambda *args, **kwargs: (lambda fn: fn)

# --- Streamlit compat shims ---
if hasattr(st, "rerun"):
    def _rerun(): st.rerun()
else:
    def _rerun(): st.experimental_rerun()


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
        # Fix 1: Convert dots to underscores in station names (except extension)
        parts = url.split("/")
        station_part = parts[-1] if parts else ""
        if "." in station_part and "_TMY" in station_part:
            # Only replace dots before the extension
            base, ext = station_part.rsplit('.', 1)
            fixed_station = base.replace('.', '_') + '.' + ext
            fixed_url = "/".join(parts[:-1] + [fixed_station])
            alternatives.append(fixed_url)

        # Fix 2: Handle case variations
        alternatives.append(url.upper())
        alternatives.append(url.lower())

        # Fix 3: Try different TMY versions

    return list(dict.fromkeys(alternatives))


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

        for c in ["lat", "lon", "elevation_m", "heating_db", "cooling_db"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

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
        r = requests.get(src, timeout=60)
        r.raise_for_status()
        obj = r.json()

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
                df = _finish(pd.DataFrame([{
                    "name": "BUFFALO_NIAGARA_INTL_AP_725280", "country": "USA",
                    "lat": 42.94, "lon": -78.73, "elevation_m": 215, "timezone": "UTC-5",
                    "zip_url": "https://climate.onebuilding.org/WMO_Region_4_North_and_Central_Americas/USA_United_States_of_America/NY_New_York/BUFFALO_NIAGARA_INTL_AP_725280_TMYx.2007-2021.zip",
                    "period": "2007-2021", "heating_db": -17.5, "cooling_db": 29.5
                }]))
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
            df = _finish(pd.DataFrame([{
                "name": "BUFFALO_NIAGARA_INTL_AP_725280", "country": "USA",
                "lat": 42.94, "lon": -78.73, "elevation_m": 215, "timezone": "UTC-5",
                "zip_url": "https://climate.onebuilding.org/WMO_Region_4_North_and_Central_Americas/USA_United_States_of_America/NY_New_York/BUFFALO_NIAGARA_INTL_AP_725280_TMYx.2007-2021.zip",
                "period": "2007-2021", "heating_db": -17.5, "cooling_db": 29.5
            }]))
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

    return _finish(pd.DataFrame([{
        "name": "BUFFALO_NIAGARA_INTL_AP_725280", "country": "USA",
        "lat": 42.94, "lon": -78.73, "elevation_m": 215, "timezone": "UTC-5",
        "zip_url": "https://climate.onebuilding.org/WMO_Region_4_North_and_Central_Americas/USA_United_States_of_America/NY_New_York/BUFFALO_NIAGARA_INTL_AP_725280_TMYx.2007-2021.zip",
        "period": "2007-2021", "heating_db": -17.5, "cooling_db": 29.5
    }]))


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
        pass
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
        try: return float(x) if x != "" else None
        except: return None
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

    return header, df, notes

def build_clima_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Adds derived fields and convenience columns; returns cdf.
    wx = df.copy()
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
    wx["twb"] = (T*np.arctan(0.151977*np.sqrt(RH+8.313659))
                 + np.arctan(T+RH) - np.arctan(RH-1.676331)
                 + 0.00391838*(RH**1.5)*np.arctan(0.023101*RH) - 4.686035)
    P = wx["atmos_pressure"].fillna(101325.0) if "atmos_pressure" in wx else pd.Series(101325.0, index=wx.index)
    Pv = wx["vap_press"]
    wx["w"] = 0.62198 * Pv / (P - Pv)
    wx["h_kJ_per_kg_dry"] = 1.006 * T + wx["w"] * (2501.0 + 1.86 * T)

    cdf = wx.copy()
    cdf["month"] = cdf.index.month; cdf["day"] = cdf.index.day; cdf["hour"] = cdf.index.hour; cdf["doy"] = cdf.index.dayofyear
    return cdf


def build_comfort_package(cdf: pd.DataFrame) -> Dict[str, Optional[pd.DataFrame]]:
    # Compute comfort + load summaries used across multiple tabs.
    package: Dict[str, Optional[pd.DataFrame]] = {
        "di": None,
        "utci": None,
        "heat_index": None,
        "humidex": None,
        "comfort_annual": pd.DataFrame(),
        "comfort_monthly": pd.DataFrame(),
        "degree_daily": pd.DataFrame(),
        "loads_annual": pd.DataFrame(),
    }

    if cdf is None or cdf.empty:
        return package

    di = None
    if {"drybulb", "relhum"}.issubset(cdf.columns):
        try:
            di = ce.compute_di(cdf)
            package["di"] = di
        except Exception:
            di = None

    utci = None
    if {"drybulb", "relhum", "windspd"}.issubset(cdf.columns):
        try:
            utci = ce.compute_utci_approx(cdf)
            package["utci"] = utci
        except Exception:
            utci = None

    heat_index = None
    if {"drybulb", "relhum"}.issubset(cdf.columns):
        try:
            heat_index = ce.compute_heat_index(cdf)
            package["heat_index"] = heat_index
        except Exception:
            heat_index = None

    humidex = None
    if {"drybulb", "relhum"}.issubset(cdf.columns):
        try:
            humidex = ce.compute_humidex(cdf)
            package["humidex"] = humidex
        except Exception:
            humidex = None

    try:
        package["comfort_annual"] = ce.summarize_comfort(cdf, di, utci, freq="A")
    except Exception:
        pass

    try:
        package["comfort_monthly"] = ce.summarize_comfort(cdf, di, utci, freq="M")
    except Exception:
        pass

    try:
        package["degree_daily"] = ce.compute_degree_metrics(cdf, freq="D")
    except Exception:
        pass

    try:
        deg_src = package["degree_daily"] if isinstance(package["degree_daily"], pd.DataFrame) else None
        if deg_src is None or deg_src.empty:
            deg_src = ce.compute_degree_metrics(cdf)
        package["loads_annual"] = ce.summarize_loads(deg_src, freq="A")
    except Exception:
        pass

    return package

# -------------------- Optional: helpers for ZIP/URL --------------------
def read_epw_from_zip_bytes(zip_bytes: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        epws = [m for m in z.namelist() if m.lower().endswith(".epw")]
        epws.sort(key=lambda m: z.getinfo(m).file_size, reverse=True)
        with z.open(epws[0]) as f:
            return f.read()


def compose_epw_text(header: dict, df: pd.DataFrame) -> bytes:
    # Rebuild an EPW text blob from header metadata and a data frame.

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

    row_strings = [
        ",".join(_format_cell(row[col]) for col in EPW_COLUMNS)
        for _, row in df[EPW_COLUMNS].iterrows()
    ]
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
            freq="H",
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
PREMIUM_CSS = '''
<style>
    @import url("https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap");
    * { font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    .main { background: #0b0f1a; color: #c2c8d1; padding: 0.6rem 1.4rem; }
    h1 { font-size: 2.05rem !important; font-weight: 700 !important; color: #e2e8f0 !important; margin-bottom: 0.35rem !important; letter-spacing: -0.02em; }
    h2 { font-size: 1.35rem !important; font-weight: 650 !important; color: #d6dbe3 !important; margin: 1.4rem 0 0.75rem 0 !important; letter-spacing: -0.01em; }
    h3 { font-size: 1.1rem !important; font-weight: 600 !important; color: #c5cbd8 !important; margin-bottom: 0.55rem !important; }
    body, .block-container { background: #0b0f1a; color: #c2c8d1; }
    .card { background: #111624; border-radius: 6px; padding: 1.2rem 1.25rem; border: 1px solid rgba(255,255,255,0.02); box-shadow: none; }
    .card:hover { border-color: rgba(255,255,255,0.05); box-shadow: none; transform: none; }
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 650 !important; color: #e2e8f0 !important; letter-spacing: -0.02em; }
    [data-testid="stMetricLabel"] { font-size: 0.82rem !important; color: #8e96a3 !important; font-weight: 500 !important; text-transform: uppercase; letter-spacing: 0.04em; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background: transparent; padding: 10px 2px 14px 2px; border-radius: 0; border: none; }
    .stTabs [data-baseweb="tab"] { background: transparent; border-radius: 0; padding: 6px 10px 8px 10px; font-weight: 600; color: #c2c8d1; border: none; transition: color 0.1s ease; }
    .stTabs [data-baseweb="tab"]:hover { color: #e5e7eb; }
    .stTabs [aria-selected="true"] { background: transparent !important; color: #f7f9fb !important; box-shadow: none; border-bottom: 2px solid #4dd6ff; }
    [data-testid="stSidebar"] { background: #0b0f1a; border-right: 1px solid rgba(255,255,255,0.04); }
    .stButton button { background: #131a24; color: #c5cbd8; border: 1px solid rgba(255,255,255,0.05); border-radius: 14px; padding: 0.25rem 0.8rem; font-weight: 600; font-size: 0.9rem; transition: border-color 0.1s ease, color 0.1s ease, background 0.1s ease; height: 34px; box-shadow: none; }
    .stButton button:hover { border-color: #4dd6ff; color: #f7f9fb; background: #161f2b; }
    .stButton button:active { transform: none; box-shadow: none; }
    .stButton button[data-testid="baseButton-primary"] { background: #4dd6ff; border-color: #4dd6ff; color: #0b0f1a; }
    .stButton button[data-testid="baseButton-secondary"] { background: #131a24; color: #c5cbd8; border-color: rgba(255,255,255,0.06); }
    div[data-testid="stFileUploader"] section { border: 1px solid rgba(255,255,255,0.04); border-radius: 6px; padding: 0.55rem; background: #111624; box-shadow: none; }
    .dataframe { border-radius: 6px; overflow: hidden; box-shadow: none; border: 1px solid rgba(255,255,255,0.02); }
    .dataframe th { background: rgba(255, 255, 255, 0.015) !important; color: #dfe3ea !important; font-weight: 600 !important; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.04em; padding: 0.75rem !important; }
    .dataframe td { padding: 0.65rem 0.75rem !important; border-bottom: 1px solid rgba(255, 255, 255, 0.02) !important; color: #c5cbd8 !important; }
    .dataframe tr:hover { background: rgba(255, 255, 255, 0.01) !important; }
    .stAlert { border-radius: 6px; box-shadow: none; }
    .js-plotly-plot { border-radius: 6px; background: #111624; }
    .station-info { background: #111624; border: 1px solid rgba(255,255,255,0.02); border-radius: 6px; padding: 0.9rem 1rem; margin: 0.5rem 0 1rem 0; font-weight: 500; color: #c5cbd8; box-shadow: none; }
    .station-info-title { font-size: 1.12rem; font-weight: 650; color: #e2e8f0; margin-bottom: 0.5rem; letter-spacing: -0.01em; display: flex; align-items: center; gap: 0.4rem; }
    .station-info-detail { font-size: 0.9rem; line-height: 1.55; color: #c2c8d1; }
    .station-info-detail strong { color: #dfe3ea; font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; }
    .station-country { font-size: 0.9rem; color: #c5cbd8; font-weight: 600; margin-bottom: 0.35rem; padding: 0.22rem 0.5rem; background: rgba(255, 255, 255, 0.02); border-radius: 6px; display: inline-block; border: 1px solid rgba(255, 255, 255, 0.02); }
</style>
'''
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

SECONDARY_CSS = r'''
<style>
:root {
    --hero-title-size: clamp(1.6rem, 4.5vw, 2.4rem);
    --hero-subtitle-size: clamp(0.95rem, 3vw, 1.05rem);
}

.block-container {
        padding: clamp(0.75rem, 4vw, 1.6rem) clamp(0.9rem, 5vw, 2.4rem);
        max-width: 1300px;
        margin: 0 auto;
}

.hero-wrap {
    text-align: center;
    margin: 0 auto 1.05rem auto;
    max-width: 760px;
    padding: 0.25rem 0.75rem;
}

.hero-title {
    font-size: var(--hero-title-size);
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.25rem;
    letter-spacing: -0.01em;
    line-height: 1.08;
    word-break: break-word;
}

.hero-subtitle {
    color: #c5cbd8;
    font-size: var(--hero-subtitle-size);
    font-weight: 500;
    max-width: 620px;
    margin: 0 auto;
    line-height: 1.35;
}

.map-wrapper .js-plotly-plot {
    width: 100% !important;
}

.section-gap { height: 12px; }
.section-gap-lg { height: 24px; }
.section-gap-xl { height: 32px; }
.line-row { display: flex; gap: 12px; align-items: center; }
.flat-bar { background: #111624; border: 1px solid rgba(255,255,255,0.03); border-radius: 6px; padding: 8px 12px; }
.chip-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 8px; }
.chip-row .stButton>button { width: 100%; text-align: center; height: 32px; }
.nav-band { margin: 14px 0 14px 0; }
.map-wrapper { margin-top: 16px; }
.hairline { height: 1px; background: rgba(255,255,255,0.06); margin: 10px 0; }
.tab-guard { margin: 26px 0; }
[data-baseweb="select"] { min-height: 34px; }
[data-baseweb="select"] > div { padding-top: 2px; padding-bottom: 2px; }
.stAlert { background: #111624; border: 1px solid rgba(255,255,255,0.03); color: #c5cbd8; }

@media (max-width: 768px) {
    .clima-alert { font-size: 0.9rem; padding: 0.7rem 0.9rem; }
    .hero-wrap { margin-bottom: 1.4rem; padding: 0.35rem 0.5rem; }
    .station-info { padding: 0.9rem; }
    .station-info-title { font-size: 1.2rem; }
    .map-wrapper .js-plotly-plot { min-height: 440px !important; }
}
</style>
'''
st.markdown(SECONDARY_CSS, unsafe_allow_html=True)

# ========== SIDEBAR WITH IMPROVED UX ==========
SIDEBAR_HERO = (
    "<div style='text-align: center; margin-bottom: 1.5rem; padding: 0.9rem 0.75rem;'>"
    "<div style='font-size: 2.1rem; margin-bottom: 0.35rem;'>&#127780;</div>"
    "<h2 style='font-size: clamp(1.1rem, 3vw, 1.35rem); font-weight: 700; color:#e2e8f0; margin-bottom: 0.25rem; letter-spacing: -0.01em; word-break: break-word;'>Climate Analysis Pro</h2>"
    "<p style='color: #c5cbd8; font-size: 0.9rem; font-weight: 500;'>Weather data workspace</p>"
    "</div>"
)

with st.sidebar:
    st.markdown(SIDEBAR_HERO, unsafe_allow_html=True)
    st.divider()

    epw_loaded = bool(st.session_state.get("cdf") is not None and st.session_state.get("header"))
    current_page = st.session_state.get("active_page", DEFAULT_PAGE)
    if current_page in FROZEN_PAGES:
        current_page = DEFAULT_PAGE
    if current_page not in ALLOWED_PAGES:
        current_page = DEFAULT_PAGE
    st.session_state["active_page"] = current_page

    nav_labels = [label for label, _ in NAV_ITEMS]
    st.markdown("### Visualize weather file")
    if epw_loaded:
        current_label = PAGE_TO_LABEL.get(current_page, nav_labels[0])
        nav_choice = st.radio(
            "",
            options=nav_labels,
            index=nav_labels.index(current_label) if current_label in nav_labels else 0,
            label_visibility="collapsed",
            key="sidebar_nav",
        )
        st.markdown(
            """
            <style>
            /* Freeze roadmap items visually and functionally */
            [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(9),
            [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:nth-child(10) {
                opacity: 0.45 !important;
                pointer-events: none !important;
                cursor: not-allowed !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.caption("🚧 Short-term prediction & future climate are coming soon")
    else:
        nav_choice = "Select weather file"
        st.radio(
            "",
            options=[nav_choice],
            index=0,
            label_visibility="collapsed",
            key="sidebar_nav_locked",
        )
        locked_labels = nav_labels[1:]
        if locked_labels:
            st.markdown(
                "<div style='color:#5c6472; font-size:0.9rem;'>" +
                "<br>".join([f"• {lbl}" for lbl in locked_labels]) +
                "</div>",
                unsafe_allow_html=True,
            )
        st.info("Load a station from the map or upload an EPW/ZIP to unlock the dashboard views.")
    frozen_hit = nav_choice in FROZEN_NAV_LABELS
    nav_choice_effective = nav_choice if not frozen_hit else current_label
    chosen_page = LABEL_TO_PAGE.get(nav_choice_effective, DEFAULT_PAGE)
    st.session_state["active_page"] = chosen_page

    st.markdown("### Filters and units")
    with st.expander("Filters and units", expanded=False):
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

        st.checkbox(
            "Prefer adaptive comfort by default",
            value=st.session_state.get("prefer_adaptive_comfort", False),
            key="prefer_adaptive_comfort",
        )

        uhi_enabled = st.checkbox(
            "Apply urban heat island bias",
            value=st.session_state.get("apply_uhi_bias", False),
            key="apply_uhi_bias",
            help="Adds a uniform temperature uplift to simulate urban cores or poor nighttime ventilation."
        )
        if uhi_enabled:
            st.slider(
                "Bias magnitude (°C)",
                min_value=0.5,
                max_value=5.0,
                value=float(st.session_state.get("uhi_bias_delta", 1.5)),
                step=0.1,
                key="uhi_bias_delta",
            )
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
        if st.session_state.get("apply_uhi_bias") and "drybulb" in cdf_adjusted.columns:
            delta = float(st.session_state.get("uhi_bias_delta", 1.5))
            cdf_adjusted["drybulb"] = cdf_adjusted["drybulb"] + delta
        st.session_state.cdf = cdf_adjusted
        st.session_state.comfort_pkg = build_comfort_package(cdf_adjusted)

    if epw_loaded:
        st.markdown("### 📊 Quick Stats")
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if "drybulb" in st.session_state.cdf:
                avg_temp = st.session_state.cdf['drybulb'].mean()
                st.metric("🌡️ Avg Temp", format_temperature(avg_temp))
            if "relhum" in st.session_state.cdf:
                avg_rh = st.session_state.cdf['relhum'].mean()
                st.metric("💧 Avg Humidity", f"{avg_rh:.0f} %")
            if "windspd" in st.session_state.cdf:
                avg_wind = st.session_state.cdf['windspd'].mean()
                st.metric("💨 Avg Wind", f"{avg_wind:.1f} m/s")
            st.markdown('</div>', unsafe_allow_html=True)

# Add this to the end of your sidebar section, before the quick stats
st.divider()
st.markdown("### 🔧 Troubleshooting")

if st.button("Reset Session & Try Again"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    _rerun()

# ========== MAIN CONTENT ==========

if st.session_state.get("loading_station_name"):
    st.info(f"Loading data for {st.session_state['loading_station_name']}…")

if st.session_state.get("apply_uhi_bias") and st.session_state.get("cdf") is not None:
    st.info(
        f"Urban heat island bias of {format_temperature_delta(st.session_state.get('uhi_bias_delta', 1.5))} is currently applied to all temperature-dependent charts."
    )

ss = st.session_state
raw_epw_bytes = ss.get("raw_epw_bytes")
source_label = ss.get("source_label")

def _stage_station_and_load(station_info: dict):
    """Load station data and trigger rerun. Called after station selection."""
    st.session_state["loading_station_name"] = station_info.get("name", "selected station")
    st.session_state["sel_station"] = station_info
    st.session_state["selected_station"] = station_info
    zip_url = station_info.get("zip_url", "")
    if zip_url and ('<a ' in str(zip_url) and 'href' in str(zip_url)):
        zip_url = _extract_url(zip_url) or zip_url
    st.session_state["sel_station_url"] = zip_url
    display_label = station_info.get("display_label") or station_info.get("name") or "EPW"
    st.session_state["source_label"] = f"Station: {display_label}"
    st.session_state.pop("pending_station", None)
    st.session_state.pop("raw_epw_bytes", None)
    st.session_state["page_after_station"] = "📊 Dashboard"
    # Clear the just_loaded flag before rerun so next run doesn't trigger duplicate load
    st.session_state["just_loaded_station"] = False
    _rerun()


def render_station_picker():
    # Render station selection with quick picks, dropdown, and map.

    import plotly.express as px
    from streamlit_plotly_events import plotly_events
    header = st.session_state.get("header", {}) if isinstance(st.session_state.get("header"), dict) else {}
    location_meta = header.get("location", {}) if isinstance(header, dict) else {}

    st.write("Select a weather file to unlock the dashboard. Upload from the sidebar or pick a station below.")
    st.write("")
    st.write("")

    # ---------- Station list + map ----------
    from pathlib import Path
    import re

    stations = load_station_index()
    stations = stations.dropna(subset=["lat", "lon"]).copy()
    stations["country_disp"] = stations.get("country", pd.Series(dtype=str)).fillna("—")
    stations["elev_disp"] = pd.to_numeric(stations["elevation_m"], errors="coerce").round(0).astype("Int64")
    stations["tz_disp"] = stations["timezone"].astype(str).replace({"nan": "—"})
    stations["period_disp"] = stations.get("period", pd.Series(dtype=str)).fillna("—")
    stations["heating_disp"] = stations["heating_db"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
    stations["cooling_disp"] = stations["cooling_db"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
    stations["source_url"] = stations.get("zip_url", pd.Series(dtype=str)).fillna("")

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
            pass
        return code.upper()

    def _parse_from_raw(row: pd.Series) -> pd.Series:
        raw_id = str(row.get("raw_id") or "").strip()
        if not raw_id:
            zip_url = str(row.get("zip_url") or "").strip()
            raw_id = Path(zip_url).stem if zip_url else ""

        base = raw_id
        source_token = str(row.get("source") or "").strip()
        if "_" in raw_id:
            base, maybe_src = raw_id.rsplit("_", 1)
            if maybe_src and not source_token:
                source_token = maybe_src

        country_iso3 = str(row.get("country_iso3") or "").strip().upper()
        state_code = str(row.get("state_code") or "").strip()
        city_raw = str(row.get("city_raw") or "").strip()
        station_id = str(row.get("station_id") or "").strip()
        period = str(row.get("period") or "").strip()

        if base:
            parts = base.split("_", 2)
            if len(parts) >= 1 and not country_iso3:
                country_iso3 = parts[0].upper()
            if len(parts) >= 2 and not state_code:
                state_code = parts[1]
            if len(parts) == 3 and not city_raw:
                city_raw = parts[2].split(".", 1)[0]

            tail = base.rsplit(".", 1)
            if len(tail) == 2:
                station_id = station_id or tail[1]

        city_name = city_raw.replace("-", " ").replace(".", " ")
        city_name = re.sub(r"\s+", " ", city_name).strip().title()

        country_name = str(row.get("country_name") or row.get("country") or "").strip()
        if not country_name and country_iso3:
            country_name = _country_name_from_iso3(country_iso3)

        row["raw_id"] = raw_id
        row["country_iso3"] = country_iso3
        row["state_code"] = state_code
        row["city_raw"] = city_raw
        row["city_name"] = city_name
        row["station_id"] = station_id
        row["source"] = source_token
        row["period"] = period
        row["country_name"] = country_name
        return row

    stations = stations.apply(_parse_from_raw, axis=1)

    # Clean label pieces to avoid 'nan'
    for col in ["country_name", "state_code", "source", "period", "city_name"]:
        if col not in stations.columns:
            stations[col] = ""
        stations[col] = stations[col].fillna("").astype(str)

    def make_label(row: pd.Series) -> str:
        location_bits = []
        if row.get("city_name"):
            location_bits.append(row.get("city_name"))
        if row.get("state_code"):
            location_bits.append(row.get("state_code"))
        if row.get("country_name"):
            location_bits.append(row.get("country_name"))
        location_str = ", ".join([b for b in location_bits if b])

        meta_bits = []
        if row.get("station_id"):
            meta_bits.append(f"WMO {row.get('station_id')}")
        if row.get("source"):
            meta_bits.append(row.get("source"))
        if row.get("period"):
            meta_bits.append(row.get("period"))
        meta_str = ", ".join([b for b in meta_bits if b])

        if meta_str:
            return f"{location_str} ({meta_str})" if location_str else meta_str
        return location_str or (row.get("raw_id") or "Unknown station")

    stations["display_label"] = stations.apply(make_label, axis=1)

    # ========== INTERACTIVE MAP (full-width card) ==========
    # Map renders immediately on page load - appears FIRST
    with st.container(border=True):
        st.markdown("### 🗺️ Interactive Map")
        st.caption("Click a station dot to load it instantly.")
        
        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

    MAP_HEIGHT = 700
    fig_map = go.Figure(
        data=[
            go.Scattermapbox(
                lat=stations["lat"].tolist(),
                lon=stations["lon"].tolist(),
                mode="markers",
                marker=dict(size=11, color="#5fd4ff", opacity=0.9),
                text=stations["name"].tolist(),
                customdata=stations[["name", "country", "lat", "lon", "elevation_m", "timezone", "zip_url", "period", "heating_db", "cooling_db"]].values.tolist(),
                hoverinfo="text",
                hovertemplate="%{text}",
            )
        ]
    )

    fig_map.update_layout(
        mapbox=dict(
            style="open-street-map",
            bearing=0,
            pitch=0,
            center=dict(lat=float(stations["lat"].median()), lon=float(stations["lon"].median())),
            zoom=2.2,
        ),
        dragmode="pan",
        margin=dict(l=0, r=0, t=12, b=0),
        height=MAP_HEIGHT,
        paper_bgcolor="#0b0f1a",
        plot_bgcolor="#0b0f1a",
        hovermode="closest",
        clickmode="event+select",
        showlegend=False,
        uirevision="north_up",
    )

    # plotly_events() renders the map internally - no need for separate st.plotly_chart()
    selected_points = plotly_events(
        fig_map,
        click_event=True,
        hover_event=False,
        select_event=False,
        override_height=MAP_HEIGHT,
        override_width=None,
        key="map_click_v4"
    )

    # Process map click events - only load on NEW selections
    if selected_points and len(selected_points) > 0:
        point = selected_points[0]
        if "pointIndex" in point:
            idx = point["pointIndex"]
            if 0 <= idx < len(stations):
                row = stations.iloc[idx]
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
                }
                # Get station ID to track if already loaded
                station_id = row.get("station_id") or row.get("raw_id") or str(idx)
                last_loaded_id = st.session_state.get("last_loaded_station_id")
                just_loaded = st.session_state.get("just_loaded_station", False)
                
                # Only load if this is a NEW station selection AND we haven't just loaded
                # This prevents duplicate loads on reruns
                if station_id != last_loaded_id and not just_loaded:
                    # Mark that we're about to load to prevent rerun loops
                    st.session_state["last_loaded_station_id"] = station_id
                    st.session_state["just_loaded_station"] = True
                    st.session_state["selected_station"] = station_info
                    st.session_state["pending_station"] = station_info
                    
                    # Auto-load station immediately with spinner
                    station_name_display = station_info.get("display_label") or station_info.get("name", "Unknown")
                    with st.spinner(f"Loading station {station_name_display}..."):
                        try:
                            _stage_station_and_load(station_info)
                            display_name = station_info.get("display_label") or station_info.get("name", "Unknown")
                            station_id_display = station_info.get("station_id") or station_id
                            st.success(f"✅ Loaded **{display_name}** ({station_id_display})")
                        except Exception as e:
                            st.error(f"❌ Failed to load station: {str(e)}")
                            # On error, allow retry by clearing the loaded ID
                            st.session_state.pop("last_loaded_station_id", None)
                            st.session_state["just_loaded_station"] = False
                elif station_id == last_loaded_id:
                    # Station already loaded - clear the just_loaded flag if it was set
                    st.session_state["just_loaded_station"] = False

    # Removed sticky confirmation bar - stations load automatically on map click

    st.divider()
    
    # ========== STATION SEARCH ==========
    st.markdown("### 🔍 Station Search")
    st.caption(f"{len(stations):,} verified download links. Search, then load once.")
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

    search_col, load_col = st.columns([3, 1])
    with search_col:
        search_query = st.text_input(
            "Search by station name, country, or period",
            key="station_selector_query",
            help="Type a city, ISO3 code, WMO station ID, source, or year range. Top 25 matches shown.",
            placeholder="e.g., Paris FRA 2021 TMYx"
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

        if st.button("📍 Load Selected Station", type="primary", use_container_width=True):
            if chosen_row is not None:
                st.session_state["selected_station"] = chosen_row.to_dict()
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
                }
                _stage_station_and_load(station_info)


def handle_epw_upload(uploaded_file, picker_key: str = "sidebar") -> Optional[bytes]:
    """Process an uploaded EPW/ZIP and persist to session state.

    picker_key differentiates selectbox keys between sidebar/main uploads.
    Returns the raw EPW bytes or None if selection failed.
    """
    if uploaded_file is None:
        return None

    try:
        uploaded_file.seek(0)
    except Exception:
        pass

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
    st.session_state["page_after_station"] = "📊 Dashboard"
    st.session_state.pop("loading_station_name", None)
    return raw_epw_bytes

@CACHE(show_spinner=False)
def fetch_epw_bytes(url: str) -> Optional[bytes]:
    import requests, io, zipfile
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; ClimateAnalysisPro/1.0)",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate",
        }
        
        # Add timeout and better error handling
        r = requests.get(url, headers=headers, timeout=30, stream=True)
        r.raise_for_status()
        
        # Check content type and size
        content_type = r.headers.get('content-type', '').lower()
        content_length = r.headers.get('content-length')
        
        if content_length and int(content_length) > 100_000_000:  # 100MB limit
            st.warning(f"File too large: {content_length} bytes")
            return None
            
        content = r.content

        # If it's a ZIP, extract the largest EPW inside
        if url.lower().endswith(".zip") or zipfile.is_zipfile(io.BytesIO(content)):
            with zipfile.ZipFile(io.BytesIO(content), "r") as z:
                epws = [m for m in z.namelist() if m.lower().endswith(".epw")]
                if not epws:
                    st.warning("ZIP file contains no EPW files")
                    return None
                epws.sort(key=lambda m: z.getinfo(m).file_size, reverse=True)
                with z.open(epws[0]) as f:
                    return f.read()
        return content
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {str(e)}")
        return None
    except zipfile.BadZipFile:
        st.error("Downloaded file is not a valid ZIP file")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None


# Alternative EPW sources as fallbacks
ALTERNATIVE_EPW_SOURCES = [
    # Keep a minimal, general-purpose fallback list for manual selections
    "https://energyplus-weather.s3.amazonaws.com/north_america_wmo_region_4/USA/NY/Buffalo/Buffalo_Greater_International_AP_725280_TMY3.epw",
    "https://energyplus-weather.s3.amazonaws.com/north_america_wmo_region_4/USA/AZ/Phoenix/Phoenix_Sky_Harbor_Intl_Airport_722780_TMY3.epw",
    "https://energyplus-weather.s3.amazonaws.com/north_america_wmo_region_4/USA/IL/Chicago/Chicago_OHare_Intl_Airport_725300_TMY3.epw",
    "https://energyplus-weather.s3.amazonaws.com/north_america_wmo_region_4/USA/FL/Miami/Miami_Intl_Airport_722020_TMY3.epw",
]

def try_multiple_sources(sources: List[str]) -> Tuple[Optional[bytes], Optional[str]]:
    for url in sources:
        epw_bytes = fetch_epw_bytes(url)
        if epw_bytes is not None:
            return epw_bytes, url
    return None, None


active_page = st.session_state.get("active_page", DEFAULT_PAGE)
if active_page not in ALLOWED_PAGES:
    active_page = DEFAULT_PAGE
    st.session_state["active_page"] = active_page

main_upload = None
if active_page == "Select weather file":
    st.info("Load a station from the map or upload an EPW/ZIP to unlock the dashboard views.")
    # Clear just_loaded flag at start of page render to allow new selections after rerun
    if st.session_state.get("just_loaded_station") is True:
        # Reset flag after page renders once after load
        st.session_state["just_loaded_station"] = False
    render_station_picker()
    main_upload = st.file_uploader(
        "Upload EPW or ZIP file",
        type=["epw", "zip"],
        help="Upload an EnergyPlus Weather file or a ZIP containing EPWs",
        key="main_epw_upload_primary",
    )
else:
    st.session_state.pop("pending_station", None)
    st.session_state.pop("just_loaded_station", None)

if main_upload is not None:
    # ---- Uploader path ----
    raw_epw_bytes = handle_epw_upload(main_upload, picker_key="main")

elif ss.get("sel_station_url"):
    url = ss["sel_station_url"]
    alt_urls = ss.get("sel_station_alt_urls", [])
    station_name = ss.get("sel_station", {}).get("name", "selected station")

    # If the URL is an HTML anchor, extract the href
    if '<a ' in str(url) and 'href' in str(url):
        url = _extract_url(url) or url

    # Generate alternative URLs to try (station-specific first)
    urls_to_try = fix_station_url(url)
    urls_to_try.extend(alt_urls)
    attempted_urls = list(urls_to_try)

    success = False
    successful_url = None
    for i, test_url in enumerate(urls_to_try):
        url_clean = re.search(r'https.*?\.epw', test_url)
        if url_clean:
            test_url = url_clean.group()
        with st.spinner(f"⏳ Trying source {i+1}/{len(urls_to_try)} for {station_name}..."):
            raw_epw_bytes = fetch_epw_bytes(test_url)
            if raw_epw_bytes is not None:
                success = True
                successful_url = test_url
                break

    # Auto-fallback to global alternates if station-specific URLs failed
    if not success:
        for j, alt_url in enumerate(ALTERNATIVE_EPW_SOURCES, start=1):
            st.info(f"🔁 {station_name}: primary links unavailable — trying alternate source {j}/{len(ALTERNATIVE_EPW_SOURCES)}")
            attempted_urls.append(alt_url)
            raw_epw_bytes = fetch_epw_bytes(alt_url)
            if raw_epw_bytes is not None:
                success = True
                successful_url = alt_url
                break

    if not success:
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
        
        # Enhanced debug information
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
        
        # Recovery options
        st.markdown("### Recovery Options")
        col1, col2 = st.columns(2)
        
        if col1.button("Clear & Start Over", use_container_width=True):
            ss.pop("sel_station_url", None)
            ss.pop("sel_station", None)
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
        
        st.stop()
    
    # Success case
    source_label = f"Station: {station_name}" if successful_url == url else f"Station: {station_name} (alt)"
    ss["raw_epw_bytes"] = raw_epw_bytes
    ss["source_label"] = source_label
    ss.pop("sel_station_url", None)


elif raw_epw_bytes is None:
    pass


@CACHE(show_spinner=False)
def read_epw_with_schema_cached(epw_bytes: Union[bytes, Path]):
    return read_epw_with_schema(epw_bytes)

@CACHE(show_spinner=False)
def build_clima_dataframe_cached(df: pd.DataFrame) -> pd.DataFrame:
    return build_clima_dataframe(df)


if raw_epw_bytes is not None:
    # Ingest → Clima DF
    with st.spinner("Parsing EPW and building climate tables…"):
        header, df, epw_notes = read_epw_with_schema_cached(raw_epw_bytes)
        cdf = build_clima_dataframe_cached(df)

    # ========== STORE IN SESSION STATE ==========
    st.session_state.header = header
    st.session_state.cdf = cdf
    st.session_state.df = df
    st.session_state.comfort_pkg = build_comfort_package(cdf)
    st.session_state.cdf_raw = cdf.copy(deep=True)

    location_meta = header.get("location", {}) if isinstance(header, dict) else {}
    city = location_meta.get("city") or location_meta.get("state_province") or "Unknown"
    country = location_meta.get("country", "")
    period = location_meta.get("period") or location_meta.get("data_periods") or "—"
    domain = location_meta.get("source", "EPW")
    record_count = len(df)

    summary_label = f"✅ Loaded EPW for {city}, {country} — {period} ({record_count:,} hours)"
    st.success(summary_label)
    if hasattr(st, "toast"):
        st.toast("EPW loaded successfully ✅", icon="✅")
    if source_label and source_label not in summary_label:
        st.caption(f"Source: {source_label}")
    st.session_state.pop("loading_station_name", None)

    if epw_notes:
        for note in epw_notes:
            st.warning(f"EPW file note: {note}")

    with st.expander("EPW metadata", expanded=False):
        meta_rows = [
            ("Location", f"{city}, {country}".strip().strip(',')),
            ("Source", domain or "—"),
            ("WMO", location_meta.get("wmo", "—")),
            ("Elevation (m)", location_meta.get("elevation_m", "—")),
            ("Timezone", location_meta.get("timezone", "—")),
            ("Period", period),
            ("Records", f"{record_count:,}"),
        ]
        meta_df = pd.DataFrame(meta_rows, columns=["Field", "Value"])
        st.table(meta_df)

    target_page = st.session_state.pop("page_after_station", None)
    if target_page:
        st.session_state["active_page"] = target_page
        _rerun()


cdf = st.session_state.get("cdf")

if cdf is None and st.session_state.get("active_page") != DEFAULT_PAGE:
    st.session_state["active_page"] = DEFAULT_PAGE

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
        "pressure": "atmos_pressure",
        "atmospheric_pressure": "atmos_pressure",
    }
    for src, dest in alias_columns.items():
        if dest not in cdf.columns and src in cdf.columns:
            cdf[dest] = cdf[src]

# Apply any queued navigation changes before the nav renders
if "page_after_station" in st.session_state:
    target_page = st.session_state.pop("page_after_station")
    st.session_state["active_page"] = target_page

page = st.session_state.get("active_page", DEFAULT_PAGE)
if page in FROZEN_PAGES:
    page = DEFAULT_PAGE
if page not in ALLOWED_PAGES:
    page = DEFAULT_PAGE
st.session_state["active_page"] = page
effective_page = page

# ========== HEATMAP HELPERS (ANNUAL DIURNAL RESOURCE) ==========

def find_column_by_fuzzy_match(df: pd.DataFrame, keywords: List[str], exclude_cols: List[str] = None) -> Optional[str]:
    """Find a column in df by fuzzy matching against keywords (case-insensitive, substring)."""
    if exclude_cols is None:
        exclude_cols = []
    col_lower = [c.lower() for c in df.columns if c not in exclude_cols]
    for keyword in keywords:
        kw_lower = keyword.lower()
        for col in col_lower:
            if kw_lower in col or col in kw_lower:
                # Return the original column name (not lowercase version)
                return df.columns[col_lower.index(col)]
    return None


def get_metric_column(df: pd.DataFrame, keywords: List[str], exclude_cols: List[str] = None) -> Optional[str]:
    """Alias for find_column_by_fuzzy_match."""
    return find_column_by_fuzzy_match(df, keywords, exclude_cols)


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
    
    if any(x in metric_lower for x in ["temp", "dry_bulb"]):
        colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c", "#8b0000"]
        labels = ["<10°C", "10–20°C", "20–26°C", "26–35°C", ">35°C"]
    elif any(x in metric_lower for x in ["radiation", "solar", "ghi", "global"]):
        colors = ["#1a1a2e", "#3498db", "#f39c12", "#e74c3c", "#fff700"]
        labels = ["<100", "100–300", "300–500", "500–700", ">700"]
    elif any(x in metric_lower for x in ["humidity", "rh", "relhum"]):
        colors = ["#e74c3c", "#f39c12", "#3498db", "#1a472a"]
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
    
    heatmap_dict: {"metric_name": (pivot_binned, info_dict), ...}
    """
    if not heatmap_dict or all("error" in v[1] for v in heatmap_dict.values()):
        return None
    
    # Filter out metrics with errors
    valid_strips = {k: v for k, v in heatmap_dict.items() if "error" not in v[1]}
    if not valid_strips:
        return None
    
    n_strips = len(valid_strips)
    fig = make_subplots(
        rows=n_strips, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[f"{name}" for name, (_, info) in valid_strips.items()],
    )
    
    month_doy, month_letters = month_labels_at_midpoints(leap_year=False)
    month_boundaries = month_boundaries_doy(leap_year=False)
    
    for row, (strip_name, (pivot_binned, info)) in enumerate(valid_strips.items(), start=1):
        if pivot_binned.empty:
            continue
        
        # Prefer labels/colors supplied in info (for editable thresholds); fallback to defaults
        colors_default, labels_default = get_color_scale_for_metric(info["metric"])
        labels = info.get("labels", labels_default)
        colors = info.get("colors", colors_default[: len(labels)])
        
        hover_labels = info.get("hover_labels")
        customdata = hover_labels if hover_labels is not None else None
        hovertemplate = (
            "DoY: %{x}<br>HoD: %{y}<br>Value: %{z:.2f}<br>Band: %{customdata}<extra></extra>"
            if customdata is not None
            else "DoY: %{x}<br>HoD: %{y}<br>Value: %{z:.2f}<extra></extra>"
        )

        trace = go.Heatmap(
            z=pivot_binned.values,
            x=pivot_binned.columns,  # DOY 1..366
            y=pivot_binned.index,     # HOD 0..23
            colorscale=list(zip([i / (len(colors) - 1) for i in range(len(colors))], colors)),
            showscale=False,
            customdata=customdata,
            hovertemplate=hovertemplate,
        )
        fig.add_trace(trace, row=row, col=1)
        
        # Configure y-axis (HOD)
        fig.update_yaxes(
            title_text="Time of Day",
            tickmode="array",
            tickvals=[0, 12, 23],
            ticktext=["12:00am", "noon", "11:59pm"],
            row=row, col=1,
        )
        # Add vertical month boundaries (per-row so lines align with each subplot)
        for doy_boundary in month_boundaries[1:-1]:  # Skip year start/end
            fig.add_vline(x=doy_boundary, line_dash="solid", line_color="rgba(255, 255, 255, 0.3)", line_width=1, row=row, col=1)
    
    # Configure bottom x-axis to show month letters only on the shared axis
    fig.update_xaxes(
        title_text="Month",
        tickmode="array",
        tickvals=month_doy,
        ticktext=month_letters,
        row=n_strips, col=1,
        tickfont=dict(size=11),
    )

    # Add per-strip discrete legend swatches as shapes + annotations placed in paper coordinates
    # Reserve space on the right by increasing right margin
    legend_x0 = 1.02
    legend_x1 = 1.06
    legend_label_x = 1.08
    fig.update_layout(margin=dict(l=80, r=200, t=80, b=60))

    # For each strip, draw legend boxes at appropriate paper y positions
    for idx, (strip_name, (pivot_binned, info)) in enumerate(valid_strips.items(), start=1):
        colors, labels = get_color_scale_for_metric(info["metric"])
        if info.get("labels"):
            labels = info["labels"]
        if info.get("colors"):
            colors = info["colors"]
        # Determine y-domain for this subplot
        axis_key = "yaxis" if idx == 1 else f"yaxis{idx}"
        try:
            domain = fig.layout[axis_key].domain
        except Exception:
            domain = None
        if not domain:
            continue
        y_low, y_high = domain[0], domain[1]
        center = (y_low + y_high) / 2.0
        n_labels = len(labels)
        height_total = (y_high - y_low) * 0.8
        height_per = min(0.03, height_total / max(1, n_labels))
        # stack labels vertically centered at `center`
        for i, (col_hex, lbl) in enumerate(zip(colors, labels)):
            y0 = center + ( (n_labels - 1) / 2.0 - i) * height_per - (height_per / 2.0)
            y1 = y0 + height_per * 0.9
            # Add rectangle shape
            fig.add_shape(type="rect", xref="paper", x0=legend_x0, x1=legend_x1, yref="paper", y0=y0, y1=y1, fillcolor=col_hex, line=dict(width=0))
            # Add label annotation
            fig.add_annotation(x=legend_label_x, y=(y0 + y1) / 2.0, xref="paper", yref="paper",
                               text=lbl, showarrow=False, align="left", font=dict(size=10))

    fig.update_layout(
        height=220 * n_strips,
        showlegend=False,
        title_text="Annual Diurnal Resource Heatmaps",
        title_x=0.5,
        font=dict(size=10),
    )
    
    return fig


# ========== MAIN TABS WITH IMPROVED ORGANIZATION ==========
if cdf is not None:
    if effective_page == "📊 Dashboard":
        import plotly.express as px
        st.markdown("### 📊 Dashboard")
        st.caption(
            "Start here for a quick health check: tabs bundle location context, thermal comfort, "
            "and data quality so you can understand what the climate looks like and whether the EPW "
            "inputs are trustworthy before diving deeper."
        )

        loc = header["location"]
        overview_tab, comfort_tab, diagnostics_tab, heatmaps_tab = st.tabs([
            "Overview & Stats",
            "Comfort & Loads",
            "Data Quality",
            "Heatmaps",
        ])

        with overview_tab:
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
            seasonal_df = seasonal_df.applymap(lambda v: "—" if pd.isna(v) else (f"{v:.1f}" if isinstance(v, float) else v))
            st.markdown("#### Seasonal snapshot")
            st.table(seasonal_df)

            try:
                tmin, tmax = cdf.index.min(), cdf.index.max()
                n_hours = len(cdf)
                st.caption(
                    f"Data window: **{tmin:%b %d, %Y} — {tmax:%b %d, %Y}**  ·  Records: **{n_hours:,}** hours"
                )
            except Exception:
                pass

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

        with comfort_tab:
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
                    freq="A",
                    comfort_band=comfort_band,
                    adaptive_band=adaptive_band,
                    overheating_thresholds=hot_thresholds,
                    cold_thresholds=cold_thresholds,
                    percentiles=percentiles,
                    occupancy_mask=occupancy_mask,
                )
                if comfort_dyn_a is not None and not comfort_dyn_a.empty:
                    comfort_annual = comfort_dyn_a
            except Exception:
                pass

            try:
                comfort_dyn_m = ce.summarize_comfort(
                    cdf,
                    di_series,
                    utci_series,
                    freq="M",
                    comfort_band=comfort_band,
                    adaptive_band=adaptive_band,
                    overheating_thresholds=hot_thresholds,
                    cold_thresholds=cold_thresholds,
                    percentiles=percentiles,
                    occupancy_mask=occupancy_mask,
                )
                if comfort_dyn_m is not None and not comfort_dyn_m.empty:
                    comfort_monthly = comfort_dyn_m
            except Exception:
                pass

            def _fmt_hours(val: float) -> str:
                return "—" if pd.isna(val) else f"{float(val):.0f} h"

            def _fmt_value(val: float, suffix: str = "") -> str:
                return "—" if pd.isna(val) else f"{float(val):.0f}{suffix}"

            if comfort_annual is None or comfort_annual.empty:
                st.info("Comfort insights unlock automatically when dry-bulb, humidity, and wind speed data are available.")
            else:
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
                    st.plotly_chart(fig_comfort, use_container_width=True)

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
                        )
                        chosen_hour = st.slider("Hour (0–23)", 0, 23, 14)
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

        with diagnostics_tab:
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

            # Diagnostics tab: keep Data Quality content only (heatmaps moved to the Heatmaps tab)
            st.caption("Data quality diagnostics shown above. Use the Heatmaps tab to explore annual diurnal resource heatmaps.")

        # ========== HEATMAPS TAB CONTENT ==========
        with heatmaps_tab:
            st.divider()
            st.subheader("Annual Diurnal Resource Heatmaps")
            st.caption("Adjust legend thresholds to explore how different performance ranges appear across the year.")

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
                    st.experimental_rerun()

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
                # Continuous heatmap helper: aggregate first (mean/median), keep thresholds for hover/legend only
                def _build_pivot_with_thresholds(df: pd.DataFrame, col: str, metric_label: str, thresholds: list[float], units_suffix: str, palette_metric: str, agg: str = "mean"):
                    series = pd.to_numeric(df[col], errors="coerce")
                    series = series.dropna()
                    if series.empty:
                        return pd.DataFrame(), {"error": f"No valid data for {metric_label}"}

                    work = pd.DataFrame({"val": series})
                    work["hod"] = work.index.hour
                    work["doy"] = work.index.dayofyear

                    aggfunc = np.median if agg == "median" else "mean"
                    pivot_raw = work.pivot_table(index="hod", columns="doy", values="val", aggfunc=aggfunc)
                    pivot_raw = pivot_raw.reindex(index=range(24), columns=range(1, 367))

                    # Thresholds for interpretation (hover/legend), not for coloring
                    bins = [-np.inf] + thresholds + [np.inf]
                    labels = _labels_from_thresholds(thresholds, units_suffix)
                    cat = pd.cut(pivot_raw.values.flatten(), bins=bins, labels=labels, right=False)
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
                solar_col = get_metric_column(cdf, ["ghi", "global_horiz", "global_horizontal", "solar", "radiation"])
                if solar_col:
                    pivot_binned, info = _build_pivot_with_thresholds(cdf, solar_col, "Solar Radiation", thresholds_state["solar"], "", "solar", agg="mean")
                    if not pivot_binned.empty:
                        heatmap_dict["Solar Radiation"] = (pivot_binned, info)

                rh_col = get_metric_column(cdf, ["relative_humidity", "relhum", "rh"])
                if rh_col:
                    pivot_binned, info = _build_pivot_with_thresholds(cdf, rh_col, "Humidity", thresholds_state["humidity"], "%", "humidity", agg="mean")
                    if not pivot_binned.empty:
                        heatmap_dict["Humidity"] = (pivot_binned, info)

                wind_col = get_metric_column(cdf, ["wind_speed", "windspd", "wspd", "wind"])
                if wind_col:
                    pivot_binned, info = _build_pivot_with_thresholds(cdf, wind_col, "Wind Speed", thresholds_state["wind"], " m/s", "wind", agg="median")
                    if not pivot_binned.empty:
                        heatmap_dict["Wind Speed"] = (pivot_binned, info)

                if not heatmap_dict:
                    st.info("No metric data available for heatmap generation.")
                else:
                    fig = build_diurnal_heatmap_figure(heatmap_dict, cdf, header)
                    if fig:
                        # Full-year ticks
                        month_doy, month_letters = month_labels_at_midpoints(leap_year=False)
                        fig.update_xaxes(tickmode="array", tickvals=month_doy, ticktext=month_letters)
                        fig.update_xaxes(title_text=None)
                        try:
                            fig.update_yaxes(title_text=None)
                        except Exception:
                            pass
                        fig.add_annotation(x=-0.06, y=0.5, xref='paper', yref='paper', text='Time of Day', showarrow=False, textangle=-90, font=dict(size=11))

                        st.plotly_chart(fig, use_container_width=True)

                        # Downloads reflecting current thresholds
                        c1d, c2d = st.columns(2)
                        with c1d:
                            try:
                                png_bytes = fig.to_image(format="png", scale=2)
                                st.download_button(label="📥 Download heatmaps as PNG", data=png_bytes, file_name="diurnal_heatmaps.png", mime="image/png")
                            except Exception:
                                html_bytes = fig.to_html(include_plotlyjs='cdn').encode('utf-8')
                                st.download_button(label="📥 Download heatmaps (HTML)", data=html_bytes, file_name="diurnal_heatmaps.html", mime="text/html")

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
                                    bins = [-np.inf] + info.get("thresholds", []) + [np.inf]
                                    labels = info.get("labels", [])
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
                                    st.download_button(label="📥 Download heatmap data (CSV)", data=csv_bytes, file_name="diurnal_heatmaps_data.csv", mime="text/csv")
                            except Exception as e:
                                st.caption(f"CSV export failed: {str(e)[:80]}")
                    else:
                        st.warning("Could not generate heatmap figure from available data.")

# ====================== TEMPERATURE & HUMIDITY (CLEAN) ======================
if effective_page in ("🌡️ Temperature & Humidity", "Temp & Humidity"):
    st.markdown("### Temperature & Humidity")
    st.caption("Clean reference plots with comfort ribbons and a single linked time window. Use this space to compare how temperature and humidity evolve at hourly, daily, or monthly scales.")
    # -------------------- Controls --------------------
    c1, c2, c3 = st.columns([1.2, 1, 1.2])
    agg = c1.selectbox("Aggregation", ["Hourly", "Daily", "Monthly"], index=1)
    smooth_n = c2.slider("Smoothing (periods)", 1, 15, 7, help="Rolling mean applied after aggregation.")
    rh_lo, rh_hi = c3.slider("RH comfort band (%)", 0, 100, (30, 70))
    c4, c5 = st.columns([1, 1])
    temp_band = c4.selectbox("Temperature comfort band", ["None", "ASHRAE 80%", "ASHRAE 80% + 90%"], index=1)
    show_temp_range = c5.checkbox("Show temperature range", True)


    # -------------------- Helpers --------------------
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    def resample_mean_range(series: pd.Series, g: str):
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
        # --- build a complete time index and reindex everything onto it
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
        if smooth_n and smooth_n > 1:
            s_mean = s_mean.rolling(smooth_n, min_periods=1, center=True).mean()
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
        Trm = daily.ewm(alpha=0.2, adjust=False).mean()
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
    T_mean,  T_min,  T_max  = resample_mean_range(T_hourly, agg)
    RH_mean, RH_min, RH_max = resample_mean_range(RH_hourly, agg)
    # -------- Hover customdata for temperature (max, min, mean) --------
    temp_hover_cdata = np.c_[T_max.values, T_min.values, T_mean.values]

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

    # Hover customdata for mean lines: [min, mean, max, xlabel]
    def _hover_customdata(s_min, s_mean, s_max, xlabels):
        return np.column_stack([s_min.values, s_mean.values, s_max.values, xlabels])

    cd_T  = _hover_customdata(T_min,  T_mean,  T_max,  xlab_T)
    cd_RH = _hover_customdata(RH_min, RH_mean, RH_max, xlab_RH)

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
        hovertemplate="<b>%{customdata[3]}</b><br>Avg: %{customdata[1]:.2f} °C<br>Max: %{customdata[2]:.2f} °C<br>Min: %{customdata[0]:.2f} °C<extra></extra>"
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
        hovertemplate="<b>%{customdata[3]}</b><br>Avg: %{customdata[1]:.2f} %<br>Max: %{customdata[2]:.2f} %<br>Min: %{customdata[0]:.2f} %<extra></extra>"
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
            ticklabelstandoff=6, ticks="outside", ticklen=6,
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
        row=2, col=1
    )


    # top y (T)
    fig.update_yaxes(
        title="Dry-bulb temperature (°C)",
        title_standoff=24,            # a little more space from ticks
        automargin=True,              # let Plotly grow the left margin if needed
        showticklabels=True,          # make sure labels are drawn
        tickfont=dict(size=12, color="rgba(240,240,240,0.96)"),  # visible on dark bg
        ticks="outside", ticklen=6, ticklabelstandoff=8,
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
        ticks="outside", ticklen=6, ticklabelstandoff=8,
        showgrid=True, gridcolor="rgba(255,255,255,0.08)",
        showline=True, linecolor="rgba(255,255,255,0.38)", linewidth=1.1,
        range=[0, 100], dtick=10,
        row=2, col=1
    )

    # Reduce excess padding so the two panels read as a single, compact stack
    fig.update_layout(
        margin=dict(l=70, r=30, t=45, b=40),
        plot_bgcolor="rgba(12, 17, 26, 1)",
        paper_bgcolor="rgba(12, 17, 26, 1)",
        legend=dict(font=dict(color="#e5e7eb", size=12)),
        hoverlabel=dict(bgcolor="#0f172a", font=dict(color="#e5e7eb"))
    )

    # Keep a single axis in the slider preview to ensure both traces render together
    # -------------------- Legend ordering (exact order requested) --------------------

    # ---------- FORCE LEGEND ORDER (must be RIGHT BEFORE plotly_chart) ----------
    # -------------------- FORCE LEGEND ORDER (bulletproof) --------------------
    desired_order = [
        ("Average Dry bulb temperature", dict(mode="lines", line=dict(width=2.8, color="#ff5c52"))),
        ("Dry bulb temperature range",   dict(mode="markers", marker=dict(size=10, color="#E74C3C"))),
        ("Below comfort (T)",            dict(mode="lines", line=dict(width=2, color="rgba(52, 152, 219, 0.78)"))),
        ("Within comfort (T)",           dict(mode="lines", line=dict(width=2, color="rgba(46, 204, 113, 0.82)"))),
        ("Above comfort (T)",            dict(mode="lines", line=dict(width=2, color="rgba(231, 76, 60, 0.85)"))),
        ("ASHRAE adaptive comfort (80%)",dict(mode="lines", line=dict(width=2, color="rgba(46, 204, 113, 0.9)"))),
        ("Average Relative humidity",    dict(mode="lines", line=dict(width=2.8, color="#7cc7ff"))),
        ("Below comfort (RH)",           dict(mode="lines", line=dict(width=2, color="rgba(174, 214, 241, 0.74)"))),
        ("Within comfort (RH)",          dict(mode="lines", line=dict(width=2, color="rgba(93, 173, 226, 0.82)"))),
        ("Above comfort (RH)",           dict(mode="lines", line=dict(width=2, color="rgba(27, 79, 114, 0.82)"))),
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


    st.plotly_chart(fig, use_container_width=True)

    # ==================== HUMIDITY BAR CHART ====================
    if effective_page in ("RH Bars", "Temp & Humidity", "🌡️ Temperature & Humidity"):
        st.markdown("---")
        st.markdown("#### Humidity — bar chart")
        if "relhum" not in cdf:
            st.info("This EPW has no Relative Humidity column.")
        else:
            import plotly.express as px
            c1, c2 = st.columns([1.2, 1])
            st.caption("Monthly view only to keep the chart readable.")
            stat = c2.selectbox("Statistic", ["Mean", "Median"], index=0, key=f"rh_stat_{effective_page}")
            rh = cdf[["relhum"]].dropna().copy()
            rh["month"] = rh.index.month
            monthly = rh.groupby("month")["relhum"]
            y = monthly.mean() if stat == "Mean" else monthly.median()
            df_bar = y.reset_index().rename(columns={"relhum": "RH"})
            df_bar["month_name"] = df_bar["month"].apply(lambda m: pd.Timestamp(2001, int(m), 1).strftime("%b"))
            fig_rh = px.bar(
                df_bar, x="month_name", y="RH",
                labels={"month_name": "Month", "RH": "Relative Humidity (%)"},
                title=f"Monthly {stat.lower()} RH"
            )
            fig_rh.update_xaxes(type="category")
            # Consistent styling
            fig_rh.update_traces(marker_line_width=0.2, opacity=0.9)
            fig_rh.update_layout(
                yaxis=dict(range=[0, 100], title="Relative Humidity (%)"),
                margin=dict(l=40, r=20, t=60, b=30),
                legend=dict(orientation="h", x=0, y=1.02),
                bargap=0.15
            )
            st.plotly_chart(fig_rh, use_container_width=True)

    col = "drybulb"  # base variable for downstream charts

    if effective_page in ("Temp & Humidity", "🌡️ Temperature & Humidity", "Daily Scatter"):
        # ==================== DAILY SCATTER (FACETS) ====================
        label_y    = "Dry-bulb temperature (°C)" if col == "drybulb" else "Relative Humidity (%)"
        line_color = "crimson" if col == "drybulb" else "dodgerblue"
        st.markdown("---")
        st.markdown("#### Daily scatter (hourly points, faceted by month)")
        st.caption(
            "Each panel bundles all hours for a given month so you can spot diurnal swings, "
            "shoulder-season variability, and whether the adaptive comfort envelope is mostly "
            "above or below the bias-adjusted temperature line."
        )
        # ---------------------- data prep ----------------------------
        scat = cdf[[col]].copy()
        scat["month"] = scat.index.month
        scat["hour"]  = scat.index.hour
        # keep all 12 months even if some are missing in the data
        ordered_months = list(range(1, 13))
        scat["month"]  = pd.Categorical(scat["month"], categories=ordered_months, ordered=True)
        # hourly mean per month from the SAME data used in each facet
        curve = (
            scat.groupby(["month", "hour"], observed=False)[col]
                .mean()
                .reset_index()
        )
        # light smoother so the line matches the reference look
        curve["smooth"] = (
            curve.groupby("month", observed=False)["hour"]
                .transform(lambda h: 0)  # just to keep index aligned; we overwrite next line
        )
        curve["smooth"] = (
            curve.groupby("month", observed=False)[col]
                .transform(lambda s: s.rolling(3, center=True, min_periods=1).mean())
        )
        # ---------------------- base scatter figure ------------------
        import plotly.express as px
        import plotly.graph_objects as go
        fig_sc = px.scatter(
            scat, x="hour", y=col,
            facet_col="month", facet_col_wrap=4,
            category_orders={"month": ordered_months},
            opacity=0.35,
            labels={"hour": "Hour", col: label_y},
            height=720
        )
        # ---------------------- overlay lines (matching facets) ------
        # Build a FACETED line figure with the exact same facet spec
        fig_ln = px.line(
            curve, x="hour", y="smooth",
            facet_col="month", facet_col_wrap=4,
            category_orders={"month": ordered_months},
        )
        # Copy the line traces onto the scatter; subplot targets stay correct
        for t in fig_ln.data:
            t.showlegend = False
            t.mode = "lines"
            t.line.width = 2
            t.line.color = line_color
            fig_sc.add_trace(t)
        # ---------------------- legend (clean) -----------------------
        pt_color = (
            fig_sc.data[0].marker.color
            if len(fig_sc.data) and hasattr(fig_sc.data[0], "marker")
            else "rgba(120,170,220,0.8)"
        )
        fig_sc.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=7, color=pt_color, opacity=0.6),
                name="Hourly points", showlegend=True
            ),
            row=1, col=1
        )
        fig_sc.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(width=3, color=line_color),
                name="Monthly mean (smoothed)", showlegend=True
            ),
            row=1, col=1
        )
        # ---------------------- month labels + spacing ---------------
        # Turn "month=1" → "Jan" and nudge down so titles don’t crowd
        fig_sc.for_each_annotation(
            lambda a: a.update(
                text=pd.Timestamp(2001, int(a.text.split("=")[-1]), 1).strftime("%b"),
                y=a.y - 0.03
            )
        )
        fig_sc.update_layout(
            legend=dict(orientation="h", x=0, xanchor="left", y=1.08, yanchor="bottom"),
            margin=dict(t=105, b=70, l=50, r=30),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    if effective_page in ("Temp & Humidity", "🌡️ Temperature & Humidity", "Annual Heatmap"):
        # ==================== HEATMAP ====================
        st.markdown("---")
        st.markdown("#### Annual heatmap (Day × Hour)")
        st.caption(
            "Rows track calendar days and columns track hours, so warm streaks, cold snaps, and "
            "overnight recovery patterns appear instantly. Use it to quickly locate persistent "
            "outliers before drilling into hourly tables."
        )
        tmp = pd.DataFrame({"doy": cdf.index.dayofyear, "hour": cdf.index.hour, "val": cdf[col]}).dropna()
        mat = tmp.pivot_table(index="doy", columns="hour", values="val", aggfunc="mean").sort_index()
        import plotly.express as px
        fig_hm = px.imshow(
            mat.values, origin="lower", aspect="auto",
            labels=dict(x="Hour", y="Day", color=("°C" if col=="drybulb" else "%")),
            height=350, color_continuous_scale=("RdYlBu_r" if col=="drybulb" else "Blues")
        )
        fig_hm.update_xaxes(side="bottom")
        st.plotly_chart(fig_hm, use_container_width=True)
# ---------------------- SUN & CLOUDS ----------------------

if effective_page == "☀️ Solar Analysis":
    st.markdown("### ☀️ Solar Analysis")
    st.caption(
        "Trace the sun’s path, check solar-time positions, and pair those tracks with EPW temperatures "
        "before taking shading or PV decisions. The plots highlight seasonal envelopes so you can "
        "see when the sun is high, low, or missing entirely."
    )

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


    # ---------- Solar helpers ----------

    def solar_positions(site: Site, date: pd.Timestamp) -> pd.DataFrame:
        """Hourly sun positions for a given civil date (local time: site.tz)."""
        # Local times from 0..23 at whole hours
        idx = pd.date_range(
            start=pd.Timestamp(date.date(), tz=site.tz),
            periods=24, freq="H", tz=site.tz
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
        cmap = matplotlib.cm.get_cmap("RdYlBu_r")
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
        show_envelope: bool = True,
        show_analemmas: bool = True,
        hours_for_analemma=range(6, 19),
        analemma_step_days: int = 7,
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
        shapes.append(dict(type="circle", xref="x", yref="y", x0=-1, y0=-1, x1=1, y1=1, line=dict(width=1.2, color="rgba(255,255,255,0.25)")))

        # Altitude concentric rings (10..80)
        for alt in range(10, 90, 10):
            x0, y0 = project_angular(np.array([0.0]), np.array([alt]), projection)
            r = float(np.hypot(x0, y0))
            shapes.append(dict(type="circle", xref="x", yref="y", x0=-r, y0=-r, x1=r, y1=r, line=dict(width=0.7, color="rgba(255,255,255,0.18)", dash="dot")))

        # Azimuth ticks (every 10°; major every 90°)
        for az in range(0, 360, 10):
            ux, uy = az_unit(az)
            inner = 0.98
            outer = 1.03 if az % 90 == 0 else 1.01
            lw = 2 if az % 90 == 0 else 1
            shapes.append(dict(type="line", x0=ux * inner, y0=uy * inner, x1=ux * outer, y1=uy * outer, line=dict(width=lw, color="rgba(255,255,255,0.35)")))

        # Hour radial lines (6..18), light dash
        for h in range(6, 19):
            az = (h - 12) * 15 + 180
            ux, uy = az_unit(az)
            shapes.append(dict(type="line", xref="x", yref="y", x0=0, y0=0, x1=ux * 1.0, y1=uy * 1.0, line=dict(width=1, color="rgba(255,255,255,0.25)", dash="dot")))

        fig.update_layout(shapes=shapes)

        # Cardinal labels (N,E,S,W) at perimeter
        cardinals = [("N", 0), ("E", 90), ("S", 180), ("W", 270)]
        fig.add_trace(go.Scatter(
            x=[az_unit(a)[0] * 1.07 for _, a in cardinals],
            y=[az_unit(a)[1] * 1.07 for _, a in cardinals],
            text=[t for t, _ in cardinals], mode="text", showlegend=False, hoverinfo="skip",
            textfont=dict(size=16, color="#e5e7eb", family="Arial Black")
        ))

        # Altitude ring labels (placed toward southern rim for readability)
        for alt in range(10, 90, 10):
            x0, y0 = project_angular(np.array([0.0]), np.array([alt]), projection)
            r = float(np.hypot(x0, y0))
            fig.add_trace(go.Scatter(x=[0.0], y=[-r], text=[f"{alt}°"], mode="text", showlegend=False, hoverinfo="skip",
                                     textfont=dict(size=11, color="#94a3b8")))

        # Perimeter azimuth degree labels (every 30°)
        az_lab = list(range(0, 360, 30))
        az_x = [az_unit(a)[0] * 1.09 for a in az_lab]
        az_y = [az_unit(a)[1] * 1.09 for a in az_lab]
        az_text = [f"{a}°" for a in az_lab]
        fig.add_trace(go.Scatter(x=az_x, y=az_y, text=az_text, mode="text", showlegend=False, hoverinfo="skip",
                     textfont=dict(size=9, color="#d1d5db")))

        # Hour labels placed on rim (6..18)
        hr_x = []
        hr_y = []
        hr_text = []
        for h in range(6, 19):
            az = (h - 12) * 15 + 180
            ux, uy = az_unit(az)
            hr_x.append(ux * 1.08)
            hr_y.append(uy * 1.08)
            hr_text.append(str(h))
            fig.add_trace(go.Scatter(x=hr_x, y=hr_y, text=hr_text, mode="text", showlegend=False, hoverinfo="skip",
                                     textfont=dict(size=11, color="#fbbf24")))

        # Daily seasonal arcs (Dec/Mar/Jun/Sep) and selected day
        def add_day_trace(day, color="#E67E22", width=2, dash="solid", name=None, opacity=1.0):
            xs, ys, _, _, idx = sunpath_for_day(day)
            if xs.size == 0:
                return
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width, dash=dash), name=name, opacity=opacity, showlegend=True, hoverinfo="skip"))

        year = date.year
        add_day_trace(pd.Timestamp(year=year, month=6, day=21, tz=site.tz), color="#facc15", width=2, name="Jun 21 (Summer)")
        add_day_trace(pd.Timestamp(year=year, month=12, day=21, tz=site.tz), color="#ef4444", width=2, name="Dec 21 (Winter)")
        add_day_trace(pd.Timestamp(year=year, month=3, day=21, tz=site.tz), color="#fb923c", width=2, name="Mar/Sept (Equinox)")

        # Selected date (red dotted)
        xs_sel, ys_sel, az_sel, el_sel, idx_sel = sunpath_for_day(date)
        if xs_sel.size > 0:
            fig.add_trace(go.Scatter(x=xs_sel, y=ys_sel, mode="lines", line=dict(color="#22d3ee", width=2.4, dash="dot"), name=f"{date:%b %d} Path", showlegend=True, hoverinfo="skip"))

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
                fig.add_trace(go.Scatter(x=[sun_marker_x], y=[sun_marker_y], mode="markers", marker=dict(size=16, color="#ff4d4d", line=dict(width=3, color="rgba(255,255,255,0.7)")), name="Current Sun", showlegend=True,
                                         hovertemplate=f"Time: {solar_time}<br>Az: {azimuth:.1f}°<br>Alt: {altitude:.1f}°<extra></extra>"))
                # radial line
                fig.add_shape(type="line", x0=0, y0=0, x1=sun_marker_x, y1=sun_marker_y, line=dict(color="#ff4d4d", width=2))

        # Solar info panel
        info_lines = []
        if azimuth is not None:
            info_lines.append(f"Solar Time: {solar_time}")
            info_lines.append(f"Azimuth: {azimuth:.1f}°")
            info_lines.append(f"Altitude: {altitude:.1f}°")
        else:
            info_lines.append("No sun above horizon on this date/site")

        # compute rise/set/twilight from full-day samples if available
        if len(idx_sel) > 0:
            sp_full = pvlib.solarposition.get_solarposition(idx_sel, site.lat, site.lon, altitude=site.elev_m)
            positives = sp_full[sp_full["elevation"] > 0]
            if not positives.empty:
                sunrise = positives.index[0].strftime("%H:%M")
                sunset = positives.index[-1].strftime("%H:%M")
                info_lines.append(f"Sunrise: {sunrise}")
                info_lines.append(f"Sunset: {sunset}")

            def tw(elev):
                mask = sp_full[sp_full["elevation"] > elev]
                if mask.empty:
                    return "--","--"
                return mask.index[0].strftime("%H:%M"), mask.index[-1].strftime("%H:%M")

            civil = tw(-6); naut = tw(-12); astro = tw(-18)
            info_lines.append(f"Civil: {civil[0]}–{civil[1]}")
            info_lines.append(f"Nautical: {naut[0]}–{naut[1]}")
            info_lines.append(f"Astronomical: {astro[0]}–{astro[1]}")

        info_lines.append(f"Lat: {site.lat:.4f}")
        info_lines.append(f"Lon: {site.lon:.4f}")
        info_text = "<br>".join(info_lines)
        fig.add_annotation(text=info_text, x=1.02, y=0.5, xref="paper", yref="paper", showarrow=False, align="left",
                   font=dict(size=12, color="#e5e7eb"), bordercolor="#334155", borderwidth=1, bgcolor="rgba(15,23,42,0.85)", borderpad=6)

        fig.update_layout(
            xaxis=dict(scaleanchor="y", range=[-1.12, 1.12], visible=False),
            yaxis=dict(range=[-1.12, 1.12], visible=False),
            margin=dict(l=20, r=180, t=56, b=20),
            height=700,
            title=f"Sun Path Diagram • {date:%b %d, %Y}",
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

        return fig



    def sunpath_plotly_3d(
        site: Site,
        date: pd.Timestamp,
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

        # --- Ground disk + light dome surface ---
        grid_xy = np.linspace(-1.05, 1.05, 40)
        gx, gy = np.meshgrid(grid_xy, grid_xy)
        gz = np.zeros_like(gx)
        fig.add_trace(go.Surface(
            x=gx, y=gy, z=gz,
            showscale=False,
            colorscale=[[0, "#1f2937"], [1, "#111827"]],
            opacity=0.08,
            hoverinfo="skip",
            name=None,
            showlegend=False,
        ))

        theta = np.linspace(0, 2 * np.pi, 64)
        phi = np.linspace(0, np.pi, 32)
        th, ph = np.meshgrid(theta, phi)
        x_s = np.sin(ph) * np.cos(th)
        y_s = np.sin(ph) * np.sin(th)
        z_s = np.cos(ph)
        fig.add_trace(go.Surface(
            x=x_s, y=y_s, z=z_s,
            opacity=0.04,
            colorscale=[[0, "#334155"], [1, "#0f172a"]],
            showscale=False,
            lighting=dict(ambient=0.8, diffuse=0.2, specular=0),
            hoverinfo="skip",
            name=None,
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

        # Horizon ring
        hx, hy, hz = _ring_points(0)
        fig.add_trace(go.Scatter3d(
            x=hx, y=hy, z=hz,
            mode="lines",
            line=dict(color="rgba(255,255,255,0.2)", width=2.2),
            name=None,
            showlegend=False,
            hoverinfo="skip",
        ))
        # Altitude rings every 15° with labels placed at az=180° (south)
        for alt_deg in range(15, 90, 15):
            rx, ry, rz = _ring_points(alt_deg)
            fig.add_trace(go.Scatter3d(
                x=rx, y=ry, z=rz,
                mode="lines",
                line=dict(color="rgba(255,255,255,0.12)", width=1.1),
                name=None,
                showlegend=False,
                hoverinfo="skip",
            ))
            alt_r = np.deg2rad(alt_deg)
            az_r = np.deg2rad(180)
            lx = 1.02 * np.cos(alt_r) * np.sin(az_r)
            ly = 1.02 * np.cos(alt_r) * np.cos(az_r)
            lz = np.sin(alt_r)
            fig.add_trace(go.Scatter3d(
                x=[lx], y=[ly], z=[lz],
                mode="text",
                text=[f"{alt_deg}°"],
                textposition="middle center",
                textfont=dict(size=10, color="rgba(230,230,230,0.65)"),
                name=None,
                showlegend=False,
                hoverinfo="skip",
            ))

        # Azimuth spokes every 30° (cardinals emphasized)
        for az_deg in range(0, 360, 30):
            az_r = np.deg2rad(az_deg)
            x0 = 0; y0 = 0; z0 = 0
            x1 = np.sin(az_r); y1 = np.cos(az_r); z1 = 0
            is_cardinal = az_deg % 90 == 0
            fig.add_trace(go.Scatter3d(
                x=[x0, x1], y=[y0, y1], z=[z0, z1],
                mode="lines",
                line=dict(color="rgba(255,255,255,{:.2f})".format(0.22 if is_cardinal else 0.1), width=2.2 if is_cardinal else 1.2),
                name=None,
                showlegend=False,
                hoverinfo="skip",
            ))

        # Cardinal labels on horizon
        cardinals = {
            "N": (0, 1.08, 0),   # az=0 -> y positive
            "E": (1.08, 0, 0),   # az=90 -> x positive
            "S": (0, -1.08, 0),  # az=180 -> y negative
            "W": (-1.08, 0, 0),  # az=270 -> x negative
        }
        fig.add_trace(go.Scatter3d(
            x=[v[0] for v in cardinals.values()],
            y=[v[1] for v in cardinals.values()],
            z=[v[2] for v in cardinals.values()],
            mode="text",
            text=list(cardinals.keys()),
            textposition="middle center",
            textfont=dict(color="rgba(230,230,230,0.8)", size=13),
            name=None,
            showlegend=False,
            hoverinfo="skip",
        ))

        def _sunpath_for(ts: pd.Timestamp, color: str, label: str):
            df = solar_positions(site, ts)
            if df.empty:
                return
            az = np.deg2rad(df["azimuth"].values)
            alt = np.deg2rad(df["elevation"].values)
            x = np.cos(alt) * np.sin(az)
            y = np.cos(alt) * np.cos(az)
            z = np.sin(alt)
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode="lines",
                line=dict(width=2, color=color),
                name=label,
                hovertemplate="%{customdata[0]:.1f}° az<br>%{customdata[1]:.1f}° alt<extra></extra>",
                customdata=np.c_[df["azimuth"].values, df["elevation"].values],
            ))

        # seasonal arcs (full 24h)
        season_days = [
            (12, 21, "#2D7DD2", "Dec 21"),
            (3, 21, "#00B0F0", "Mar 21"),
            (6, 21, "#FF6B3D", "Jun 21"),
            (9, 21, "#FFA14A", "Sep 21"),
        ]
        for m, d, col, label in season_days:
            ts = pd.Timestamp(year=date.year, month=m, day=d, tz=site.tz)
            _sunpath_for(ts, col, label)

        # selected day: hourly positions (solar time labels + data-driven colors)
        df = solar_positions(site, date)
        if not df.empty:
            df = _with_solar_time(df, site)
            # subsample per requested hour stride (keep first row)
            df = df.iloc[::max(1, int(hour_stride))]
            # keep marker cadence by hour modulo marker_every
            df = df[df.index.hour % max(1, int(marker_every)) == 0]

            color_series = None
            color_title = None
            cmin = None
            cmax = None
            colorscale = "Cividis"
            if epw is not None:
                if color_var == "temperature":
                    color_series = _nearest_epw_by_solar_time(epw, df["solar_time"], ["temp_air","DryBulb","Dry_Bulb","drybulb","Temperature"])
                    color_title = "Dry Bulb (°C)"
                    cmin, cmax = None, None
                elif color_var == "solar":
                    color_series = _nearest_epw_by_solar_time(epw, df["solar_time"], ["glohorrad","ghi","global_horiz","global_horizontal","solar","radiation"])
                    color_title = "Solar Radiation (W/m²)"
                    cmin, cmax = None, None
                elif color_var == "humidity":
                    color_series = _nearest_epw_by_solar_time(epw, df["solar_time"], ["relhum","relative_humidity","rh"])
                    color_title = "Relative Humidity (%)"
                    cmin, cmax = None, None
                elif color_var == "wind":
                    color_series = _nearest_epw_by_solar_time(epw, df["solar_time"], ["windspd","wind_speed","wspd","wind"])
                    color_title = "Wind Speed (m/s)"
                    cmin, cmax = None, None
            if color_series is not None:
                color_series = pd.to_numeric(color_series, errors="coerce")
                df["color_val"] = color_series
                finite_vals = df["color_val"].replace([np.inf, -np.inf], np.nan).dropna()
                if not finite_vals.empty:
                    low, high = np.percentile(finite_vals, [5, 95])
                    if low == high:
                        high = low + 1e-6
                    cmin, cmax = low, high

            az, alt = np.deg2rad(df["azimuth"].values), np.deg2rad(df["elevation"].values)
            sx = np.cos(alt) * np.sin(az)
            sy = np.cos(alt) * np.cos(az)
            sz = np.sin(alt)

            path_customdata = np.c_[df["azimuth"].values, df["elevation"].values]

            colors = df["color_val"].to_numpy() if "color_val" in df.columns else None
            unit = "" if color_title is None else color_title.split("(")[-1].replace(")", "")
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
                line=dict(width=4, color="#fbbf24"),
                name="Selected day",
                hoverinfo="skip",
                customdata=path_customdata,
                showlegend=True,
            ))

            every = max(1, int(label_every))
            text_labels = []
            if show_labels:
                for i, t in enumerate(time_strs):
                    text_labels.append(t if i % every == 0 else "")
            else:
                text_labels = [""] * len(time_strs)

            fig.add_trace(go.Scatter3d(
                x=sx, y=sy, z=sz,
                mode="markers+text" if show_labels else "markers",
                text=text_labels,
                textposition="top center",
                textfont=dict(size=11, color="rgba(240,240,240,0.92)"),
                marker=dict(
                    size=9,
                    color=colors if colors is not None else "#fbbf24",
                    colorscale=colorscale or "Cividis",
                    cmin=cmin,
                    cmax=cmax,
                    showscale=bool(colors is not None),
                    colorbar=(dict(title=color_title or "Value", thickness=12, len=0.55, x=1.03) if colors is not None else None),
                    line=dict(width=1.8, color="rgba(255,255,255,0.95)"),
                    symbol="circle",
                    opacity=0.96,
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
                xaxis=dict(range=[-1.1, 1.1], visible=False),
                yaxis=dict(range=[-1.1, 1.1], visible=False),
                zaxis=dict(range=[0.0, 1.05], visible=False),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=0.6),
                bgcolor="#0d1117",
            ),
            scene_camera=dict(
                eye=camera_eye if camera_eye is not None else dict(x=1.6, y=1.6, z=1.1),
                up=dict(x=0, y=0, z=1)
            ),
            plot_bgcolor="#0d1117",
            paper_bgcolor="#0d1117",
            font=dict(color="#e5e7eb"),
            dragmode="orbit",
            scene_dragmode="orbit",
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


        # ===== SUN PATH (interactive) =====

    c1, c2, c3 = st.columns([1,1,1])
    # Sun-path date input removed per UX request; use today's date in site timezone
    sel_date = pd.Timestamp.now(tz=tzinfo).date()
    proj = c2.selectbox("2D projection", ["stereographic", "orthographic"], index=0)
    show3d = c3.checkbox("Show 3D dome & rays", value=False)

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

    # Ensure tz-awareness consistent with site tz, then convert to UTC for clean joins
    if epw_df.index.tz is None:
        epw_df.index = epw_df.index.tz_localize(tzinfo)
    epw_df = epw_df.tz_convert("UTC").sort_index()
    epw_df = epw_df[~epw_df.index.duplicated(keep="first")]

    # 3D-specific controls
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

    c4, c5 = st.columns([1,1])
    color_choice_label = c4.selectbox("Color sun points by", options=avail_options, index=0)
    hour_stride = c5.slider("Show sun every N hours", min_value=1, max_value=4, value=1, step=1)
    color_var = option_map.get(color_choice_label, "temperature")

    show_labels = st.checkbox("Show labels on sun markers", value=True)
    marker_every = st.slider("Show sun markers every N hours", min_value=1, max_value=4, value=1, step=1)
    label_every = marker_every




    sel_ts = pd.Timestamp(sel_date, tz=tzinfo)
    display_date = sel_ts.strftime("%b %d, %Y")

    fig2d = sunpath_plotly_2d(site, sel_ts, proj)
    st.plotly_chart(fig2d, use_container_width=True, config={"displayModeBar": True})

    if show3d:
        st.subheader(f"Sun Path (3D) — {display_date}")
        view_state = st.session_state.get("sunpath3d_view", "default")
        v1, v2, v3, v4 = st.columns(4)
        if v1.button("Reset view"):
            view_state = "default"
        if v2.button("Top"):
            view_state = "top"
        if v3.button("South-facing"):
            view_state = "south"
        if v4.button("East-facing"):
            view_state = "east"
        st.session_state["sunpath3d_view"] = view_state

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
            st.plotly_chart(
                fig3d,
                use_container_width=True,
                config={
                    "scrollZoom": True,
                    "displayModeBar": True,
                    "displaylogo": False,
                },
            )
            st.caption("Sun position colored by selected environmental variable.")
        else:
            st.warning(f"3D sun path unavailable (got {type(fig3d).__name__}).")


    # ======================== PLOT: CARTESIAN (PVSyst-style) ========================
    st.markdown("#### Cartesian sun path (PVSyst style)")

    # Only temperature mode currently used for coloring (keeps prior behavior explicit)
    color_mode = "temperature"

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

    # ---- Build a color scalar for temperature mode (via nearest EPW to *solar time*) ----
    def _temps_for_index(idx_local):
        """Return temperatures aligned to idx_local (tz-aware) using nearest match in epw_df."""
        if epw_df is None or epw_df.empty:
            return np.full(len(idx_local), np.nan)
        # make sure epw_df index tz matches
        epw = epw_df.copy()
        if epw.index.tz is None:
            epw.index = epw.index.tz_localize(site.tz)
        # nearest lookup
        locs = epw.index.get_indexer(idx_local, method="nearest")
        locs = np.clip(locs, 0, len(epw) - 1)
        return epw.iloc[locs]["temp_air"].to_numpy(dtype=float)

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
    for m in range(1, 13):
        day = 21 if m != 2 else 20  # avoid Feb-31 style pitfalls
        ts_local = pd.date_range(pd.Timestamp(year, m, day, tz=tz), periods=24, freq="1H")
        az, alt, _ = solar_pos(ts_local.tz_convert("UTC"))
        mask = alt > 0
        if not mask.any():
            continue
        x = az[mask]; y = alt[mask]
        if color_mode == "temperature":
            temps = _temps_for_index(ts_local[mask])
            tr = go.Scatter(
                x=x, y=y, mode="lines+markers",
                marker=dict(size=4, color=temps, colorscale="RdYlBu_r",
                            cmin=-10, cmax=35, showscale=False),
                line=dict(width=2, color="rgba(200,200,200,0.75)"),
                name=pd.Timestamp(year, m, day).strftime("%b 21"),
                showlegend=(m in (6, 12))  # keep legend compact
            )
        else:
            col = _season_color(m)
            tr = go.Scatter(
                x=x, y=y, mode="lines+markers",
                marker=dict(size=4, color=col),
                line=dict(width=2, color=col),
                name=pd.Timestamp(year, m, day).strftime("%b 21"),
                showlegend=(m in (6, 12))
            )
        month_traces_cart.append(tr)

    # (B) Solstice envelope (Jun 21 vs Dec 21)
    def _daily_arc(month, day):
        tloc = pd.date_range(pd.Timestamp(year, month, day, tz=tz), periods=24, freq="1H")
        az, alt, _ = solar_pos(tloc.tz_convert("UTC"))
        m = alt > 0
        return (az[m], alt[m]) if m.any() else (np.array([]), np.array([]))

    az_su, alt_su = _daily_arc(6, 21)
    az_wi, alt_wi = _daily_arc(12, 21)
    if len(az_su) and len(az_wi):
        # We can't form a polygon in (az,alt) reliably (az wraps), so draw as two filled ribbons near horizon:
        # Instead, add two reference lines and a semi-transparent band near the lower arc
        envelope_cart.append(go.Scatter(x=az_su, y=alt_su, mode="lines",
                                        line=dict(width=2.6, color="#E67E22"),
                                        name="Summer solstice"))
        envelope_cart.append(go.Scatter(x=az_wi, y=alt_wi, mode="lines",
                                        line=dict(width=2.6, color="#3498DB"),
                                        name="Winter solstice"))

    # (C) Analemmas (fixed clock hour over the year)
    if show_analemmas:
        hours = range(6, 19)  # 6..18
        days  = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="7D", tz=tz)
        for hh in hours:
            idx = [pd.Timestamp(d.date(), tz=tz) + pd.Timedelta(hours=hh) for d in days]
            idx = pd.DatetimeIndex(idx)
            az, alt, _ = solar_pos(idx.tz_convert("UTC"))
            m = alt > 0
            if not m.any(): 
                continue
            analemma_cart.append(
                go.Scatter(
                    x=az[m], y=alt[m], mode="lines",
                    line=dict(width=1, dash="dot", color="rgba(60,80,160,0.55)"),
                    name=(f"{hh:02d}:00" if hh in (6, 12, 18) else None),
                    showlegend=hh in (6, 12, 18)
                )
            )

    # (D) Selected day (suns) with optional temperature colors & labels
    idx_sel = pd.date_range(pd.Timestamp(sel_ts.date(), tz=tz), periods=24, freq="1H")
    az_sel, alt_sel, _ = solar_pos(idx_sel.tz_convert("UTC"))
    m_sel = alt_sel > 0
    x_sel = az_sel[m_sel]; y_sel = alt_sel[m_sel]
    labels = [f"{t.hour:02d}:00" for t in idx_sel[m_sel]]
    if color_mode == "temperature":
        temps_sel = _temps_for_index(idx_sel[m_sel])
        suns_traces_cart.append(
            go.Scatter(
                x=x_sel, y=y_sel, mode="lines+markers+text",
                text=labels if show_hour_labels else None,
                textposition="top center",
                marker=dict(size=6, color=temps_sel, colorscale="RdYlBu_r",
                            cmin=-10, cmax=35, showscale=False, line=dict(width=0.5, color="#222")),
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

    # Optional coloraxis with colorbar (only if temperature mode AND requested)
    if color_mode == "temperature" and show_colorbar:
        # Add a tiny hidden dummy trace to carry the colorbar
        fig_cart.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=0.1, color=[-10, 35], colorscale="RdYlBu_r",
                        cmin=-10, cmax=35, showscale=True,
                        colorbar=dict(title="°C", len=0.85)),
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
                                pd.Timestamp(year,12,31,23,tz=tz), freq="1H")
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
    st.plotly_chart(fig_cart, use_container_width=True, config={"displayModeBar": False})








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
        st.plotly_chart(fig_cloud, use_container_width=True)

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

        st.plotly_chart(fig_sc, use_container_width=True)
        # ---- Annual heatmap (day-of-year × hour) ----
        tmp = pd.DataFrame({
            "doy": cdf.index.dayofyear,
            "hour": cdf.index.hour,
            "val": cdf[vcol]
        }).dropna()
        mat = tmp.pivot_table(index="doy", columns="hour", values="val", aggfunc="mean").sort_index()
        scale = "RdYlBu_r" if ("Wh/m²" in vlabel or "Dry-bulb" in vlabel) else "Blues"
        fig_hm = px.imshow(mat.values, origin="lower", aspect="auto",
                        labels=dict(x="Hour", y="Day of year", color=vlabel),
                        title=f"Annual heatmap — {vlabel}",
                        height=360, color_continuous_scale=scale)
        fig_hm.update_xaxes(side="bottom")
        st.plotly_chart(fig_hm, use_container_width=True)


if page == "📈 Psychrometrics":
    st.markdown("### 📈 Psychrometrics")
    st.caption(
        "Plot hourly temperature/ humidity points on the classic psychrometric grid to see "
        "when air falls inside comfort envelopes, where latent loads spike, and how far "
        "conditioning has to move conditions."
    )

    st.subheader("Psychrometric Chart")
    st.caption("Clean grid with absolute humidity (g/kg), RH isolines, saturation curve, and hourly EPW scatter.")

    # ------- Controls -------
    cA, cB = st.columns([1.2, 1])
    auto_zoom = cA.checkbox("Auto zoom to EPW range", value=True, help="Fit axes to EPW hourly temperature and absolute humidity range.")
    show_enthalpy = cB.checkbox("Show enthalpy & v lines", value=True)

    # ------- Thermo helpers (SI) -------
    def p_ws_kPa(TC: np.ndarray) -> np.ndarray:
        """Saturation vapor pressure over water [kPa] (Magnus/Tetens)."""
        return 0.61094 * np.exp(17.625*TC / (TC + 243.04))

    def w_from_Pv_kPa(Pv_kPa: np.ndarray, P_kPa: float) -> np.ndarray:
        """Humidity ratio w [kg/kg] from vapor pressure Pv [kPa] and total pressure P [kPa]."""
        Pv_kPa = np.clip(Pv_kPa, 0.0, 0.999 * P_kPa)
        return 0.62198 * Pv_kPa / (P_kPa - Pv_kPa)

    def abs_hum_gpkg_from_w(w: np.ndarray) -> np.ndarray:
        """Convert humidity ratio w [kg/kg] to absolute humidity [g/kg]."""
        return 1000.0 * w

    def dew_point_C(TC: np.ndarray, RH_pct: np.ndarray) -> np.ndarray:
        # Dew point in Celsius, Magnus approximation.
        a, b = 17.625, 243.04
        gamma = np.log(np.clip(RH_pct, 1e-6, 100)/100.0) + (a*TC)/(b+TC)
        return (b*gamma)/(a-gamma)

    def wet_bulb_C_stull(TC: np.ndarray, RH_pct: np.ndarray) -> np.ndarray:
        # Stull (2011) empirical wet-bulb approximation in Celsius.
        RH = np.clip(RH_pct, 1e-6, 100.0)
        return (TC*np.arctan(0.151977*np.sqrt(RH + 8.313659)) +
                np.arctan(TC + RH) - np.arctan(RH - 1.676331) +
                0.00391838*(RH**1.5)*np.arctan(0.023101*RH) - 4.686035)

    def enthalpy_kJkg(TC: np.ndarray, w: np.ndarray) -> np.ndarray:
        # Moist-air specific enthalpy (kJ/kg dry air).
        return 1.006*TC + w*(2501.0 + 1.86*TC)

    def specific_volume_m3kg(TC: np.ndarray, w: np.ndarray, P_kPa: float) -> np.ndarray:
        # Specific volume (m3/kg dry air).
        R = 0.287042  # kPa*m3/(kg*K) for dry air
        return R*(TC+273.15)*(1 + 1.6078*w)/P_kPa

    # ------- Data prep -------
    needed = ["drybulb", "relhum"]
    if not all(k in cdf.columns for k in needed):
        st.info("This EPW is missing required fields for the psychrometric plot.")
        st.stop()

    # Pressure: median of atmos_pressure if present, else 101.325 kPa
    if "atmos_pressure" in cdf and cdf["atmos_pressure"].notna().any():
        P_kPa = float(np.nanmedian(cdf["atmos_pressure"].values))/1000.0
    else:
        P_kPa = 101.325

    dfp = cdf[["drybulb", "relhum"]].dropna().copy()
    if dfp.empty:
        st.info("No points to plot.")
        st.stop()

    # Compute scatter values (absolute humidity g/kg)
    T_pts = dfp["drybulb"].to_numpy(float)
    RH_pts = dfp["relhum"].to_numpy(float)
    Pv_pts = (RH_pts/100.0) * p_ws_kPa(T_pts)
    w_pts  = w_from_Pv_kPa(Pv_pts, P_kPa)
    Y_pts_gpkg = abs_hum_gpkg_from_w(w_pts)

    # Auto ranges from EPW data (fallback to defaults if needed)
    def _safe_min(arr, default):
        v = np.nanmin(arr) if arr.size else np.nan
        return v if np.isfinite(v) else default

    def _safe_max(arr, default):
        v = np.nanmax(arr) if arr.size else np.nan
        return v if np.isfinite(v) else default

    if auto_zoom:
        x_min = _safe_min(T_pts, -10.0) - 2
        x_max = _safe_max(T_pts, 40.0) + 2
        y_min = max(0.0, _safe_min(Y_pts_gpkg, 0.0) - 1)
        y_max = _safe_max(Y_pts_gpkg, 30.0) + 1
    else:
        x_min, x_max = -10.0, 50.0
        y_min, y_max = 0.0, 40.0

    # Extra metrics for hover
    dp_pts = dew_point_C(T_pts, RH_pts)
    tw_pts = wet_bulb_C_stull(T_pts, RH_pts)
    h_pts  = enthalpy_kJkg(T_pts, w_pts)
    v_pts  = specific_volume_m3kg(T_pts, w_pts, P_kPa)
    # --- Psychro axis extents & styling constants ---
    X_MIN, X_MAX = x_min, x_max
    Y_MIN, Y_MAX = y_min, y_max

    # ------- Background curves (built in SI then plotted as °C vs g/kg) -------
    # Temperature axis for curves
    T_axis = np.linspace(X_MIN, X_MAX, 600)

    Pws_axis = p_ws_kPa(T_axis)

    # Saturation curve (100% RH)
    w_sat = w_from_Pv_kPa(Pws_axis, P_kPa)
    y_sat_gpkg = abs_hum_gpkg_from_w(w_sat)

    # RH isolines
    rh_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    rh_curves_gpkg = {}
    for rh in rh_list:
        Pv = (rh/100.0) * Pws_axis
        w  = w_from_Pv_kPa(Pv, P_kPa)
        rh_curves_gpkg[rh] = abs_hum_gpkg_from_w(w)

    # Optional: Enthalpy & specific volume guide families (light)
    enthalpy_levels = [20, 40, 60, 80]  # kJ/kg
    v_levels = [0.82]                   # m3/kg (typical label)

    def w_from_enthalpy(T, h):
        # Invert h = 1.006T + w(2501 + 1.86T) to w(T,h).
        return (h - 1.006*T) / (2501.0 + 1.86*T)

    def w_from_specific_volume(T, v, P_kPa):
        # Invert v = R(T+273.15)(1+1.6078w)/P to w(T,v).
        R = 0.287042
        return (v*P_kPa/(R*(T+273.15)) - 1.0)/1.6078

    # ------- Figure -------
    fig_psy = go.Figure()

    # Gray grid (optional: keep minimal; Plotly axes grid off, we emulate major lines)
    x_grid = np.arange(np.ceil(X_MIN/5)*5, np.floor(X_MAX/5)*5 + 0.1, 5)
    y_grid = np.arange(np.floor(Y_MIN/5)*5, np.floor(Y_MAX/5)*5 + 0.1, 5)

    for xv in x_grid:
        fig_psy.add_shape(type="line", x0=xv, x1=xv, y0=0, y1=y_max,
                        line=dict(color="rgba(150,150,150,0.25)", width=1), layer="below")
    for yv in y_grid:
        fig_psy.add_shape(type="line", x0=X_MIN, x1=X_MAX, y0=yv, y1=yv,
                        line=dict(color="rgba(150,150,150,0.25)", width=1), layer="below")

    # Saturation curve
    fig_psy.add_trace(go.Scatter(
        x=T_axis, y=y_sat_gpkg, mode="lines",
        line=dict(width=2.5, color="rgba(120,120,120,1.0)"),
        name="Saturation (100% RH)", hovertemplate="Tdb %{x:.1f}°C<br>Abs %{y:.2f} g/kg<extra></extra>"
    ))

    # RH isolines (dashed gray)
    for rh in rh_list:
        fig_psy.add_trace(go.Scatter(
            x=T_axis, y=rh_curves_gpkg[rh], mode="lines",
            line=dict(width=1.2, dash="dot", color="rgba(120,120,120,0.8)"),
            name=f"{rh}% RH", showlegend=(rh in (20,40,60,80)),
            hovertemplate=f"{rh}% RH<br>Tdb %{{x:.1f}}°C<br>Abs %{{y:.2f}} g/kg<extra></extra>"
        ))

    # Optional: enthalpy & specific volume helpers
    if show_enthalpy:
        for h in enthalpy_levels:
            w_line = w_from_enthalpy(T_axis, h)
            y_line = abs_hum_gpkg_from_w(w_line)
            fig_psy.add_trace(go.Scatter(
                x=T_axis, y=y_line, mode="lines",
                line=dict(width=1.25, dash="dash", color="rgba(255,165,0,0.85)"),
                name=(f"h={h} kJ/kg"), hoverinfo="skip"
            ))
        for v in v_levels:
            w_line = w_from_specific_volume(T_axis, v, P_kPa)
            y_line = abs_hum_gpkg_from_w(w_line)
            fig_psy.add_trace(go.Scatter(
                x=T_axis, y=y_line, mode="lines",
                line=dict(width=1.25, dash="dot", color="rgba(90,140,255,0.85)"),
                name=(f"v={v:.2f} m3/kg"), hoverinfo="skip"
            ))

    # RH % labels along the right margin (like PVSyst look)
    for rh in rh_list:
        x_lab = X_MAX - 0.2
        y_curve = np.interp(x_lab, T_axis, rh_curves_gpkg[rh])
        if Y_MIN <= y_curve <= Y_MAX:
            fig_psy.add_annotation(x=X_MAX, y=y_curve, text=f"{rh}%",
                                xanchor="left", showarrow=False,
                                font=dict(size=11, color="rgba(120,120,120,0.9)"))

    # Scatter points (hourly conditions)
    custom = np.c_[RH_pts, Pv_pts, h_pts, v_pts, dp_pts, tw_pts]
    fig_psy.add_trace(go.Scatter(
        x=T_pts, y=Y_pts_gpkg, mode="markers",
        marker=dict(size=4, opacity=0.35, color="royalblue"),
        name="Hourly conditions",
        customdata=custom,
        hovertemplate=(
            "<b>Hourly</b><br>"
            "Tdb %{x:.2f} °C<br>"
            "Abs %{y:.2f} g/kg<br>"
            "RH %{customdata[0]:.1f}%<br>"
            "Pv %{customdata[1]:.3f} kPa<br>"
            "h %{customdata[2]:.2f} kJ/kg<br>"
            "v %{customdata[3]:.3f} m3/kg<br>"
            "Tdp %{customdata[4]:.2f} °C<br>"
            "Twb %{customdata[5]:.2f} °C<extra></extra>"
        )
    ))

    

    # Axes & layout (minimal legend; neutral theme)
    fig_psy.update_xaxes(
        range=[X_MIN, X_MAX],
        dtick=10,
        ticks="outside",
        ticklen=6,
        gridcolor="rgba(255,255,255,0.06)",
        gridwidth=0.6,
        zeroline=False,
        showline=True,
        linecolor="rgba(255,255,255,0.25)",
        title="Dry Bulb Temperature (°C)",
    )
    fig_psy.update_yaxes(
        range=[Y_MIN, Y_MAX],
        dtick=5,
        ticks="outside",
        ticklen=6,
        gridcolor="rgba(255,255,255,0.06)",
        gridwidth=0.6,
        zeroline=False,
        showline=True,
        linecolor="rgba(255,255,255,0.25)",
        title="Absolute Humidity (g/kg)",
    )

    for tr in fig_psy.data:
        n = getattr(tr, "name", "")

        if n.endswith("% RH"):
            tr.line.width = 0.8
            tr.line.color = "rgba(255,255,255,0.28)"
            tr.hoverinfo = "skip"
            tr.showlegend = (n in {"20% RH", "40% RH", "60% RH", "80% RH"})

        if n.startswith("h="):  # enthalpy
            tr.line.width = 1.0
            tr.line.color = "rgba(255,180,0,0.65)"
            tr.hoverinfo = "skip"

        if n.startswith("v="):  # specific volume
            tr.line.width = 1.0
            tr.line.color = "rgba(120,170,255,0.55)"
            tr.hoverinfo = "skip"

        if n == "Saturation (100% RH)":
            tr.line.width = 3.0
            tr.line.color = "rgba(220,220,220,0.95)"

        if n == "Hourly conditions":
            tr.mode = "markers"
            tr.marker.size = 5
            tr.marker.opacity = 0.45


    fig_psy.update_layout(
        margin=dict(l=90, r=280, t=70, b=80),  # extra right for the legend
        height=680,
        legend=dict(
            orientation="v",
            x=1.18, y=1.0,
            xanchor="left", yanchor="top",
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.20)",
            borderwidth=0.8,
            itemclick=False,
            itemdoubleclick=False
        ),
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Psychrometric Chart", x=0.01, xanchor="left", yanchor="top", pad=dict(t=10, b=10)),
    )
    fig_psy.add_annotation(
        xref="paper", yref="paper", x=0.0, y=1.12,
        text="Based on EPW hourly data range.", showarrow=False,
        font=dict(size=11, color="rgba(120,120,120,0.9)")
    )
    st.plotly_chart(
        fig_psy,
        use_container_width=True,
        config={
            "displaylogo": False,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
        },
    )

if page == "📡 Live Data vs EPW":
    st.markdown("### Local Sensors vs Climate Baseline (EPW)")
    st.caption(
        "Compare on-site sensor readings to a long-term climate baseline (EnergyPlus Weather 'typical year'). Comparisons are statistical, not timestamp-based."
    )

    # ---------- State + helpers ----------
    focus_threshold = float(st.session_state.get("custom_overheat_threshold", 30))
    st.session_state.setdefault("sensor_df", pd.DataFrame())
    st.session_state.setdefault("sensor_history", [])
    st.session_state.setdefault("sensors", {})  # sensor_id -> dataframe
    st.session_state.setdefault("sensor_meta", {})  # sensor_id -> metadata dict
    st.session_state.setdefault("active_sensor_id", None)

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
                                raw = pd.read_csv(uploaded_file)
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
    st.markdown("#### Ingested sensors")
    sensors_store = st.session_state.get("sensors", {})
    sensor_meta = st.session_state.get("sensor_meta", {})
    history = st.session_state.get("sensor_history", [])

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

        def _alt_rows(df):
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            styles.iloc[::2, :] = "background-color: #f7f9fc"
            return styles

        styled = table_df.style.apply(_alt_rows, axis=None).set_properties(**{"text-align": "left"})
        st.dataframe(styled, use_container_width=True, height=240, column_config={
            "sensor_id": st.column_config.Column("sensor_id", width="small"),
            "records": st.column_config.Column("records", width="small"),
        })
        with st.expander("Choose active sensor for Live vs EPW", expanded=False):
            chosen = st.selectbox(
                "Active sensor",
                options=list(sensors_store.keys()),
                index=list(sensors_store.keys()).index(st.session_state.get("active_sensor_id")) if st.session_state.get("active_sensor_id") in sensors_store else 0,
            )
            st.session_state["active_sensor_id"] = chosen
            st.caption(f"Active sensor set to {chosen}. This drives the Live Data vs EPW analysis.")
    else:
        st.info("No ingests yet. Upload a file or fetch from an API to get started.")

    st.divider()

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
        st.plotly_chart(fig_temp_bar, use_container_width=True)
        st.caption("Sensor temperatures over the selected period. Shows the observed range and average conditions on site.")

        if rh_stats:
            fig_rh_bar = go.Figure([
                go.Bar(x=["Min", "Mean", "Max"], y=[rh_stats[k] for k in ["min", "mean", "max"]], marker_color="#4c78a8")
            ])
            fig_rh_bar.update_layout(title="Observed sensor humidity (entire period)", yaxis_title="Relative Humidity (%)", height=280, margin=dict(l=10,r=10,t=32,b=16))
            st.plotly_chart(fig_rh_bar, use_container_width=True)
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
            st.plotly_chart(fig_hour, use_container_width=True)
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
            st.plotly_chart(fig_month, use_container_width=True)
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
            st.plotly_chart(fig_hot, use_container_width=True)
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
            st.plotly_chart(fig_comfort, use_container_width=True)
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
            resampled = work.resample("1H").mean()
            if numeric_cols:
                resampled[numeric_cols] = resampled[numeric_cols].interpolate(method="time", limit_direction="both")
                resampled[numeric_cols] = resampled[numeric_cols].rolling(window=1, center=True, min_periods=1).mean()
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
                return aligned[(aligned.index >= start_ts.floor("H")) & (aligned.index <= end_ts.ceil("H"))]

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
                st.plotly_chart(fig, use_container_width=True)
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
                st.plotly_chart(fig, use_container_width=True)
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

            epw_hot = {thr: int((epw_slice.set_index("timestamp")["T_db"] > thr).resample("1H").sum().sum()) for thr in [26, 28, 30]}
            sensor_hot = {thr: int((sensor_slice.set_index("timestamp")["T_db"] > thr).resample("1H").sum().sum()) for thr in [26, 28, 30]}

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
                st.plotly_chart(fig_hot, use_container_width=True)
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

if page == "Sensor Comparison":
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
            except Exception:
                # If anything goes wrong, fall back to including the column
                pass

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

    if "selected_sensors" not in st.session_state:
        st.session_state["selected_sensors"] = available_sensors[:2] if len(available_sensors) >= 2 else available_sensors[:1]

    # keep only sensors still present
    st.session_state["selected_sensors"] = [s for s in st.session_state["selected_sensors"] if s in available_sensors]
    if not st.session_state["selected_sensors"]:
        st.session_state["selected_sensors"] = available_sensors[:2] if len(available_sensors) >= 2 else available_sensors[:1]

    # Top Card: Compare Different Sensors
    with st.container(border=True):
        st.markdown("#### Compare Different Sensors")
        st.caption("Select sensors to compare")
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
            for sensor in selected_sensors:
                sensor_data = plot_df[plot_df[sensor_col] == sensor]
                if not sensor_data.empty and "_metric" in sensor_data.columns and "_ts" in sensor_data.columns:
                    fig_primary.add_trace(go.Scatter(
                        x=sensor_data["_ts"],
                        y=sensor_data["_metric"],
                        mode="lines",
                        name=sensor,
                        hovertemplate=f"{sensor}<br>%{{x}}<br>%{{y:.2f}}{unit_str}<extra></extra>"
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

            st.plotly_chart(fig_primary, use_container_width=True)

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
                    for sensor in selected_sensors:
                        sensor_data = plot_df_dist[plot_df_dist[sensor_col] == sensor]["_metric"].dropna()
                        if not sensor_data.empty:
                            fig_dist.add_trace(go.Histogram(x=sensor_data, name=sensor, opacity=0.7, nbinsx=40))
                    fig_dist.update_layout(title=f"{primary_metric.replace('_', ' ').title()} Distribution", xaxis_title=f"{primary_metric.replace('_', ' ').title()}{unit_str}", yaxis_title="Frequency", barmode="overlay", height=380)
                else:
                    st.info("Metric appears categorical/low-cardinality; distribution histogram unavailable.")

        st.plotly_chart(fig_dist, use_container_width=True)

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
                for sensor in selected_sensors:
                    sensor_data = plot_df[plot_df[sensor_col] == sensor]
                    if not sensor_data.empty and "_metric" in sensor_data.columns and "_ts" in sensor_data.columns:
                        fig_secondary.add_trace(go.Scatter(
                            x=sensor_data["_ts"],
                            y=sensor_data["_metric"],
                            mode="lines",
                            name=sensor,
                            hovertemplate=f"{sensor}<br>%{{x}}<br>%{{y:.2f}}{unit_str}<extra></extra>"
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

                st.plotly_chart(fig_secondary, use_container_width=True)
    
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
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
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

# ====================== RAW DATA ======================
if page == "📁 Raw Data":
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

        st.dataframe(view, use_container_width=True, height=360)

        # download filtered CSV
        csv_bytes = view.to_csv(index=False).replace("\r\n", "\n").encode("utf-8")
        st.download_button("⬇️ Download filtered CSV", data=csv_bytes,
                           file_name="climate_filtered.csv", mime="text/csv")

    # ---------- EPW HEADER (optional) ----------
    with st.expander("EPW header (metadata)"):
        st.json(st.session_state.header, expanded=False)


if page == "📈 Short-Term Prediction (24–72h)":
    st.markdown("### 📈 Short-Term Prediction (24–72h)")
    st.caption(
        "Train a lightweight SARIMAX model on the last 1–2 weeks of sensor data, then compare the next few days "
        "against EPW expectations to see if a heat event is brewing. The charts below highlight when forecasts "
        "exceed comfort thresholds or diverge from the TMY baseline."
    )

    model_choice = st.session_state.get("forecast_model_choice", "Auto SARIMAX (default)")
    focus_threshold = float(st.session_state.get("custom_overheat_threshold", 30))

    sensor_hourly = fc.load_sensor_data()
    if sensor_hourly.empty:
        st.info("No sensor history available. Ingest data via the Live Data tab to unlock forecasting.")
    else:
        st.session_state.setdefault("short_forecast", None)
        st.session_state.setdefault("short_forecast_bias", None)
        st.session_state.setdefault("short_forecast_meta", None)

        stored_meta = st.session_state.get("short_forecast_meta") or {}
        default_horizon = int(stored_meta.get("horizon_hours", 72))
        default_training = int(stored_meta.get("training_days", 14))
        default_conf_pct = int(round((stored_meta.get("confidence_level", 0.8)) * 100))
        default_horizon = min(max(default_horizon, 24), 168)
        default_training = min(max(default_training, 7), 30)
        default_conf_pct = min(max(default_conf_pct, 60), 95)

        with st.container():
            ctrl1, ctrl2, ctrl3 = st.columns(3)
            horizon_hours = ctrl1.slider(
                "Horizon (hours)", 24, 168, value=default_horizon, step=12,
                help="Choose how far ahead to forecast. Longer horizons widen uncertainty."
            )
            training_days = ctrl2.slider(
                "Training window (days)", 7, 30, value=default_training, step=1,
                help="Use this many trailing days of sensor data for model fitting."
            )
            confidence_pct = ctrl3.slider(
                "Confidence band (%)", 60, 95, value=default_conf_pct, step=5,
                help="The shaded band and textual summary use this interval."
            )
            confidence_level = confidence_pct / 100.0

            st.caption(f"Active forecast model: {model_choice}")
            if model_choice != "Auto SARIMAX (default)":
                st.warning("Alternative model options are in preview; falling back to SARIMAX while we finish their implementations.")

            if st.button("Train forecast", type="primary"):
                with st.spinner("Fitting SARIMAX models per variable…"):
                    forecast_df = fc.build_forecast_model(
                        sensor_hourly,
                        horizon_hours=horizon_hours,
                        training_days=training_days,
                        confidence_level=confidence_level,
                    )
                    epw_clim_short = fc.load_epw_climatology(cdf)
                    bias_df = fc.compare_forecast_to_epw(forecast_df, epw_clim_short)
                    st.session_state["short_forecast"] = forecast_df
                    st.session_state["short_forecast_bias"] = bias_df
                    st.session_state["short_forecast_meta"] = {
                        "horizon_hours": horizon_hours,
                        "training_days": training_days,
                        "confidence_level": confidence_level,
                        "model_choice": model_choice,
                    }
        forecast_df = st.session_state.get("short_forecast")
        bias_df = st.session_state.get("short_forecast_bias")
        meta = st.session_state.get("short_forecast_meta") or {}
        active_conf = float(meta.get("confidence_level", 0.8))
        active_horizon = int(meta.get("horizon_hours", 72))
        active_training = int(meta.get("training_days", 14))
        active_model = meta.get("model_choice", model_choice)
        if forecast_df is None or forecast_df.empty:
            st.info("Click the button above to generate an outlook with the selected settings.")
        else:
            temp_series = pd.Series(forecast_df["temp_forecast"])
            max_temp = float(temp_series.dropna().max()) if not temp_series.dropna().empty else float("nan")
            overheating_hours = int((temp_series >= focus_threshold).sum())
            delta_series = bias_df["epw_temp_bias_forecast"] if bias_df is not None and not bias_df.empty else pd.Series(dtype=float)
            delta_mean = float(delta_series.mean()) if not delta_series.empty else np.nan

            history_series = None
            if "temperature" in sensor_hourly.columns:
                history_window_start = sensor_hourly.index.max() - pd.Timedelta(days=active_training)
                recent = sensor_hourly.loc[sensor_hourly.index >= history_window_start, "temperature"].dropna()
                if not recent.empty:
                    history_series = recent

            m1, m2, m3 = st.columns(3)
            m1.metric(f"{active_horizon}h max temperature", format_temperature(max_temp))
            m2.metric(f"{format_threshold_label(focus_threshold)} hours", f"{overheating_hours}")
            m3.metric("Mean Δ forecast vs EPW", format_temperature_delta(delta_mean) if not np.isnan(delta_mean) else "—")

            st.markdown("#### Forecast outlook")
            st.plotly_chart(
                fc.plot_forecast(
                    forecast_df,
                    confidence_level=active_conf,
                    recent_history=history_series,
                ),
                use_container_width=True,
            )

            peak = fc.summarize_peak_event(forecast_df)
            if peak and peak.get("temp") is not None and not np.isnan(peak.get("temp", np.nan)):
                ts = peak.get("timestamp")
                if pd.isna(ts):
                    ts_label = "Peak hour"
                else:
                    ts_label = pd.Timestamp(ts).strftime("%b %d %H:%M")
                lower = peak.get("lower")
                upper = peak.get("upper")
                if lower is None or upper is None or np.isnan(lower) or np.isnan(upper):
                    band_text = "band unavailable"
                else:
                    band_text = f"{format_temperature(lower)} – {format_temperature(upper)}"
                band_pct = int(round(active_conf * 100))
                st.caption(
                    f"Peak around {ts_label}: {format_temperature(peak['temp'])} with {band_pct}% band {band_text}."
                )

            st.info(
                f"Forecasts come from {active_model} fit to the last {active_training} days of hourly sensor temperatures. "
                "Adjust the training window or try another model preset in the sidebar to test sensitivity."
            )

            st.markdown("#### EPW vs forecast bias")
            st.plotly_chart(fc.plot_bias(bias_df if bias_df is not None else pd.DataFrame()), use_container_width=True)

            st.markdown("#### Overheating flags")
            st.plotly_chart(fc.plot_overheating(forecast_df), use_container_width=True)

            with st.expander("Forecast table", expanded=False):
                st.dataframe(forecast_df.set_index("timestamp"), use_container_width=True, height=260)


if page == "🌍 Future Climate (2050 / 2080 SSP)":
    st.markdown("### 🌍 Future Climate Scenarios")
    st.caption(
        "Blend today’s EPW (optionally bias-corrected by your sensors) with CMIP6 deltas to sketch how typical "
        "years shift under SSP scenarios. Pick a pathway and horizon below to see the temperature/comfort "
        "impacts and download morphed EPWs."
    )
    focus_threshold = float(st.session_state.get("custom_overheat_threshold", 30))

    base_df = st.session_state.get("df")
    header_meta = st.session_state.get("header")
    if base_df is None or base_df.empty or header_meta is None:
        st.info("Load an EPW file on the main tabs to unlock future morphing.")
    else:
        scenario_label = st.selectbox(
            "Scenario",
            list(fepw.SCENARIO_MAP.keys()),
            format_func=lambda key: f"{key} · {fepw.SCENARIO_DESCRIPTIONS.get(key, '')}"
        )
        target_year = st.selectbox("Reporting year", fepw.TARGET_YEARS, format_func=lambda y: f"{y}")
        use_sensor_baseline = st.toggle("Use sensor-calibrated baseline", value=True)
        temp_only = st.toggle("Apply CMIP6 deltas to temperature only", value=False)
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

        with st.expander("Use custom CMIP6 delta CSV", expanded=False):
            st.caption(
                "Upload a CSV with columns scenario, year, month, delta_temp, delta_rh, "
                "delta_wind, delta_ghi (optionally lat_band). Sources: WeatherShift™, epwshiftr, etc."
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

        st.info("These scenarios morph a typical year by applying average climate deltas—they are not weather forecasts for a specific future year.")

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
            st.plotly_chart(fig_future, use_container_width=True)

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

                    def _metric_value(series, key, pct=False):
                        val = series.get(key, np.nan)
                        if pd.isna(val):
                            return "—"
                        return f"{val * 100:.1f} %" if pct else f"{val:.0f} h"

                    def _delta(target, base, key, pct=False):
                        tv, bv = target.get(key, np.nan), base.get(key, np.nan)
                        if pd.isna(tv) or pd.isna(bv):
                            return None
                        delta = tv - bv
                        if pct:
                            return f"{delta * 100:+.1f} ppt"
                        return f"{delta:+.0f} h"

                    k1, k2, k3 = st.columns(3)
                    k1.metric(
                        "Comfort compliance",
                        _metric_value(target_row, "fraction_in_comfort_band", pct=True),
                        delta=_delta(target_row, base_row, "fraction_in_comfort_band", pct=True),
                    )
                    k2.metric(
                        ">28 °C hours",
                        _metric_value(target_row, "overheating_hours_28C"),
                        delta=_delta(target_row, base_row, "overheating_hours_28C"),
                    )
                    k3.metric(
                        "UTCI heat stress",
                        _metric_value(target_row, "hours_utci_heat_stress"),
                        delta=_delta(target_row, base_row, "hours_utci_heat_stress"),
                    )

                    bars = [
                        ("Comfort band %", target_row.get("fraction_in_comfort_band", np.nan) * 100.0,
                         base_row.get("fraction_in_comfort_band", np.nan) * 100.0),
                        (">28 °C h", target_row.get("overheating_hours_28C", np.nan), base_row.get("overheating_hours_28C", np.nan)),
                        ("UTCI heat h", target_row.get("hours_utci_heat_stress", np.nan), base_row.get("hours_utci_heat_stress", np.nan)),
                        ("CDD", target_row.get("cooling_degree_days", np.nan), base_row.get("cooling_degree_days", np.nan)),
                    ]
                    def _safe_val(v):
                        return None if pd.isna(v) else float(v)

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
                        yaxis_title="Hours / %",
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_compare, use_container_width=True)

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


# ========== FOOTER ==========
st.markdown("#### 📚 Learn more")
st.markdown(
    "\n".join([
        "- [What is a Typical Meteorological Year (TMY)?](https://energyplus.net/weather/help) — background on how representative weather files are built.",
        "- [pvlib-python reference](https://pvlib-python.readthedocs.io/) — documentation for the solar position and irradiance models powering the Solar Analysis tab.",
        "- [Universal Thermal Climate Index (UTCI)](https://www.utci.org/) — explains the heat stress metric used in the comfort dashboards.",
        "- [Future climate scenarios & CMIP6 overview](https://www.ipcc.ch/report/ar6/wg1/) — learn how SSP pathways drive the delta tables used for morphing.",
        "- [EPW datasets via OneBuilding](https://climate.onebuilding.org/) — source for most typical-year weather files.",
        "- [Project source on GitHub](https://github.com/your-org/bevl-climate-app) — inspect or contribute to the code powering this app."
    ])
)

st.markdown("---")
footer_html = (
    "<div style='text-align: center; color: #64748b; font-size: 0.9rem;'>"
    "<p>Climate Analysis Pro - Professional Weather Data Analysis Tool</p>"
    "</div>"
)
st.markdown(footer_html, unsafe_allow_html=True)
##EPW_hour[h] = mean(EPW values at hour h across the whole EPW year)

##Sensor_hour[h] = mean(sensor values at hour h across the sensor’s available dates)