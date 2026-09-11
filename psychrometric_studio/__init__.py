"""Embed the locally bundled Psychrometric Studio and serve its export requests."""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import tempfile
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests
import streamlit as st
import streamlit.components.v1 as components
from fpdf import FPDF
from fpdf import __version__ as _fpdf_version
from PIL import Image

ROOT = Path(__file__).resolve().parent
_component = components.declare_component("climateclock_psychrometric_studio", path=str(ROOT / "build"))
MAX_ARCHIVE = 20 * 1024 * 1024


def _text(value) -> str:
    return str(value if value is not None else "-").replace("—", "-").replace("–", "-").encode("latin-1", "replace").decode("latin-1")


def _number(value) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
        return f"{number:,.2f}" if math.isfinite(number) else "-"
    except (TypeError, ValueError):
        return _text(value)


def _paragraph(pdf, text, *, size=9, bold=False, colour=(45, 65, 82)):
    pdf.set_x(16)
    pdf.set_font("Helvetica", "B" if bold else "", size)
    pdf.set_text_color(*colour)
    pdf.multi_cell(pdf.w - 32, 5, _text(text))
    pdf.ln(2)


def _page(pdf, title, subtitle=""):
    if hasattr(pdf, "current_section"):
        pdf.current_section = "Psychrometric Analysis"
    pdf.add_page(orientation="L")
    pdf.set_xy(16, 27)
    _paragraph(pdf, title, size=18, bold=True, colour=(25, 83, 105))
    if subtitle:
        _paragraph(pdf, subtitle, size=9)


def _table(pdf, title, columns, rows):
    _paragraph(pdf, title, size=12, bold=True)
    width = (pdf.w - 32) / len(columns)
    def header():
        pdf.set_x(16)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(230, 240, 245)
        pdf.set_text_color(25, 60, 80)
        for key, label in columns:
            pdf.cell(width, 7, _text(label), border=0, fill=True)
        pdf.ln(7)
    header()
    for row in rows:
        values = [_text(row.get(key, "-")) if key in {"name", "type"} else _number(row.get(key)) for key, _ in columns]
        # Measure text before drawing the row, preserving long equipment names.
        line_count = max(max(1, math.ceil(pdf.get_string_width(value) / max(width - 3, 1))) for value in values)
        height = max(8, line_count * 4.2 + 2)
        if pdf.get_y() + height > pdf.h - 18:
            _page(pdf, "Psychrometric Analysis - continued")
            header()
        y = pdf.get_y()
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(40, 55, 70)
        for i, value in enumerate(values):
            pdf.set_xy(16 + width*i, y + 1)
            pdf.multi_cell(width - 2, 4.2, value)
        pdf.set_draw_color(222, 230, 235)
        pdf.line(16, y + height, pdf.w - 16, y + height)
        pdf.set_xy(16, y + height)
    pdf.ln(5)


def validate_cases(cases):
    if not isinstance(cases, list) or not cases or len(cases) > 16:
        raise ValueError("A report must contain between 1 and 16 operating cases.")
    for case in cases:
        if not isinstance(case, dict) or case.get("units") not in {"SI", "IP"}:
            raise ValueError("Invalid studio report units.")
        if not isinstance(case.get("statePoints"), list) or len(case["statePoints"]) > 200:
            raise ValueError("The studio report has too many state points.")
        if not isinstance(case.get("loads"), list) or len(case["loads"]) > 200:
            raise ValueError("The studio report has too many process rows.")
        if case.get("chartPng"):
            raw = base64.b64decode(case["chartPng"], validate=True)
            if len(raw) > 16 * 1024 * 1024:
                raise ValueError("The chart image is too large. Reduce the chart size and retry.")
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != "PNG" or image.width * image.height > 20_000_000:
                    raise ValueError("Invalid chart image.")
                image.verify()
    return cases


def append_report_pages(pdf, cases) -> int:
    """Append solved state schedules and chart images, without recalculating them."""
    if not cases:
        return 0
    validate_cases(cases)
    image_count = 0
    for case in cases:
        meta = case.get("meta") or {}
        name = meta.get("name") or "Psychrometric design"
        subtitle = f"{case['units']} units | {case.get('pressure', '')}"
        _page(pdf, f"Psychrometric Analysis - {name}", subtitle)
        if case.get("weatherStation"):
            _paragraph(pdf, f"Weather: {case['weatherStation']} | {case.get('weatherHours', '-')} selected hours | {case.get('weatherMode', 'off')} layer")
        if case.get("chartPng"):
            raw = io.BytesIO(base64.b64decode(case["chartPng"]))
            with Image.open(raw) as im:
                ratio = im.height / im.width
            raw.seek(0)
            available_h = pdf.h - pdf.get_y() - 27
            w = min(pdf.w - 32, available_h / ratio)
            h = w * ratio
            # PyFPDF 1.x accepts filenames only; fpdf2 also accepts streams.
            # Close the file before reading it on Windows. Both engines consume
            # the PNG immediately, so its temporary directory can then be removed.
            with tempfile.TemporaryDirectory(prefix="climateclock-chart-") as folder:
                chart_path = Path(folder) / "chart.png"
                chart_path.write_bytes(raw.getvalue())
                pdf.image(str(chart_path), x=(pdf.w - w)/2, y=pdf.get_y(), w=w, h=h)
            pdf.set_y(pdf.get_y() + h + 3)
            image_count += 1
        else:
            _paragraph(pdf, "No chart image was supplied for this operating case.")
        provenance = case.get("provenance") or {}
        _paragraph(pdf, f"{provenance.get('libraryVersion', 'PsychroLib 2.5.0')} | {provenance.get('calculationBasis', '')}", size=7)
        _page(pdf, name, subtitle)
        labels = case.get("labels") or {}
        details = [f"{label}: {meta[key]}" for key, label in [("projectNumber", "Project"), ("client", "Client"), ("engineer", "Engineer")] if meta.get(key)]
        if details:
            _paragraph(pdf, " | ".join(details))
        _table(pdf, "State points", [
            ("point", "Point"), ("name", "Equipment"), ("tdb", f"Tdb ({labels.get('temperature', 'C' if case['units']=='SI' else 'F')})"),
            ("twb", "Wet bulb"), ("rh", "RH (%)"), ("tdp", "Dew point"),
            ("w", "W (g/kg)" if case['units']=='SI' else "W (gr/lb)"),
            ("h", "h (kJ/kg)" if case['units']=='SI' else "h (Btu/lb)"),
        ], [{**row, "rh": float(row["rh"])*100 if row.get("rh") is not None else None} for row in case["statePoints"]])
        _table(pdf, "Airflow and specific volume", [
            ("point", "Point"), ("name", "Equipment"), ("airflow", labels.get("airflow", "Airflow")),
            ("massFlow", labels.get("massFlow", "Mass flow")), ("v", "v (m3/kg)" if case['units']=='SI' else "v (ft3/lb)"),
        ], case["statePoints"])
        for row in case["statePoints"]:
            if row.get("error"):
                _paragraph(pdf, f"Point {row.get('point')}: {row['error']}")
        _table(pdf, f"Process duties ({labels.get('duty', '')}; moisture in {labels.get('moistureRate', '')}; positive into the airstream)", [
            ("point", "Point"), ("name", "Equipment"), ("total", "Total"), ("sensible", "Sensible"),
            ("latent", "Latent"), ("moisture", "Moisture"), ("shr", "SHR"), ("adp", "ADP"), ("bypass", "Bypass factor"),
        ], case["loads"])
        for row in case["loads"]:
            if row.get("note"):
                _paragraph(pdf, f"{row.get('name', 'Process')}: {row['note']}")
        totals = case.get("totals") or {}
        _paragraph(pdf, f"Total cooling: {_number(totals.get('cooling'))} | Total heating: {_number(totals.get('heating'))} {labels.get('duty', '')}", bold=True)
        _paragraph(pdf, f"Humidification: {_number(totals.get('humidification'))} | Dehumidification: {_number(totals.get('dehumidification'))} {labels.get('moistureRate', '')}")
        _paragraph(pdf, totals.get("balance") or "Energy-balance check unavailable.")
        if meta.get("notes"):
            _paragraph(pdf, f"Project notes: {meta['notes']}")
        _paragraph(pdf, provenance.get("disclaimer", "For engineering analysis and education."), size=7)
    return image_count


class PsychrometricPDF(FPDF):
    def footer(self):
        x = 16
        self.set_draw_color(25, 83, 105)
        self.line(16, self.h-14, self.w-16, self.h-14)
        for name in ['1.png', '2.png']:
            path = ROOT.parent / 'assets' / name
            if path.exists():
                with Image.open(path) as logo:
                    width = 7 * logo.width / logo.height
                self.image(str(path), x=x, y=self.h-12, w=width, h=7)
                x += width + 5
        self.set_xy(self.w-60, self.h-11)
        self.set_font('Helvetica', '', 8)
        self.set_text_color(45,65,82)
        self.cell(44,5,f'Page {self.page_no()} of {{nb}}',align='R')


def build_studio_pdf(cases) -> bytes:
    pdf = PsychrometricPDF(orientation="L", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(16, 20, 16)
    append_report_pages(pdf, cases)
    result = pdf.output(dest="S") if _fpdf_version.startswith("1.") else pdf.output()
    return bytes(result) if isinstance(result, (bytes, bytearray)) else result.encode("latin-1")


def _allowed_weather_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if (parsed.scheme != "https" or parsed.hostname != "climate.onebuilding.org"
            or parsed.port not in (None, 443) or parsed.username or parsed.password
            or not parsed.path.lower().endswith(".zip")):
        raise ValueError("Use an HTTPS .zip link from climate.onebuilding.org, or upload the file in the studio.")
    return value.strip()


def fetch_weather_archive(url: str) -> bytes:
    url = _allowed_weather_url(url)
    for _ in range(4):
        with requests.get(url, timeout=(10, 50), stream=True, allow_redirects=False) as response:
            if response.is_redirect:
                url = _allowed_weather_url(urljoin(url, response.headers.get("Location", "")))
                continue
            response.raise_for_status()
            if int(response.headers.get("Content-Length", "0")) > MAX_ARCHIVE:
                raise ValueError("The archive exceeds the 20 MB limit.")
            data = bytearray()
            for chunk in response.iter_content(64 * 1024):
                data.extend(chunk)
                if len(data) > MAX_ARCHIVE:
                    raise ValueError("The archive exceeds the 20 MB limit.")
            if not data.startswith(b"PK"):
                raise ValueError("The address did not return a ZIP archive.")
            return bytes(data)
    raise ValueError("The weather archive redirected too many times.")


def weather_text_from_frame(frame, header) -> str:
    """Supply the studio's EPW reader from normalized data when raw bytes are absent."""
    import pandas as pd
    location = (header or {}).get("location") or {}
    clean = lambda value: str(value if value is not None else "").replace(",", " ").replace("\n", " ")
    location_row = ["LOCATION", location.get("city", "Loaded station"), location.get("state_province", ""),
                    location.get("country", ""), location.get("source", "Loaded climate data"), location.get("wmo", ""),
                    location.get("latitude", 0), location.get("longitude", 0), location.get("timezone", 0), location.get("elevation_m", 0)]
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow([clean(value) for value in location_row])
    for line in ["DESIGN CONDITIONS,0", "TYPICAL/EXTREME PERIODS,0", "GROUND TEMPERATURES,0", "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0", "COMMENTS 1,Normalized climate data", "COMMENTS 2,", "DATA PERIODS,1,1,Data,Sunday,1/1,12/31"]:
        output.write(line + "\n")
    for timestamp, row in frame.iterrows():
        timestamp = pd.Timestamp(timestamp)
        if pd.isna(timestamp):
            continue
        def number(column, default):
            try:
                value = float(row.get(column, default))
                return value if math.isfinite(value) else default
            except (TypeError, ValueError):
                return default
        writer.writerow([timestamp.year, timestamp.month, timestamp.day, timestamp.hour + 1, 60, "?9",
                         number("drybulb", 99.9), number("dewpoint", 99.9), number("relhum", 999),
                         number("atmos_pressure", 101325)] + [0] * 25)
    return output.getvalue()


def _source_id():
    return str(st.session_state.get("last_parsed_epw_hash") or st.session_state.get("last_loaded_station_id") or st.session_state.get("climate_clock_tour_session"))


def current_report_cases():
    if st.session_state.get("psychrometric_studio_report_source") != _source_id():
        return []
    return st.session_state.get("psychrometric_studio_report") or []


def _render_component(epw_text: str, station_label: str, pressure_kpa: float, session_key: str):
    station_key = session_key + "-" + hashlib.sha256(epw_text.encode()).hexdigest()[:16]
    response_key = "_psychrometric_studio_response"
    value = _component(station_key=station_key, station_label=station_label,
                       epw_text=epw_text, pressure_kpa=pressure_kpa,
                       response=st.session_state.get(response_key),
                       key="psychrometric-studio-" + station_key, default=None)
    if not isinstance(value, dict) or not value.get("id"):
        return
    if st.session_state.get("_psychrometric_studio_last_request") == value["id"]:
        return
    response = {"id": value["id"]}
    try:
        action, payload = value.get("action"), value.get("payload") or {}
        if action == "sync_report":
            if payload.get("station_key") != station_key:
                raise ValueError("The location changed; please reopen the studio.")
            cases = validate_cases(payload.get("cases"))
            st.session_state["psychrometric_studio_report"] = cases
            st.session_state["psychrometric_studio_report_source"] = _source_id()
            st.session_state["pdf_download_bytes"] = None
            st.session_state["pdf_download_name"] = None
            response["result"] = {"saved": True}
        elif action == "pdf":
            response["result"] = {"pdf": base64.b64encode(build_studio_pdf([payload])).decode()}
        elif action == "weather":
            response["result"] = {"data": base64.b64encode(fetch_weather_archive(str(payload.get("url", "")))).decode()}
        else:
            raise ValueError("Unknown studio request.")
    except Exception as exc:
        response["error"] = str(exc)
    st.session_state[response_key] = response
    st.session_state["_psychrometric_studio_last_request"] = value["id"]
    st.rerun()


def render_studio(epw_text: str, station_label: str, pressure_kpa: float, session_key: str):
    _render_component(epw_text, station_label, pressure_kpa, session_key)
    cases = current_report_cases()
    if not cases:
        st.caption("The PDF download appears here once the psychrometric results are ready.")
        return
    choice_col, download_col = st.columns([2, 1])
    options = ["All operating cases"] + [f"{index + 1}: {case.get('caseLabel') or (case.get('meta') or {}).get('name', 'Operating case')}" for index, case in enumerate(cases)]
    with choice_col:
        choice = st.selectbox("PDF cases", options, key="psychrometric_pdf_case")
    selected = cases if choice == options[0] else [cases[options.index(choice)-1]]
    fingerprint = hashlib.sha256(("pdf-v3:" + json.dumps(selected, sort_keys=True)).encode()).hexdigest()
    cached = st.session_state.get("_psychrometric_pdf_cache")
    if not cached or cached[0] != fingerprint:
        try:
            cached = (fingerprint, build_studio_pdf(selected))
        except Exception as exc:
            # An export failure must not replace the interactive analysis page.
            with download_col:
                st.error(f"PDF could not be prepared: {exc}")
                st.caption("Your chart is still available. Retry after an edit, or restart with py run_app.py.")
            return
        st.session_state["_psychrometric_pdf_cache"] = cached
    with download_col:
        st.download_button("Download psychrometric PDF", data=cached[1], file_name="psychrometric-analysis.pdf", mime="application/pdf", key="psychrometric_pdf_download", use_container_width=True)
