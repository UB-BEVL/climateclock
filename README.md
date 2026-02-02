Climate Analysis Pro - Feature Overview
======================================

Purpose
-------
Streamlit-based weather/EPW exploration with station search, quick picks, interactive map, uploads, and analysis tabs.

Key Flows
---------
1) Station search (type-ahead)
   - Large search input filters stations live by city/state/country/ISO3/station ID/source/period.
   - Shows top matches with human-friendly labels (e.g., "Buffalo Greater, NY, United States (WMO 725280, TMY, 2007-2021)").
   - Selecting a match stores the full station row in session_state (selected_station) and stages it for loading.

2) Quick-pick buttons
   - Four shortcut stations (Buffalo/Denver/Chicago/Phoenix) prefill station metadata and load immediately.

3) Interactive map
   - Plotly scatter map of all stations; clicking a point stages and loads that station.

4) Sidebar upload
   - EPW/ZIP upload; auto extraction of EPWs from ZIPs; optional pick if multiple EPWs inside.

5) Current selection panel
   - Displays the friendly label for the loaded station or uploaded EPW, plus period/source.

6) Analysis tabs
   - Multiple pages (Dashboard, Temperature/Humidity, Solar, Psychrometrics, Live vs EPW, Raw Data, Short-Term Prediction, Future Climate) rendered after a station or EPW is loaded.

 Main Modules in app.py
 ----------------------
 - Styling: PRIMARY_CSS/SECONDARY_CSS inject dark theme; nav uses styled horizontal radio buttons.
 - Loaders: load_station_index() normalizes station records (lat/lon, URLs, period, heating/cooling db), parses raw_id into country/state/city/station_id/source, cleans NaNs, and builds display_label.
 - State helpers: _rerun(), _stage_station_and_load() keep sel_station, selected_station, sel_station_url, source_label, and navigate to the first tab after selection.
 - Station picker: search bar with top-N matches, quick-picks, and map click integration, all feeding _stage_station_and_load().
 - Upload flow: handle_epw_upload() supports EPW or ZIP, prompts when multiple EPWs in ZIP, stores raw bytes and source labels.
 - Sidebar controls: temperature units, custom overheat threshold, adaptive comfort toggle, UHI bias slider, forecast model choice.
 - Analysis sections (post-load):
   * Dashboard & Temperature/Humidity: key charts, stats, degree-day style outputs.
   * Solar Analysis: sun-path sphere, irradiance charts, shading/azimuth context.
   * Psychrometrics: psych chart with points/regions.
   * Live Data vs EPW: compares real-time feed (if configured) to EPW baselines.
   * Raw Data: tabular EPW exploration and downloads.
   * Short-Term Prediction: SARIMAX default with fallback options.
   * Future Climate: SSP 2050/2080 scenarios overlay.
 - Caching: STREAMLIT cache decorators are no-ops when unavailable (for compatibility).
 - Robustness: URL extraction from anchor tags, fallback URL guessing, period/heating/cooling coercion, and safe defaults when data are missing.

Data Handling
-------------
- Station index is normalized to ensure lat/lon present and to clean URLs.
- raw_id is parsed when available to derive country_iso3, state_code, city_raw/city_name, station_id, source, and period.
- Labels avoid nan by filling blanks with empty strings before formatting.
- Selected station metadata propagates to session_state for downstream charts and labeling.

UI Styling (Dark Theme)
----------------------
- Flat top navigation via styled radio buttons.
- Dark cards, subtle borders, cyan accents for active states.
- Compact controls: reduced paddings on select components; map wrapper sized for clarity.

How to Run
----------
- Install dependencies: pip install -r requirements.txt
- Launch app: streamlit run app.py
- Optional (fixed port): streamlit run app.py --server.port 8502
- Open the provided local URL in a browser.

Notes
-----
- Large tables (e.g., Raw Data views) use `streamlit-aggrid` for paginated rendering to keep the browser responsive.
   If `streamlit-aggrid` is not available, the app falls back to Streamlit's built-in `st.dataframe`.

Tips
----
- Use the search box for fast lookup; keep queries short (city, ISO3, or WMO ID).
- If no matches appear, broaden the term or check spelling.
- Uploading an EPW overrides the current station selection and updates the source label.
