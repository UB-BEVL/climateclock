# Climate Analysis Pro - Detailed Documentation
=============================================

## Run locally on Windows

From the project folder, run:

```powershell
py run_app.py
```

The launcher finds Python 3.12 or 3.11, creates `.venv-climateclock`, installs
`requirements.txt` on first use, and starts Streamlit using that environment.
First setup needs internet access. Later launches reuse the environment.
Use Ctrl+C in the terminal to stop the app. Install Python 3.12 first if the
launcher cannot find a supported interpreter.

Avoid launching this project with `py -m streamlit run app.py`: the default
Python launcher may select a different environment without the required UTCI
and PMV packages. The existing `.venv` folder may also refer to another machine.

## Weather intelligence and team AI assistant

Open the **AI Assistant** messenger bar at the bottom right of any page.
It expands upward into a docked chat with suggested questions and conversation history.
Click its header again to minimize it while keeping your conversation.
The chatbot can use the loaded typical-year climate, future scenario summaries,
and any forecast context already loaded by Short-Term Prediction. Select
**Refresh context** inside the chatbot when you also want current apparent
temperature, hourly precipitation, wind gusts, and air quality included.

The assistant is optional. Copy `.streamlit/secrets.example.toml` to
`.streamlit/secrets.toml`, then set your team's chat-completions endpoint, model
name, and API key. The endpoint should accept an OpenAI-compatible JSON request
with `model` and `messages`, and return either `choices[0].message.content` or a
top-level `answer`, `response`, `text`, `content`, or `output` string. The app
sends compact climate summaries rather than the complete hourly EPW dataset.
The real secrets file is ignored by Git. Equivalent `SLM_API_URL`, `SLM_API_KEY`,
`SLM_MODEL`, `SLM_AUTH_HEADER`, and `SLM_AUTH_SCHEME` environment variables are
also supported.

## Interactive psychrometrics and tours

Overview now has five tour steps; all six climate cards share one explanation,
with individual help available on hover. Detail sections keep their own guides.
Edit the step lists in `app.py::_onboarding_steps`; `tour.py` handles positioning,
Next/Back, navigation, and reset behavior.

Open **Detail View > Psychrometrics > Interactive chart** for the integrated
equipment builder, weather overlays, comfort tools, teaching guide, and exports.
Use **Focus chart** for more drawing space. **Climate Strategies** contains the
existing passive-design analysis. The studio saves edits while switching pages
in the current browser tab; use JSON export for permanent storage. Wait for the
studio's saved-report status before opening **Report** to include its operating
cases in the full climate PDF. **Reset Session** starts a new location/design.
See [the studio guide](psychrometric_studio/README.md) for features and editing.

## Purpose

Climate Analysis Pro is a comprehensive, Streamlit-based web application designed for built-environment professionals, architects, and researchers. It allows users to explore hourly weather data (EPW files) through interactive visualizations, thermal comfort metrics, and solar analysis. 

The application provides a "Research-grade sandbox" to evaluate climates globally, either by searching a live database of over 30,000 WMO weather stations or by uploading custom `.epw` / `.zip` files.

---

## 1. Application Layout & Navigation
The application is structured into a main visualization area and a fixed sidebar for global controls.

### Sidebar Controls
These filters act globally, meaning changes here instantly recalculate the underlying dataset and update all charts simultaneously.
* **Navigation Menu**: Switch between different analysis dashboards (e.g., Dashboard, Temperature, Solar).
* **Temperature Units**: Toggle between Celsius (°C) and Fahrenheit (°F).
* **Focus Comfort Threshold**: A user-defined "Overheating" setpoint. All hours exceeding this temperature are tallied and displayed across the app as a custom heat load metric.
* **Urban Heat Island (UHI) Bias**: A toggle and slider to artificially inflate the dry-bulb temperature data by a fixed delta (e.g., +1.5°C). This simulates the warming effect of dense urban environments.
* **Month Range**: Restrict all datasets and visualizations to a specific time of year (e.g., Months 6-8 for Summer only).

---

## 2. Terminology & Core Metrics
The application relies on several standard meteorological and building-science metrics.

### Basic Meteorological Terms
* **Dry-bulb Temperature**: The ambient air temperature, uninfluenced by radiation or moisture.
* **Relative Humidity (RH)**: The percentage of water vapor present in the air relative to the maximum it could hold at that temperature.
* **Dew Point**: The temperature to which air must be cooled to become fully saturated with water vapor.
* **Global Horizontal Irradiance (GHI)**: Total solar radiation received on a horizontal surface (Direct + Diffuse).
* **Direct Normal Irradiance (DNI)**: Solar radiation received precisely perpendicular to the sun's rays.
* **Diffuse Horizontal Irradiance (DHI)**: Solar radiation scattered by the atmosphere, received on a horizontal surface.

### Thermal Comfort Metrics
* **Discomfort Index (DI) / Thom's Index**: A simple empirical index combining dry-bulb temperature and wet-bulb temperature to estimate human discomfort.
* **UTCI (Universal Thermal Climate Index)**: An advanced, bio-meteorological index that models human thermal stress. It accounts for temperature, humidity, wind speed, and mean radiant temperature to output a "feels-like" equivalent temperature.
* **Humidex & Heat Index**: Standard indices used by weather agencies to describe how hot the weather feels to the average person, factoring in humidity.
* **Adaptive Comfort**: A dynamic comfort model (ASHRAE 55) recognizing that people in naturally ventilated buildings adapt to their local climate. The "comfort zone" shifts historically based on the prevailing mean outdoor temperatures.

---

## 3. Analysis Dashboards & Visualizations

### 📊 Dashboard (Overview)
This page acts as a high-level summary of the loaded climate.
* **Climate Overview**: Displays site metadata (Latitude, Longitude, Elevation) and annual/seasonal averages for Temperature, Humidity, Wind, and Solar Radiation.
* **Comfort & Loads**: Summarizes the percentage of the year spent inside the comfort band. It categorizes unmet hours into cold stress and heat stress (using UTCI).
* **Data Quality**: Evaluates the EPW file for missing or invalid data, plotting the chronological completeness of key metrics to ensure simulation validity.
* **Heatmaps**: A high-density chronological visualization. The Y-axis represents the hour of the day (0-23), and the X-axis represents the day of the year (1-365). Colors indicate the intensity of metrics like temperature and solar radiation, allowing quick spotting of diurnal and seasonal patterns.

### 🌡️ Temperature & Humidity
Deep dive into sensible heat and moisture.
* **Temperature/Humidity Distribution (Violin & Box Plots)**: Shows the statistical spread, median, and quartile range of temperatures for every month. Violin width indicates frequency at that temperature.
* **Diurnal Swing (Line Charts)**: Plots the average daily profile (minimum, average, maximum) for each month, highlighting the temperature swing between night and day.
* **Comfort Diagnostics (Bar Charts)**: A visual breakdown of heating required, cooling required, and "free running" (comfortable) hours per month.

### ☀️ Solar Analysis
Visualizations dedicated to understanding solar geometry and site radiation.
* **Interactive 3D Sun Path**: A volumetric, interactive globe showing the sun's trajectory across the sky vault throughout the year. The red/orange arcs represent the sun's daily path at different seasons, while the vertical analemmas show the sun's position at a specific hour over the year.
* **Solar Radiation Heatmaps**: 24x365 contour plots showing when the site receives the most intense solar energy.
* **Irradiance Distribution**: Histograms showing the frequency and intensity of Direct Normal and Diffuse Horizontal irradiance.

### 📈 Psychrometrics
The engineer's view of the climate's thermodynamic properties.
* **Psychrometric Chart**: A fundamental thermodynamic graph plotting Dry-Bulb Temperature (X-axis) against Humidity Ratio (Y-axis). 
  * **Saturation Curve**: The upper boundary where air is 100% saturated (condensation point).
  * **Scatter Points**: Every hour of the year is plotted as a point.
  * **Comfort Polygon**: An overlaid boundary indicating the zone of human comfort. Points outside this zone require heating, cooling, humidification, or dehumidification.
  * **Enthalpy & Specific Volume Lines**: Isoclines revealing the total energy and physical density of the air at any given state.

### 📡 Live Data vs EPW (Live Sensors)
* **Ingested Sensors**: Connects to real-time IoT datastreams (if configured) to compare actual measured conditions on-site against the historical EPW baseline. Useful for tracking climate drift or verifying local microclimates.

### 🔀 Sensor Comparison
* **Multi-Trace Plotting**: Overlay multiple incoming live sensor streams on a single temporal X-axis for hardware comparisons and anomaly detection.

### 📁 Raw Data
* **Tabular EPW Explorer**: A paginated, filterable grid displaying the raw, hour-by-hour numeric data of the loaded weather file. Allows for localized CSV exports of specific time slices.

---

## 4. Troubleshooting & Notes
* If the charts appear blank, ensure an EPW file corresponds to the correct geographic location and contains valid hourly data.
* **Performance**: Loading extreme datasets spanning decades may cause the browser to temporarily throttle. The application uses smart caching and data downcasting (Float32) to mitigate memory overhead.
* **Browser Compatibility**: Recommended for use on Chromium-based browsers (Chrome, Edge) or Firefox. Safar's WebGL implementation may occasionally struggle with the 3D Sun Path.


### Official Future Weather Generator

Long-Term Prediction uses the official FWG CMIP6 Global v5.2.0 distribution (Engine 1.1.0), not synthetic climate deltas. Install Java 21 and run `python tools/install_fwg.py` on each app host before using generation. The distribution download is approximately 3.7 GB; allow additional space for extraction and cached results. Runtime files are excluded from Git.

`FWG_JAR` overrides the distribution path, `FWG_JAVA` overrides the Java executable, and `FWG_CACHE` selects a writable cache directory. The default cache is `climateclock-fwg` inside the operating system temporary directory, outside OneDrive and with shorter Windows paths. A full annual EPW is required. Generation runs automatically when baseline or scenario changes and produces both 2050 (2036–2065) and 2080 (2066–2095) ensemble EPWs for the chosen SSP. Temperature-only/custom delta controls are not part of this engine workflow. Sensor bias correction, when available, is applied to the input baseline before FWG runs.

Completed, validated runs are reused; errors never fall back to fabricated climate signals. Runtime logs and FWG manifests remain in the cached run folder. On hosted deployments, provision Java, the pinned distribution, writable storage and sufficient memory before enabling this feature; changing Python code alone does not install the engine.

FWG by Eugénio Rodrigues, David Carvalho and Marco S. Fernandes is distributed under CC BY-NC-SA. See https://future-weather-generator.adai.pt/download/ and https://future-weather-generator.adai.pt/documentation/cli-reference.html. Scientific reference: Rodrigues et al. (2023), https://doi.org/10.1016/j.buildenv.2023.110104.


On Windows, the bundled, version-checked compatibility sidecar fixes an access-denied canonical-path lookup while retaining checksum validation and scientific bytecode. See `tools/fwg_compat/README.md` for source, build steps and attribution. Expanded climate CSVs use NTFS compression where available.

The visible default is a **two-model ensemble (CanESM5 + CNRM-CM6-1)**, not the full 23-model ensemble. The climate-model selector lists the installed catalog and supports larger ensembles. This avoids exhausting modest local disks; users needing broader model coverage must provision sufficient free space. Model selection is included in both UI and persistent cache identity. Real EPW validation can be run with `python tools/verify_fwg_real.py --epw INPUT.epw --output OUTPUT_DIRECTORY`.


### Portable local climate assistant

The configured assistant uses Qwen2.5 1.5B Q4_K_M (approximately 986 MB) with the CPU build of llama.cpp b12328. Runtime files live under `vendor/llama` and are excluded from Git. The GGUF download was verified against its Ollama registry SHA256. Source distributions: https://github.com/ggml-org/llama.cpp/releases/tag/b12328 and https://ollama.com/library/qwen2.5:1.5b .

The `[slm]` secrets section selects `model = "qwen2.5:1.5b"`, `api_url = "http://127.0.0.1:11434/v1/chat/completions"`, and `local_managed = true`. The existing API-key field is a local compatibility value, not a cloud credential. Requests explicitly marked managed start the installed server on demand and reuse it thereafter. Other provider configurations are unchanged. The server binds only to loopback, uses four CPU threads, and logs to `weather_cache/logs/local-ai.log`. No cloud API is called for answers; only the app's compact climate summary is sent to the local process. Startup and inference require the app and model to be on the same Windows computer.

On the Buffalo EPW test, the assistant correctly reported 9.0 C annual mean dry-bulb temperature and 70.9% relative humidity, distinguished typical-year data from live weather, and declined to infer flood risk without rainfall data. This is a small language model: responses still require review, especially design advice beyond supplied measurements.
