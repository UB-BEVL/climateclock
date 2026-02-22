# Climate Analysis Pro - Detailed Documentation
=============================================

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
