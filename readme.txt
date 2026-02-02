Climate Analysis Pro - Feature Overview
======================================

Purpose
-------
A comprehensive Streamlit-based application for exploring EnergyPlus Weather (EPW) files, analyzing climate data, and projecting future climate scenarios. The tool integrates global station search, interactive visualizations, and advanced analytics for built environment professionals.

Tab-by-Tab Analysis Guide (Simplified)
--------------------------------------

1. 🛰️ Select Weather File
   Start here to choose a location.
   - **Search**: Find any city from a huge global list (over 10,000 locations).
   - **Map**: Click a point on the world map to find the nearest weather station.
   - **Upload**: Have your own weather file? Drag and drop it here.

2. 📊 Dashboard
   A quick summary of the local climate.
   - **Overview**: See the average temperature, wind, and sunlight for the location.
   - **Comfort Check**: Checks if the climate is generally comfortable or if it gets too hot/cold.
   - **Heating/Cooling Needs**: Estimates how much heating or AC you might need during the year.

3. 📁 Raw Data
   Look at the actual weather numbers.
   - **Data Table**: Scroll through every hour of the year to see detailed records.
   - **Download**: Save the weather data as a CSV file to use in Excel or other tools.

4. 🌡️ Temperature & Humidity
   Understand the hot and cold patterns.
   - **Daily Charts**: See how temperature changes from day to night for every month.
   - **Heatmaps**: A color-coded map showing the hottest and coldest times of the year at a glance.
   - **Humidity**: Charts showing how humid or dry it gets each month.

5. ☀️ Solar Analysis
   See where the sun is and how much energy it gives.
   - **Sun Path**: Interactive diagrams showing the sun's position in the sky throughout the year.
   - **Solar Energy**: Charts showing how much sunlight (solar radiation) hits the ground.
   - **Shadows**: (Experimental) A basic tool to check for potential shadows.

6. 📈 Psychrometrics (Comfort Chart)
   A tool to check human thermal comfort.
   - **Comfort Zone**: Visualizes air temperature and humidity together to see if they fall within a comfortable range for people.
   - **Design Strategies**: Suggests passive cooling ideas (like fans or natural ventilation) based on the climate.

7. 📡 Live Data vs. Typical Weather
   Compare your local sensor data with historical averages.
   - **Connect Sensors**: Link up your own temperature sensors to the app.
   - **Real vs. Typical**: See if your building site is hotter or colder than the standard weather file for that city.
   - **Heat Watch**: Automatically flags if your site is staying too hot at night (Urban Heat Island effect).

8. 🧪 Sensor Comparison
   Compare different sensors against each other.
   - **Side-by-Side**: Plot data from multiple sensors on one graph to see differences.
   - **Stats**: Quickly see which sensor is recording the highest or lowest temperatures.

9. 🧭 Short-Term Forecast (24–72h)
   Predict the weather for the next few days.
   - **Forecast**: Uses your recent sensor data to predict what will happen in the next 1-3 days.
   - **Alerts**: Warns you if an upcoming heatwave is likely to exceed comfort limits.

10. 🌍 Future Climate Scenarios
    See how climate change might affect this location.
    - **Future View**: Adjusts today's weather data to simulate conditions in 2050 or 2080.
    - **Scenarios**: Choose different "what-if" scenarios (low vs. high global emissions).
    - **Impact**: Shows how much hotter it might get and how that changes cooling needs.

Technical Architecture
----------------------
The codebase is structured for modularity and performance:

- **app.py**: Main entry point handling navigation, state management, and page routing.
- **metrics/**: Core calculation modules.
  - `comfort_energy.py`: UTCI, Heat Index, and comfort model calculations.
- **models/**: Forecasting and projection logic.
  - `forecasting.py`: SARIMAX and time-series models.
  - `future_epw.py`: Logic for applying CMIP6 climate deltas to EPW files.
- **sensors/**:
  - `live_sensors.py`: Integration with live weather APIs or local sensors.
  - `preprocess.py`: Data cleaning and normalization routines.

Key Libraries
-------------
- **Streamlit**: Web framework and UI components.
- **Pandas / NumPy**: High-performance data manipulation.
- **Plotly**: Interactive charting (Plotly Express & Graph Objects).
- **Pvlib**: Solar geometry and irradiance calculations.
- **SciPy / Statsmodels**: Statistical analysis and forecasting.

Setup & Running
---------------
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
   For a fixed port (e.g., 8502):
   ```bash
   streamlit run app.py --server.port 8502
   ```

3. **Access**:
   Open the URL provided in the terminal (usually http://localhost:8501).

Notes for Developers
--------------------
- **Caching**: Extensive use of `st.cache_data` and `st.cache_resource` for performance.
- **Theming**: Custom CSS injection for "Premium" and "Dark" modes.
- **Data Handling**: 
  - EPW files are parsed into Pandas DataFrames with a datetime index.
  - Floating point data is downcasted to `float32` to optimize memory usage.
  - Robust handling for missing data and various EPW formats (TMY3, TMYx).

Contact & Credits
-----------------
**BEVL Lab** - UB School of Architecture & Planning
Research-grade tools for the built environment.
