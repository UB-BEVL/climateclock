import re

def main():
    with open("app.py", "r", encoding="utf-8") as f:
        content = f.read()

    # The new function to insert
    new_func = """
_UTCI_BANDS = [
    (-100, -40, "Extreme Cold Stress (<-40°C)", "#1a1040"),
    (-40, -27, "Very Strong Cold Stress (-40 - -27°C)", "#2f3183"),
    (-27, -13, "Strong Cold Stress (-27 - -13°C)", "#3559a6"),
    (-13, 0, "Moderate Cold Stress (-13 - 0°C)", "#4bb3d4"),
    (0, 9, "Slight Cold Stress (0 - 9°C)", "#8fe0ee"),
    (9, 26, "No Stress (9 - 26°C)", "#ffffff"),
    (26, 32, "Moderate Heat Stress (26 - 32°C)", "#ffc080"),
    (32, 38, "Strong Heat Stress (32 - 38°C)", "#fc6554"),
    (38, 46, "Very Strong Heat Stress (38 - 46°C)", "#d12229"),
    (46, 100, "Extreme Heat Stress (>46°C)", "#7c1114"),
]

_PMV_BANDS = [
    (-100, -2.5, "Cold (< -3)", "#3559a6"),
    (-2.5, -1.5, "Cool (-3 to -2)", "#4bb3d4"),
    (-1.5, -0.5, "Slightly Cool (-2 to -1)", "#8fe0ee"),
    (-0.5, 0.5, "Neutral (-1 to +1)", "#ffffff"),
    (0.5, 1.5, "Slightly Warm (+1 to +2)", "#ffc080"),
    (1.5, 2.5, "Warm (+2 to +3)", "#fc6554"),
    (2.5, 100, "Hot (> +3)", "#d12229"),
]

_DI_BANDS = [
    (-100, 21.0, "Comfortable (<= 21)", "#ffffff"),
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
    for doy_i in range(366):
        for hr_i in range(24):
            v = mat_val[doy_i, hr_i]
            c = int(mat_cat[doy_i, hr_i]) if not np.isnan(mat_cat[doy_i, hr_i]) else -1
            if c != -1:
                hovertext[doy_i, hr_i] = f"Day {doy_i+1}, Hour {hr_i}<br>Val: {v:.1f}<br>{labels[c]}"
    fig.add_trace(go.Heatmap(
        x=mat_df.index, y=mat_df.columns,
        z=mat_cat.T,
        colorscale=dscale,
        zmin=0, zmax=n_colors-1,
        showscale=False,
        xgap=1, ygap=1,
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
    fig.update_xaxes(tickvals=month_days, ticktext=month_names, ticklen=5, range=[0.5, 366.5])
    fig.update_yaxes(tickvals=[0, 6, 12, 18, 23], ticktext=["12AM", "6AM", "12PM", "6PM", "11PM"], autorange="reversed", ticklen=5, range=[-0.5, 23.5])
    fig.update_layout(height=450, margin=dict(t=30, b=40, l=50, r=20), legend=dict(y=1, yanchor="top", x=1.02, xanchor="left", traceorder="reversed"))
    st.plotly_chart(fig, use_container_width=True)
    d1, d2 = st.columns(2)
    clean_loc = get_clean_city_name().replace(" ", "_").replace(",", "").replace("__", "_")
    with d1:
        pass # removed to_image SVG export as per earlier fix to prevent kaleido crash
    with d2:
        try:
            st.download_button(f"📥 Download {title_suffix} (HTML)", fig.to_html(include_plotlyjs="cdn").encode("utf-8"), f"{clean_loc}_{col}_{key_suffix}.html", "text/html", key=f"dl_{col}_{key_suffix}_html")
        except Exception: pass

def _render_heatmap"""

    content = content.replace("def _render_heatmap", new_func)

    # Replace UTCI calls
    utci_old = '''    _render_bar_chart(temp_df, "utci_index", "UTCI (Bar Chart)", "UTCI (°C)", "purple", "utci_bar")
    _render_daily_scatter(temp_df, "utci_index", "UTCI Daily scatter", "UTCI (°C)", "purple", "utci_scat")
    
    # Custom divergent scale for UTCI limits (extreme cold to extreme heat)
    _render_heatmap(temp_df, "utci_index", "UTCI Annual Heatmap (Day × Hour)", "°C", "Turbo")'''
    
    utci_new = '''    _render_categorical_heatmap(temp_df, "utci_index", "UTCI Annual Heatmap", _UTCI_BANDS, "utci")'''
    content = content.replace(utci_old, utci_new)

    # Replace PMV calls
    pmv_old = '''    _render_bar_chart(temp_df, "pmv_index", "PMV (Bar Chart)", "PMV", "teal", "pmv_bar")
    _render_daily_scatter(temp_df, "pmv_index", "PMV Daily scatter", "PMV", "teal", "pmv_scat")
    
    # Custom divergent scale for PMV limits (-3 to +3)
    _render_heatmap(temp_df, "pmv_index", "PMV Annual Heatmap (Day × Hour)", "Vote", "RdYlBu_r")'''
    
    pmv_new = '''    _render_categorical_heatmap(temp_df, "pmv_index", "PMV Annual Heatmap", _PMV_BANDS, "pmv")'''
    content = content.replace(pmv_old, pmv_new)

    # Replace DI calls
    di_old = '''    _render_bar_chart(temp_df, "di_index", "DI (Bar Chart)", "DI", "orange", "di_bar")
    _render_daily_scatter(temp_df, "di_index", "DI Daily scatter", "DI", "orange", "di_scat")
    
    # Custom divergent scale for DI limits
    _render_heatmap(temp_df, "di_index", "DI Annual Heatmap (Day × Hour)", "DI", "RdYlBu_r")'''
    
    di_new = '''    _render_categorical_heatmap(temp_df, "di_index", "DI Annual Heatmap", _DI_BANDS, "di")'''
    content = content.replace(di_old, di_new)

    with open("app.py", "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    main()
