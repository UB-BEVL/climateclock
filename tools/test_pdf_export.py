import os
from pathlib import Path
import plotly.express as px

# Import app module from workspace
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app

print('Running PDF export test...')

# Sample figures
try:
    df = px.data.iris()
    fig1 = px.scatter(df, x='sepal_width', y='sepal_length', color='species', title='Test Scatter')
    import numpy as np
    heat = np.random.rand(24, 12)
    fig2 = px.imshow(heat, color_continuous_scale='Viridis', title='Random Heatmap')
except Exception as e:
    print('Failed to build sample figures:', e)
    raise

# Apply app styling if available
for name, fig in (('scatter', fig1), ('heatmap', fig2)):
    try:
        fig = app._apply_global_plot_style(fig)
    except Exception:
        pass

    try:
        path = app._fig_to_tmp_png(fig, width=1400, height=730, scale=2)
        print(f'{name} -> {path}')
    except Exception as e:
        print(f'{name} export failed: {e}')

print('Test complete.')
