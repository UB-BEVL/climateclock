"""Regression checks for missing UTCI data across chart and PDF serialization."""
from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pdf_chart_renderer import (
    ChartRenderer, ChartRenderError, _decode_plotly_arrays, _restore_numeric_marker_gaps,
)
import app


def seasonal_weather():
    times = pd.to_datetime([
        '2021-01-01', '2021-01-02', '2021-04-01', '2021-04-02',
        '2021-07-01', '2021-07-02', '2021-07-03', '2021-07-04',
        '2021-10-01', '2021-10-02',
    ])
    return pd.DataFrame({'drybulb_C': np.arange(10.) + 10, 'rh_pct': 50.,
                         'atmos_pressure': 101325.}, index=times)


class MissingValueTests(unittest.TestCase):
    def test_json_gaps_stay_missing_and_preserve_values_and_positions(self):
        figure = {'data': [{'type': 'scatter', 'x': [1, 2, 3], 'y': [3, 4, 5],
                           'marker': {'color': [12.5, None, 28], 'line': {'color': [None, 4, 5]}}}]}
        restored = _restore_numeric_marker_gaps(figure)['data'][0]
        self.assertEqual(restored['x'], [1, 2, 3])
        self.assertEqual(restored['y'], [3, 4, 5])
        self.assertEqual(restored['marker']['color'][::2], [12.5, 28])
        self.assertTrue(np.isnan(restored['marker']['color'][1]))
        self.assertTrue(np.isnan(restored['marker']['line']['color'][0]))

    def test_mixed_seasons_retain_unknown_points_without_invalid_colors(self):
        frame = seasonal_weather()
        values = np.array([np.nan, np.nan, 10., 30., 20., 35., np.nan, np.inf, 9., 26.])
        fig, caption = app.build_fig_g_seasonal_psychrometric(frame, values, 'Test station')
        known = [trace for trace in fig.data if trace.mode == 'markers' and trace.name != 'UTCI unavailable']
        unknown = [trace for trace in fig.data if trace.name == 'UTCI unavailable']
        self.assertEqual(sum(len(trace.x) for trace in known), 6)
        self.assertEqual(sum(len(trace.x) for trace in unknown), 4)
        self.assertEqual(sum(bool(trace.showlegend) for trace in unknown), 1)
        for trace in known:
            self.assertTrue(np.isfinite(trace.marker.color).all())
        self.assertTrue(fig.layout.coloraxis.showscale)  # Winter has no UTCI; other seasons still get a colorbar.
        self.assertIn('50.0%', caption)
        self.assertIn('2 valid of 4 plotted summer hours', caption)
        self.assertIn('4 hours without valid UTCI', caption)
        go.Figure(_decode_plotly_arrays(json.loads(fig.to_json())))

    def test_all_missing_utci_keeps_weather_points_and_reports_unavailable(self):
        frame = seasonal_weather()
        fig, caption = app.build_fig_g_seasonal_psychrometric(frame, np.full(len(frame), np.nan), 'Test station')
        self.assertEqual(sum(len(trace.x) for trace in fig.data if trace.mode == 'markers'), len(frame))
        self.assertFalse(fig.layout.coloraxis.showscale)
        self.assertIn('percentage is unavailable', caption)
        self.assertNotIn('0.0%', caption)

    def test_invalid_weather_is_omitted_without_shifting_colors(self):
        frame = seasonal_weather()
        frame.iloc[0, frame.columns.get_loc('drybulb_C')] = np.nan
        frame.iloc[1, frame.columns.get_loc('rh_pct')] = -1
        values = np.arange(len(frame), dtype=float) + 15
        fig, caption = app.build_fig_g_seasonal_psychrometric(frame, values, 'Test station')
        points = [trace for trace in fig.data if trace.mode == 'markers']
        self.assertEqual(sum(len(trace.x) for trace in points), len(frame) - 2)
        actual = [(float(x), float(color)) for trace in points for x, color in zip(trace.x, trace.marker.color)]
        self.assertEqual(sorted(actual), [(float(i+10), float(i+15)) for i in range(2, 10)])
        self.assertIn('2 hours with invalid weather coordinates', caption)

    def test_comfort_band_boundaries_match_overview(self):
        frame = seasonal_weather()
        values = np.array([20.] * 4 + [9., 26., 8.9, 26.1] + [20.] * 2)
        _, caption = app.build_fig_g_seasonal_psychrometric(frame, values, 'Test station')
        self.assertIn('50.0%', caption)

    def test_real_worker_renders_null_numeric_colors_and_checks_invalid_colors(self):
        with ChartRenderer() as renderer:
            for colors in ([12, None, 28], [None, None, None]):
                request = {'figure': json.dumps({'data': [{'type': 'scatter', 'x': [1, 2, 3], 'y': [2, 3, 4],
                            'mode': 'markers', 'marker': {'color': colors, 'colorscale': 'Viridis'}}]}),
                           'width': 800, 'height': 500, 'scale': 1}
                png = renderer._request(request)
                self.assertTrue(png.startswith(b'\x89PNG'))
                self.assertGreater(len(png), 5000)
            request['figure'] = json.dumps({'data': [{'type': 'scatter', 'x': [1, 2], 'y': [2, 3],
                                          'marker': {'color': ['not-a-color', None]}}]})
            with self.assertRaises(ChartRenderError):
                renderer._request(request)
            frame = seasonal_weather()
            for values in (np.array([20., np.nan] * 5), np.full(len(frame), np.nan)):
                fig, _ = app.build_fig_g_seasonal_psychrometric(frame, values, 'Test station')
                self.assertGreater(len(renderer.render(fig, 1000, 650, 1)), 5000)


if __name__ == '__main__':
    unittest.main(verbosity=2)
