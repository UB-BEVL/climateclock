"""Report assembly must keep each analysis once across dashboard captures."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plotly.graph_objects as go
import app


def chart(title, values=(1, 2, 3)):
    return go.Figure(go.Bar(x=['Jan', 'Feb', 'Mar'], y=values)).update_layout(title=title)


def flattened(figures):
    return [(section['section'], item['title'], item['raw_key'])
            for section in app._resolve_report_sections(figures) for item in section['items']]


class ReportDuplicateTests(unittest.TestCase):
    def test_reported_pairs_each_have_one_canonical_page(self):
        names = ['Overview Annual Wind Rose', 'Annual Wind Rose', 'drybulb Monthly Bar',
                 'Monthly Average Temperature', 'relhum Monthly Bar', 'Monthly Avg Humidity',
                 'Monthly Wind Speed', 'Monthly Avg Wind Speed']
        figures = {name: chart(name) for name in names}
        rows = flattened(figures)
        self.assertEqual(len(rows), 4)
        self.assertEqual({title for _, title, _ in rows}, {'Annual Wind Rose', 'Monthly Dry-Bulb Temperature',
                         'Monthly Relative Humidity', 'Monthly Mean Wind Speed'})
        self.assertTrue(all(section != 'Additional Captured Figures' for section, _, _ in rows))
        self.assertIn(('Wind Data', 'Annual Wind Rose', 'Annual Wind Rose'), rows)

    def test_detailed_capture_wins_in_either_insertion_order(self):
        detailed = chart('Monthly Average Temperature')
        detailed.add_scatter(x=['Jan', 'Feb', 'Mar'], y=[-2, 0, 1], name='Minimum')
        basic = chart('drybulb Monthly Bar')
        for entries in ([('drybulb Monthly Bar', basic), ('Monthly Average Temperature', detailed)],
                        [('Monthly Average Temperature', detailed), ('drybulb Monthly Bar', basic)]):
            result = app._deduplicate_report_figures(dict(entries))
            self.assertEqual(list(result), ['Monthly Average Temperature'])
            self.assertIs(next(iter(result.values())), detailed)

    def test_detailed_auto_capture_beats_basic_manual_variant(self):
        detailed = chart('Monthly Avg Humidity')
        with patch.object(app.st, 'session_state', {'pdf_figures': {'relhum Monthly Bar': chart('basic')},
                          'pdf_figures_auto': {'Monthly Avg Humidity': detailed}}):
            result = app._merged_pdf_figures()
        self.assertEqual(list(result), ['Monthly Avg Humidity'])
        self.assertIs(result['Monthly Avg Humidity'], detailed)

    def test_legacy_numbered_snapshots_do_not_become_appendix_pages(self):
        names = ['Monthly Avg Wind Speed', 'Monthly Avg Wind Speed (2)', 'Monthly Avg Wind Speed (3)']
        figures = {name: chart(name, (i, i+1, i+2)) for i, name in enumerate(names)}
        self.assertEqual(flattened(figures), [('Wind Data', 'Monthly Mean Wind Speed', names[-1])])

    def test_distinct_heatmaps_seasons_and_custom_cases_survive(self):
        names = ['Annual Diurnal Resource Heatmap', 'Annual Diurnal Resource Heatmap — Rain and Wind',
                 'Annual Wind Rose', 'Seasonal Wind Roses', 'Diurnal Wind Roses',
                 'drybulb Annual Heatmap', 'relhum Annual Heatmap', 'Custom case (1)', 'Custom case (2)']
        rows = flattened({name: chart(name) for name in names})
        self.assertEqual({key for _, _, key in rows}, set(names))

    def test_canonical_titles_without_internal_alias_still_resolve(self):
        rows = flattened({'Monthly Relative Humidity': chart('Humidity')})
        self.assertEqual(rows, [('Temperature & Humidity', 'Monthly Relative Humidity', 'Monthly Relative Humidity')])

    def test_capture_reruns_replace_known_chart_including_reverting_values(self):
        state = {'nav_page': 'Overview', 'pdf_figures_auto': {}, 'pdf_figure_fingerprints': set()}
        with patch.object(app.st, 'session_state', state):
            for values in ((1, 2, 3), (4, 5, 6), (1, 2, 3)):
                app._capture_plotly_figure(chart('Monthly Avg Humidity', values))
                self.assertEqual(list(state['pdf_figures_auto']), ['Monthly Avg Humidity'])
                self.assertEqual(tuple(state['pdf_figures_auto']['Monthly Avg Humidity'].data[0].y), values)


if __name__ == '__main__':
    unittest.main(verbosity=2)
