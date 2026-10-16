import unittest

import pandas as pd

from models import future_epw


class FutureEPWExportTests(unittest.TestCase):
    def setUp(self):
        self.weather = pd.DataFrame(
            {"drybulb": [20.0], "relhum": [50.0]},
            index=pd.date_range("2021-01-01", periods=1, freq="h"),
        )

    def test_missing_header_values_generate_both_downloads(self):
        header = {
            "location": {"city": "Test City", "wmo": 725280, "source": None},
            "data_periods": None,
            "comments1": None,
        }
        outputs = future_epw.generate_download_payloads(
            self.weather, header, "SSP2-4.5",
            future_epw._synthetic_cmip6_deltas(), None, False, False,
        )
        for year in future_epw.TARGET_YEARS:
            lines = outputs[year]["bytes"].decode("latin-1").splitlines()
            self.assertEqual(len(lines), 9)
            self.assertEqual(lines[1], "DESIGN CONDITIONS,0")
            self.assertTrue(lines[5].startswith("COMMENTS 1,"))
            self.assertTrue(lines[7].startswith("DATA PERIODS,"))
            self.assertEqual(len(lines[8].split(",")), 35)
            self.assertGreater(outputs[year]["df"]["drybulb"].iloc[0], 20)

    def test_original_header_records_are_preserved(self):
        records = [
            "LOCATION,Original station",
            "DESIGN CONDITIONS,0", "TYPICAL/EXTREME PERIODS,0",
            "GROUND TEMPERATURES,0", "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0",
            "COMMENTS 1,Original comment", "COMMENTS 2,More metadata",
            "DATA PERIODS,1,1,Data,Sunday,1/1,12/31",
        ]
        data = future_epw._compose_epw_text(
            {"location": {}, "_epw_header_lines": records, "data_periods": None},
            self.weather,
        )
        self.assertEqual(data.decode("latin-1").splitlines()[:8], records)


if __name__ == "__main__":
    unittest.main()
