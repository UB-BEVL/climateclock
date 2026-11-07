from __future__ import annotations

import unittest

import pandas as pd

from models import weather_intelligence as wi


class WeatherIntelligenceTests(unittest.TestCase):
    def test_aqi_categories(self):
        self.assertEqual(wi.aqi_category(42), "Good")
        self.assertEqual(wi.aqi_category(110), "Unhealthy for sensitive groups")
        self.assertEqual(wi.aqi_category(310), "Hazardous")
        self.assertEqual(wi.aqi_category(None), "Unavailable")

    def test_screening_alerts_find_rain_wind_heat_and_air_quality(self):
        forecast = pd.DataFrame({
            "temp_forecast": [31.0] * 13,
            "windgust_forecast": [23.0, 10.0] + [5.0] * 11,
        })
        daily = pd.DataFrame({"precip_sum": [55.0] + [2.0] * 9})
        air = pd.DataFrame({"us_aqi": [160.0] + [40.0] * 23})

        alerts = wi.build_screening_alerts(forecast, daily, air, heat_threshold_c=30)
        titles = {item["title"] for item in alerts}

        self.assertIn("Heat exposure window", titles)
        self.assertIn("Heavy-rain screening signal", titles)
        self.assertIn("Strong-wind screening signal", titles)
        self.assertIn("Air-quality screening signal", titles)

        fahrenheit_alerts = wi.build_screening_alerts(
            forecast, daily, air, heat_threshold_c=30, temperature_unit="F"
        )
        heat_detail = next(item["detail"] for item in fahrenheit_alerts if item["title"] == "Heat exposure window")
        self.assertIn("86 °F", heat_detail)

    def test_context_separates_typical_forecast_and_scenario_data(self):
        climate = pd.DataFrame({
            "drybulb": [10.0, 20.0],
            "relhum": [40.0, 60.0],
            "windspd": [2.0, 4.0],
        })
        forecast = pd.DataFrame({
            "temp_forecast": [22.0, 30.0],
            "apparent_forecast": [23.0, 33.0],
            "precip_forecast": [0.0, 4.5],
        })
        header = {"location": {"city": "Buffalo", "latitude": 42.9, "longitude": -78.8}}

        context = wi.build_research_context(
            climate,
            header,
            forecast,
            future_summary={"scenario": "SSP2-4.5", "target_year": 2050},
        )

        self.assertIn("Location: Buffalo", context)
        self.assertIn("Typical-year dry-bulb temperature", context)
        self.assertIn("10-day forecast temperature", context)
        self.assertIn("Selected future scenario", context)
        self.assertIn("not official weather warnings", context)

    def test_extracts_openai_compatible_and_direct_answers(self):
        self.assertEqual(
            wi._extract_answer({"choices": [{"message": {"content": "Evidence summary"}}]}),
            "Evidence summary",
        )
        self.assertEqual(wi._extract_answer({"answer": "Direct response"}), "Direct response")


if __name__ == "__main__":
    unittest.main()
