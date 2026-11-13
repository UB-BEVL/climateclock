import sys
import types
import unittest
from unittest.mock import patch
import numpy as np
from metrics.comfort_energy import compute_utci_values

class UTCICompatibilityTests(unittest.TestCase):
    def run_model(self, model):
        package = types.ModuleType("pythermalcomfort")
        models = types.ModuleType("pythermalcomfort.models")
        models.utci = model
        with patch.dict(sys.modules, {"pythermalcomfort": package, "pythermalcomfort.models": models}):
            np.testing.assert_allclose(compute_utci_values([25], [25], [1], [50]), [24.6])

    def test_legacy_numpy_two_rounding(self):
        def legacy(tdb, tr, v, rh, **kwargs):
            values = np.array([24.567])
            if kwargs.get("round", True):
                raise AttributeError("np.round_ was removed in NumPy 2.0")
            self.assertTrue(kwargs["limit_inputs"])
            return {"utci": values}
        self.run_model(legacy)

    def test_current_rounding(self):
        def current(tdb, tr, v, rh, limit_inputs=True, round_output=True):
            self.assertFalse(round_output)
            self.assertTrue(limit_inputs)
            return types.SimpleNamespace(utci=np.array([24.567]))
        self.run_model(current)

    def test_strict_legacy_signature_and_removed_numpy_alias(self):
        def legacy(tdb, tr, v, rh, units="SI", return_stress_category=False, limit_inputs=True):
            return np.round_(np.array([24.567]), 1)
        self.run_model(legacy)
