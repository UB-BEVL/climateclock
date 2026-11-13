import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import pandas as pd
from models import fwg

class FWGTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"drybulb": [20.0]*8760}, index=pd.date_range("2021-01-01", periods=8760, freq="h"))
        self.header = {"location": {"city": "Test", "latitude": 42.9, "longitude": -78.8}}

    def test_engine_outputs_cached_and_scenario_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); jar = root/"engine.jar"; jar.write_bytes(b"test")
            calls = []
            def run(command, **kwargs):
                if command[0] == "compact.exe":
                    return type("Result",(),{"returncode":0})()
                calls.append(command)
                output = Path(next(x.split("=",1)[1] for x in command if x.startswith("-output_folder=")))
                scenario = next(x.split("=",1)[1] for x in command if x.startswith("-scenarios="))
                future = self.frame.copy(); future["drybulb"] += 2
                for year in (2050,2080):
                    (output/f"ensemble_{scenario}_{year}.epw").write_bytes(fwg.epw._compose_epw_text(self.header,future))
                return type("Result",(),{"returncode":0})()
            with patch.object(fwg,"runtime",return_value=("java",jar)), patch.dict("os.environ",{"FWG_CACHE":str(root/"cache")}), patch.object(fwg.subprocess,"run",side_effect=run):
                result=fwg.generate(self.frame,self.header,"SSP2-4.5")
                self.assertEqual(result[2050]["df"]["drybulb"].mean(),22)
                fwg.generate(self.frame,self.header,"SSP2-4.5")
                self.assertEqual(len(calls),1)
                fwg.generate(self.frame,self.header,"SSP5-8.5")
                self.assertEqual(len(calls),2)
                fwg.generate(self.frame,self.header,"SSP5-8.5",models=["BCC-CSM2-MR","CanESM5"])
                self.assertEqual(len(calls),3)

    def test_failed_engine_never_returns_synthetic_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); jar=root/"engine.jar"; jar.write_bytes(b"test")
            with patch.object(fwg,"runtime",return_value=("java",jar)), patch.dict("os.environ",{"FWG_CACHE":str(root/"cache")}), patch.object(fwg.subprocess,"run",return_value=type("Result",(),{"returncode":1})()):
                with self.assertRaisesRegex(RuntimeError,"FWG failed"):
                    fwg.generate(self.frame,self.header,"SSP2-4.5")
                self.assertEqual(list((root/"cache").rglob("complete.json")),[])

    def test_partial_year_rejected(self):
        with patch.object(fwg,"runtime",return_value=("java",Path("unused"))):
            with self.assertRaisesRegex(ValueError,"complete annual"):
                fwg.generate(self.frame.iloc[:24],self.header,"SSP2-4.5")
