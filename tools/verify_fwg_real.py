"""End-to-end check of the actual FWG ensemble; no mocks or synthetic fallback."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import sys
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from models import fwg
from models.future_epw import EPW_COLUMNS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.epw.read_bytes()
    frame = pd.read_csv(io.BytesIO(source), skiprows=8, header=None, names=EPW_COLUMNS)
    frame.index = pd.date_range("2020-01-01" if len(frame)==8784 else "2021-01-01", periods=len(frame), freq="h")
    print("Running official FWG SSP2-4.5 ensemble for 2050 and 2080...", flush=True)
    try:
        results = fwg.generate(frame, {}, "SSP2-4.5", raw_epw=source)
        report = {"input": str(args.epw), "input_sha256": hashlib.sha256(source).hexdigest(), "models": list(fwg.DEFAULT_MODELS), "outputs": {}}
        for year in (2050,2080):
            result = results[year]
            generated = result["df"]
            if not generated.index.equals(frame.index): raise ValueError("Hourly index mismatch")
            if len(generated)!=len(frame): raise ValueError("Hour count mismatch")
            if generated["drybulb"].isna().any(): raise ValueError("Invalid temperature data")
            if (generated["drybulb"].to_numpy()==frame["drybulb"].to_numpy()).all(): raise ValueError("Generated temperatures are identical to the baseline")
            path = args.output / result["file_name"]
            path.write_bytes(result["bytes"])
            report["outputs"][year] = {"file":str(path),"hours":len(generated),"sha256":hashlib.sha256(result["bytes"]).hexdigest(),"mean_temperature_c":float(generated["drybulb"].mean())}
        if report["outputs"][2050]["sha256"]==report["outputs"][2080]["sha256"]: raise ValueError("2050 and 2080 files are identical")
        (args.output/"verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        (args.output/"failure.txt").unlink(missing_ok=True)
        print("PASS: both official FWG EPWs generated and validated.",flush=True)
    except Exception:
        (args.output/"failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise
if __name__=="__main__": main()
