"""Official FWG CLI adapter. No synthetic fallback."""
from __future__ import annotations
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile
import pandas as pd
from models import future_epw as epw

ROOT = Path(__file__).resolve().parents[1]
JAR_NAME = "FutureWeatherGenerator-CMIP6-Global-v5.2.0.jar"

def runtime():
    jar = Path(os.environ.get("FWG_JAR", str(ROOT / "vendor/fwg" / JAR_NAME)))
    java = os.environ.get("FWG_JAVA") or shutil.which("java")
    if not java:
        java = next((str(p) for p in (ROOT / "vendor/java").glob("*/bin/java.exe")), None)
    if not jar.is_file():
        raise RuntimeError("Official FWG CMIP6 engine is not installed. Configure FWG_JAR with the downloaded distribution.")
    if not java:
        raise RuntimeError("FWG requires Java 21. Configure FWG_JAVA or install Java on PATH.")
    compatibility = ROOT / "assets/fwg-windows-compat.jar"
    if os.name == "nt" and compatibility.exists():
        with zipfile.ZipFile(jar) as archive:
            grid_hash = hashlib.sha256(archive.read("futureweathergenerator/io/GridCache.class")).hexdigest()
        if grid_hash != "238b5233d700b802792d66be5db00ea421e538341bc98132fa3a7630be32b248":
            raise RuntimeError("The Windows compatibility patch requires FWG CMIP6 Global v5.2.0. Install the pinned distribution.")
    return java, jar

def parse_output(path, baseline):
    data = path.read_bytes()
    if not data.startswith(b"LOCATION,"):
        raise ValueError(f"FWG output is not an EPW: {path.name}")
    frame = pd.read_csv(io.BytesIO(data), skiprows=8, header=None)
    if len(frame.columns) != len(epw.EPW_COLUMNS) or len(frame) != len(baseline):
        raise ValueError(f"FWG output has unexpected columns or hour count: {path.name}")
    frame.columns = epw.EPW_COLUMNS
    for col in frame.columns:
        if col != "datasource": frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if frame["drybulb"].isna().any():
        raise ValueError("FWG output contains invalid temperatures.")
    frame.index = baseline.index
    return {"file_name": path.name, "bytes": data, "df": frame}

DEFAULT_MODELS = ("CanESM5", "CNRM-CM6-1")

def available_models():
    _, jar = runtime()
    with zipfile.ZipFile(jar) as archive:
        catalog = json.loads(archive.read("futureweathergenerator/resources/climate-model-catalog.json"))
    return [model["id"] for model in catalog["models"]]

def generate(baseline, header, scenario_label, raw_epw=None, models=None):
    java, jar = runtime()
    scenario = epw.SCENARIO_MAP[scenario_label]
    models = tuple(sorted(set(models if models is not None else DEFAULT_MODELS)))
    if len(models) < 2 or any(not re.fullmatch(r"[A-Za-z0-9_-]+", model) for model in models):
        raise ValueError("Select at least two valid climate models for the ensemble.")
    if len(baseline) not in (8760, 8784):
        raise ValueError("FWG requires a complete annual EPW. Load a full-year weather file.")
    source = raw_epw if raw_epw else epw._compose_epw_text(header, baseline)
    if isinstance(source, str): source = source.encode("utf-8")
    source = bytes(source)
    compatibility = ROOT / "assets/fwg-windows-compat.jar"
    compatibility_bytes = compatibility.read_bytes() if os.name == "nt" and compatibility.exists() else b""
    signature = hashlib.sha256(source + scenario.encode() + repr(models).encode() + compatibility_bytes + str((jar.resolve(), jar.stat().st_size, jar.stat().st_mtime_ns)).encode()).hexdigest()
    cache = Path(os.environ.get("FWG_CACHE", str(Path(tempfile.gettempdir()) / "climateclock-fwg")))
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / signature
    if not (destination / "complete.json").exists():
        with tempfile.TemporaryDirectory(prefix="fwg-", dir=cache) as temporary:
            run = Path(temporary)
            source_path = run / "baseline.epw"
            source_path.write_bytes(source)
            output = run / "results"
            output.mkdir()
            extracted = run / "model-cache"
            extracted.mkdir()
            if os.name == "nt":
                # FWG expands all scenarios for each model. NTFS compression keeps
                # these repetitive CSVs small without changing their contents.
                subprocess.run(["compact.exe", "/C", "/I", "/Q", str(extracted)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False)

            engine = ["-jar", str(jar.resolve())]
            if compatibility_bytes:
                engine = ["-cp", os.pathsep.join([str(compatibility.resolve()), str(jar.resolve())]), "futureweathergenerator.Main"]
            command = [java, "-Xmx2g", "-Djava.awt.headless=true", *engine, "-g",
                       f"-epw={source_path.resolve()}", f"-output_folder={output.resolve()}",
                       "-ensemble=true", f"-models={','.join(models)}", f"-scenarios={scenario}", "-timeframes=2050,2080",
                       "-output_type=EPW", "-output_profile=STANDARD", "-output_layout=PER_RUN_ORGANIZED",
                       f"-model_unzipped_folder={(run / 'model-cache').resolve()}",
                       "-multithread=false", "-save_run_manifest=true", "-save_technical_log=true"]
            log = run / "engine.log"
            try:
                with log.open("w", encoding="utf-8") as handle:
                    completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, timeout=1800,
                                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("FWG exceeded the 30-minute run limit; no results were cached.") from exc
            if completed.returncode:
                failure = cache / "failures" / run.name
                failure.mkdir(parents=True, exist_ok=True)
                shutil.copy2(log, failure / "engine.log")
                for number, detail in enumerate(output.rglob("*.log")):
                    shutil.copy2(detail, failure / f"detail-{number}.log")
                content = log.read_text(encoding="utf-8", errors="replace")
                errors = [line for line in content.splitlines() if any(term in line.lower() for term in ("severe", "exception", "caused by", "error", "space", "denied"))]
                raise RuntimeError("FWG failed: " + "\n".join(errors[-10:]) + f" Full diagnostics: {failure}")
            # The engine's extracted resources are disposable, not run artifacts.
            extracted = (run / "model-cache").resolve()
            if extracted.parent != run.resolve():
                raise RuntimeError("Unexpected FWG extraction directory")
            if extracted.exists():
                shutil.rmtree(extracted)
            selected = {}
            for year in epw.TARGET_YEARS:
                matches = [p for p in output.rglob("*.epw") if str(year) in p.name and scenario.lower() in p.name.lower()]
                if len(matches) != 1:
                    raise RuntimeError(f"Expected one FWG ensemble EPW for {year}; found {len(matches)}. " + log.read_text(encoding="utf-8", errors="replace")[-1500:])
                parse_output(matches[0], baseline)
                selected[str(year)] = str(matches[0].relative_to(run))
            (run / "complete.json").write_text(json.dumps(selected), encoding="utf-8")
            # Publish only a fully validated run. Separate concurrent runs are harmless.
            try: run.rename(destination)
            except FileExistsError:
                if not (destination / "complete.json").exists(): raise RuntimeError("Another FWG run is publishing results. Retry shortly.")
    selected = json.loads((destination / "complete.json").read_text(encoding="utf-8"))
    return {int(year): parse_output(destination / path, baseline) for year, path in selected.items()}
