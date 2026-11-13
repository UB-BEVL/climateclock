"""Project-local CPU model service. Only explicitly managed loopback configs start it."""
import os
from pathlib import Path
import subprocess
import threading
import time
from urllib.parse import urlsplit
import requests
ROOT = Path(__file__).resolve().parents[1]
MODEL = "qwen2.5:1.5b"
_lock = threading.Lock()
_process = None

def ensure_running(api_url, model):
    global _process
    endpoint = urlsplit(api_url)
    if endpoint.hostname not in ("127.0.0.1", "localhost") or endpoint.port != 11434 or model != MODEL:
        raise ValueError("Managed local AI requires localhost:11434 and qwen2.5:1.5b.")
    base = "http://127.0.0.1:11434"
    with _lock:
        loading = False
        try:
            response = requests.get(base + "/v1/models", timeout=2)
            if response.ok:
                names = [m.get("id") for m in response.json().get("data", [])]
                if model not in names: raise RuntimeError("Another model service is using port 11434. Stop it or update the assistant configuration.")
                return
            loading = response.status_code == 503
        except requests.ConnectionError:
            pass
        except requests.Timeout:
            loading = True
        executable = ROOT / "vendor/llama/llama-server.exe"
        weights = ROOT / "vendor/llama/qwen2.5-1.5b.gguf"
        if not executable.exists() or not weights.exists():
            raise RuntimeError("The portable local model files are missing from vendor/llama.")
        if not loading and (_process is None or _process.poll() is not None):
            logdir = ROOT / "weather_cache/logs"
            logdir.mkdir(parents=True, exist_ok=True)
            command = [str(executable), "--model", str(weights), "--alias", MODEL,
                       "--host", "127.0.0.1", "--port", "11434", "--ctx-size", "8192",
                       "--parallel", "1", "--threads", "4", "--n-gpu-layers", "0"]
            with (logdir / "local-ai.log").open("a", encoding="utf-8") as log:
                _process = subprocess.Popen(command, cwd=executable.parent,
                                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for _ in range(60):
            if _process is not None and _process.poll() is not None:
                raise RuntimeError("The local model service stopped. See weather_cache/logs/local-ai.log.")
            try:
                response = requests.get(base + "/v1/models", timeout=2)
                if response.ok and model in [m.get("id") for m in response.json().get("data",[])]: return
            except requests.RequestException:
                pass
            time.sleep(0.5)
        raise RuntimeError("The local model is still loading. Retry in a moment.")
