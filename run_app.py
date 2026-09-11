"""Run with `py run_app.py`; use a private Python 3.11/3.12 environment."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
ENVIRONMENT = ROOT / ".venv-climateclock"
PYTHON = ENVIRONMENT / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def supported(command: list[str]) -> bool:
    try:
        version = subprocess.check_output(
            [*command, "-c", "import sys; print('%s.%s' % sys.version_info[:2])"],
            text=True, stderr=subprocess.DEVNULL, timeout=15,
        ).strip()
        return version in {"3.11", "3.12"}
    except (OSError, subprocess.SubprocessError):
        return False


def find_python() -> list[str]:
    candidates = [[sys.executable]]
    if os.name == "nt":
        programs = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local"))) / "Programs/Python"
        candidates += [[str(programs / f"Python{v}" / "python.exe")] for v in ("312", "311")]
        launcher = shutil.which("py")
        if launcher:
            candidates += [[launcher, "-3.12"], [launcher, "-3.11"]]
    else:
        candidates += [[f"python{v}"] for v in ("3.12", "3.11")]
    for command in candidates:
        if supported(command):
            return command
    raise RuntimeError("Install Python 3.12, then run this launcher again. This project's dependencies use Python 3.11 or 3.12.")


def setup() -> None:
    if not ENVIRONMENT.exists():
        command = find_python()
        print(f"Creating the ClimateClock environment with {command[0]}", flush=True)
        subprocess.run([*command, "-m", "venv", str(ENVIRONMENT)], check=True)
    if not supported([str(PYTHON)]):
        raise RuntimeError(f"{ENVIRONMENT} is not a working Python 3.11/3.12 environment. Rename that folder and run this launcher again.")

    requirements = ROOT / "requirements.txt"
    fingerprint = hashlib.sha256(requirements.read_bytes()).hexdigest()
    marker = ENVIRONMENT / ".requirements.sha256"
    if not marker.exists() or marker.read_text().strip() != fingerprint:
        print("Installing the project's dependencies (first run or requirements changed)...", flush=True)
        subprocess.run(
            [str(PYTHON), "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir", "--no-compile", "-r", str(requirements)],
            cwd=ROOT, check=True,
        )
        subprocess.run([str(PYTHON), "-m", "pip", "check"], cwd=ROOT, check=True)
        subprocess.run(
            [str(PYTHON), "-c", "from pythermalcomfort.models import utci, pmv_ppd_ashrae; print('UTCI and PMV imports OK')"],
            cwd=ROOT, check=True,
        )
        marker.write_text(fingerprint + "\n")


def main() -> int:
    setup()
    arguments = sys.argv[1:]
    if arguments == ["--setup-only"]:
        print(f"Ready: {PYTHON}", flush=True)
        return 0
    print(f"Starting ClimateClock with {PYTHON}", flush=True)
    return subprocess.call([str(PYTHON), "-m", "streamlit", "run", str(ROOT / "app.py"), *arguments], cwd=ROOT)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        pass  # Streamlit receives the console interrupt and shuts down as well.
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ClimateClock could not start: {exc}", file=sys.stderr)
        raise SystemExit(1)
