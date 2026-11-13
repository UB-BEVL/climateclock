"""Install the pinned official FWG distribution: python tools/install_fwg.py."""
from pathlib import Path
import hashlib
import urllib.request
ROOT = Path(__file__).resolve().parents[1]
NAME = "FutureWeatherGenerator-CMIP6-Global-v5.2.0.jar"
URL = "https://future-weather-generator.adai.pt/app_releases/" + NAME
if __name__ == "__main__":
    target = ROOT / "vendor/fwg" / NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        partial = target.with_suffix(".jar.part")
        with urllib.request.urlopen(URL, timeout=60) as response, partial.open("wb") as output:
            while chunk := response.read(4*1024*1024): output.write(chunk)
        partial.replace(target)
    checksum = hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
    print(f"FWG installed: {target}\nSHA256: {checksum}")
    print("Java 21 must be available on PATH, or set FWG_JAVA to its executable.")
