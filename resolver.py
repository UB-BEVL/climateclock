from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math
import pandas as pd

@dataclass
class QueryPoint:
    lat: float
    lon: float

def geocode(query: str, user_agent: str = "epw-catalog") -> Optional[QueryPoint]:
    # Import here so it works even if you install geopy later in the session
    from geopy.geocoders import Nominatim
    geocoder = Nominatim(user_agent=user_agent, timeout=10)
    loc = geocoder.geocode(query)
    if not loc:
        return None
    return QueryPoint(lat=loc.latitude, lon=loc.longitude)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0088  # km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*cos(phi2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(math.sqrt(a))

def nearest_from_catalog(catalog_csv: str, q: QueryPoint, k: int = 8) -> pd.DataFrame:
    df = pd.read_csv(catalog_csv)
    if {"lat","lon"}.issubset(df.columns):
        ok = df.dropna(subset=["lat","lon"]).copy()
    else:
        ok = df.copy(); ok["lat"] = float("nan"); ok["lon"] = float("nan")
    ok["distance_km"] = [
        haversine(q.lat, q.lon, la, lo) if pd.notna(la) and pd.notna(lo) else float("inf")
        for la, lo in zip(ok["lat"], ok["lon"])
    ]
    ok = ok.sort_values("distance_km").head(k)
    cols = ["name","country","kind","years","epw_url","zip_url","distance_km"]
    return ok[[c for c in cols if c in ok.columns]]
