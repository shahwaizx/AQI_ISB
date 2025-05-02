#!/usr/bin/env python3
"""
Fetch one year of hourly pollution data from OpenWeather, aggregate to
daily means, compute AQI and rolling features, then upload to a
Hopsworks feature‑group.

Key fixes
─────────
• All numeric columns are cast to FLOAT before insertion so Hive sees
  them as DOUBLE, preventing the schema mismatch that caused orphan
  tables (error 270001).
• Version number is read from FG_VERSION in .env (set it to a fresh
  value you’ve never used, e.g. 10).
"""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import os, time, logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests, hopsworks
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type
)

# ─── Config ─────────────────────────────────────────────────────────────
OPENWEATHER_API_KEY = os.environ["OPENWEATHER_API_KEY"]
HOPSWORKS_API_KEY   = os.environ["HOPSWORKS_API_KEY"]
HOPSWORKS_HOST      = os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai")

BACKFILL_DAYS       = int(os.getenv("BACKFILL_DAYS", 365))
LATITUDE            = float(os.getenv("LATITUDE", 33.6995))
LONGITUDE           = float(os.getenv("LONGITUDE", 73.0363))
CITY_NAME           = os.getenv("CITY_NAME", "Islamabad")

FEATURESTORE_NAME   = os.getenv("FEATURESTORE_NAME", "aqi_islamabad_featurestore")
FEATUREGROUP_NAME   = os.getenv("FEATUREGROUP_NAME", "isb_aqi_history")
FG_VERSION          = int(os.getenv("FG_VERSION", "10"))   # choose an unused number

BASE_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"

# ─── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("fetch_aqi.log")]
)
log = logging.getLogger(__name__)

# ─── Retry helpers ──────────────────────────────────────────────────────
retry_http = retry(
    retry=retry_if_exception_type(requests.RequestException),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
retry_hops = retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    reraise=True,
)

# ─── AQI breakpoints (µg m‑3 unless noted) ─────────────────────────────
BREAKPOINTS = {
    "pm2_5": [(0.0,12.0,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
              (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,500.4,301,500)],
    "pm10":  [(0,54,0,50),(55,154,51,100),(155,254,101,150),
              (255,354,151,200),(355,424,201,300),(425,604,301,500)],
    "no2":   [(0,53,0,50),(54,100,51,100),(101,360,101,150),
              (361,649,151,200),(650,1249,201,300),(1250,2049,301,500)],  # ppb
    "so2":   [(0,35,0,50),(36,75,51,100),(76,185,101,150),
              (186,304,151,200),(305,604,201,300),(605,1004,301,500)],     # ppb
    "o3":    [(0,54,0,50),(55,70,51,100),(71,85,101,150),
              (86,105,151,200),(106,200,201,300)],                        # ppb
    "co":    [(0.0,4.4,0,50),(4.5,9.4,51,100),(9.5,12.4,101,150),
              (12.5,15.4,151,200),(15.5,30.4,201,300),(30.5,50.4,301,500)],# mg m‑3
}

# ─── AQI helpers ─────────────────────────────────────────────────────────
def compute_individual_aqi(cp: float, breakpoints: list[tuple]) -> int:
    for Cl, Ch, Il, Ih in breakpoints:
        if Cl <= cp <= Ch:
            return int(((Ih - Il) / (Ch - Cl)) * (cp - Cl) + Il)
    return 500

def compute_overall_aqi(row: pd.Series) -> int:
    return max(
        compute_individual_aqi(row[p], BREAKPOINTS[p])
        for p in BREAKPOINTS if pd.notna(row[p])
    )

# ─── Fetch one day of hourly observations ───────────────────────────────
@retry_http
def fetch_day(start_ts: int, end_ts: int) -> list[dict]:
    url = (f"{BASE_URL}?lat={LATITUDE}&lon={LONGITUDE}"
           f"&start={start_ts}&end={end_ts}&appid={OPENWEATHER_API_KEY}")
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json().get("list", [])

# ─── Build the daily dataframe ──────────────────────────────────────────
def fetch_historical() -> pd.DataFrame:
    log.info("Backfilling the last %d days for %s…", BACKFILL_DAYS, CITY_NAME)
    end_dt   = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=BACKFILL_DAYS)

    rows: list[dict] = []
    dt = start_dt
    while dt < end_dt:
        s, e = int(dt.timestamp()), int((dt + timedelta(days=1)).timestamp())
        try:
            for entry in fetch_day(s, e):
                rows.append({
                    "timestamp": datetime.fromtimestamp(entry["dt"], timezone.utc),
                    **entry["components"],
                })
        except Exception as ex:
            log.warning("Skipping %s: %s", dt.date(), ex)
        dt += timedelta(days=1)
        time.sleep(0.25)

    if not rows:
        raise RuntimeError("No data fetched – aborting.")

    df = (pd.DataFrame(rows)
            .set_index("timestamp")
            .sort_index())

    # unit conversions
    df["co"] *= 1/1000                 # µg m‑3 → mg m‑3
    μg_to_ppb = 1 / 1.88
    for gas in ("no2", "so2", "o3"):
        df[gas] *= μg_to_ppb

    # outlier trimming
    df.loc[df["pm2_5"] > 600,  "pm2_5"] = np.nan
    df.loc[df["pm10"]  > 1000, "pm10"]  = np.nan

    # daily aggregation (require ≥12 hourly records)
    counts = df["pm2_5"].resample("D").count()
    daily  = df.resample("D").mean()
    daily  = daily[counts >= 12]

    # AQI & rolling features
    daily["aqi"] = daily.apply(compute_overall_aqi, axis=1)
    daily["pm2_5_idx"] = daily["pm2_5"].map(lambda v: compute_individual_aqi(v, BREAKPOINTS["pm2_5"]))
    daily["pm10_idx"]  = daily["pm10"].map(lambda v: compute_individual_aqi(v, BREAKPOINTS["pm10"]))
    for w in (3,7,14):
        daily[f"aqi_roll{w}"]   = daily["aqi"].rolling(w).mean()
        daily[f"pm2_5_roll{w}"] = daily["pm2_5"].rolling(w).mean()
        daily[f"pm10_roll{w}"]  = daily["pm10"].rolling(w).mean()

    daily = daily.dropna().reset_index()
    log.info("Prepared daily table: %d rows × %d cols", *daily.shape)
    return daily

# ─── Upload to Hopsworks ────────────────────────────────────────────────
@retry_hops
def upload_to_hopsworks(df: pd.DataFrame) -> None:
    log.info("Uploading → %s:v%d …", FEATUREGROUP_NAME, FG_VERSION)
    proj = hopsworks.login(host=HOPSWORKS_HOST, api_key_value=HOPSWORKS_API_KEY)
    fs   = proj.get_feature_store(name=FEATURESTORE_NAME)

    # ➊ Cast every numeric column to float so Hive schema = DOUBLE
    num_cols = df.select_dtypes(include=["int", "int64", "float"]).columns
    df[num_cols] = df[num_cols].astype(float)

    fg = fs.get_or_create_feature_group(
        name        = FEATUREGROUP_NAME,
        version     = FG_VERSION,
        primary_key = ["timestamp"],
        event_time  = "timestamp",
        description = "Daily AQI + rolling features for Islamabad (all floats)",
    )
    fg.insert(df, write_options={"wait_for_job": True})
    log.info("✅ upload complete")

# ─── Main ────────────────────────────────────────────────────────────────
def main() -> None:
    df = fetch_historical()
    upload_to_hopsworks(df)

if __name__ == "__main__":
    main()
