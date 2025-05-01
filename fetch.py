from dotenv import load_dotenv
load_dotenv()
import os
import logging
import time
from datetime import datetime, timedelta

import requests
import pandas as pd
import hopsworks
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ─── Configuration ───────────────────────────────────────────────────────────────
# Load from env; you can use python-dotenv or set in your CI/CD secrets
OPENWEATHER_API_KEY    = os.environ["OPENWEATHER_API_KEY"]
HOPSWORKS_API_KEY      = os.environ["HOPSWORKS_API_KEY"]
HOPSWORKS_HOST         = os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai")

LATITUDE               = float(os.getenv("LATITUDE", 33.6995))
LONGITUDE              = float(os.getenv("LONGITUDE", 73.0363))
CITY_NAME              = os.getenv("CITY_NAME", "Islamabad")

# How many days back to pull (e.g. 365 for a year)
BACKFILL_DAYS          = int(os.getenv("BACKFILL_DAYS", 90))

BASE_URL               = "http://api.openweathermap.org/data/2.5/air_pollution/history"
FEATURESTORE_NAME      = os.getenv("FEATURESTORE_NAME", "aqi_islamabad_featurestore")
FEATUREGROUP_NAME      = os.getenv("FEATUREGROUP_NAME", "isb_aqi_history")
FG_VERSION             = int(os.getenv("FG_VERSION", 1))

# ─── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("fetch_aqi.log")],
)
logger = logging.getLogger(__name__)

# ─── Retry Policies ─────────────────────────────────────────────────────────────
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

# ─── AQI Breakpoints ────────────────────────────────────────────────────────────
# Each pollutant’s breakpoints from US EPA; defines (Cp_low, Cp_high, I_low, I_high)
BREAKPOINTS = {
    "pm2_5": [(0.0,12.0,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
              (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,500.4,301,500)],
    "pm10":  [(0,54,0,50),(55,154,51,100),(155,254,101,150),
              (255,354,151,200),(355,424,201,300),(425,604,301,500)],
    "no2":   [(0,53,0,50),(54,100,51,100),(101,360,101,150),
              (361,649,151,200),(650,1249,201,300),(1250,2049,301,500)],
    "so2":   [(0,35,0,50),(36,75,51,100),(76,185,101,150),
              (186,304,151,200),(305,604,201,300),(605,1004,301,500)],
    "co":    [(0.0,4.4,0,50),(4.5,9.4,51,100),(9.5,12.4,101,150),
              (12.5,15.4,151,200),(15.5,30.4,201,300),(30.5,50.4,301,500)],
    "o3":    [(0,54,0,50),(55,70,51,100),(71,85,101,150),
              (86,105,151,200),(106,200,201,300)],
    "nh3":   [(0,200,0,50),(201,400,51,100),(401,800,101,150)]
}

def compute_individual_aqi(cp: float, breakpoints: list) -> int:
    for (Cl, Ch, Il, Ih) in breakpoints:
        if Cl <= cp <= Ch:
            return int(((Ih-Il)/(Ch-Cl))*(cp-Cl) + Il)
    return 500

def compute_overall_aqi(row: dict) -> int:
    aqi_vals = []
    for pol in ["pm2_5","pm10","no2","so2","co","o3","nh3"]:
        if pol in row and row[pol] is not None:
            aqi_vals.append(compute_individual_aqi(row[pol], BREAKPOINTS[pol]))
    return max(aqi_vals)

# ─── Data Fetching ─────────────────────────────────────────────────────────────
@retry_http
def fetch_day_data(start_ts: int, end_ts: int) -> list:
    url = (f"{BASE_URL}?lat={LATITUDE}&lon={LONGITUDE}"
           f"&start={start_ts}&end={end_ts}&appid={OPENWEATHER_API_KEY}")
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json().get("list", [])

def fetch_historical() -> pd.DataFrame:
    logger.info("Backfilling last %d days of AQI data for %s", BACKFILL_DAYS, CITY_NAME)
    records = []
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=BACKFILL_DAYS)

    dt = start_dt
    while dt < end_dt:
        start_ts, end_ts = int(dt.timestamp()), int((dt + timedelta(days=1)).timestamp())
        try:
            entries = fetch_day_data(start_ts, end_ts)
            for e in entries:
                rec = {"timestamp": datetime.utcfromtimestamp(e["dt"]), "city": CITY_NAME}
                rec.update(e["components"])
                rec["aqi"] = compute_overall_aqi(rec)
                records.append(rec)
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", dt.date(), e)
        dt += timedelta(days=1)
        time.sleep(0.5)

    if not records:
        raise RuntimeError("No data fetched at all; aborting.")
    df = pd.DataFrame(records).sort_values("timestamp")
    logger.info("Fetched %d records; AQI range [%d, %d]",
                len(df), df.aqi.min(), df.aqi.max())
    return df

# ─── Hopsworks Upload ──────────────────────────────────────────────────────────
@retry_hops
def upload_to_hopsworks(df: pd.DataFrame):
    logger.info("Logging into Hopsworks at %s …", HOPSWORKS_HOST)
    proj = hopsworks.login(host=HOPSWORKS_HOST, api_key_value=HOPSWORKS_API_KEY)
    fs   = proj.get_feature_store(name=FEATURESTORE_NAME)
    fg   = fs.get_or_create_feature_group(
        name=FEATUREGROUP_NAME,
        version=FG_VERSION,
        primary_key=["timestamp"],
        event_time="timestamp",
        description="Backfilled, multivariate AQI history for Islamabad"
    )
    logger.info("Inserting %d rows into FG %s:v%d …", len(df), FEATUREGROUP_NAME, FG_VERSION)
    fg.insert(df, write_options={"wait_for_job": True})
    logger.info("Upload complete.")

# ─── Main ───────────────────────────────────────────────────────────────────────
def main():
    df = fetch_historical()
    upload_to_hopsworks(df)

if __name__ == "__main__":
    main()
