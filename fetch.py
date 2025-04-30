import requests
from datetime import datetime, timedelta
import pandas as pd
import hopsworks
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("fetch_aqi.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

OPENWEATHERMAP_API_KEY = "bd4f84998dd0e6d4c12ae08ff7d8d0df"
LAT = 33.6995  # Lahore latitude
LON = 73.0363  # Lahore longitude
CITY = "Islamabad"
BASE_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"

def pm25_to_aqi(pm25):
    """Convert PM2.5 (µg/m³) to US AQI (0–500)."""
    try:
        if 0 <= pm25 <= 12.0:
            return (50 / 12.0) * pm25
        elif 12.1 <= pm25 <= 35.4:
            return ((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1) + 51
        elif 35.5 <= pm25 <= 55.4:
            return ((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101
        elif 55.5 <= pm25 <= 150.4:
            return ((200 - 151) / (150.4 - 55.5)) * (pm25 - 55.5) + 151
        elif 150.5 <= pm25 <= 250.4:
            return ((300 - 201) / (250.4 - 150.5)) * (pm25 - 150.5) + 201
        elif 250.5 <= pm25 <= 500.4:
            return ((500 - 301) / (500.4 - 250.5)) * (pm25 - 250.5) + 301
        else:
            return 500
    except Exception as e:
        logger.error(f"Error converting PM2.5 to AQI: {e}")
        return 0

def fetch_historical_aqi():
    logger.info("Fetching 60 days of historical AQI data for %s...", CITY)
    aqi_data = []
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=90)

    current_date = start_date
    while current_date < end_date:
        start_ts = int(current_date.timestamp())
        end_ts = int((current_date + timedelta(days=1)).timestamp())
        url = f"{BASE_URL}?lat={LAT}&lon={LON}&start={start_ts}&end={end_ts}&appid={OPENWEATHERMAP_API_KEY}"
        
        try:
            response = requests.get(url)
            if response.status_code == 429:
                logger.warning("Rate limit exceeded. Waiting 60 seconds...")
                time.sleep(60)
                response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if "list" in data:
                for entry in data["list"]:
                    pm25 = entry["components"]["pm2_5"]
                    us_aqi = int(pm25_to_aqi(pm25))
                    aqi_record = {
                        "city": CITY,
                        "aqi": us_aqi,
                        "co": entry["components"]["co"],
                        "no": entry["components"]["no"],
                        "no2": entry["components"]["no2"],
                        "o3": entry["components"]["o3"],
                        "so2": entry["components"]["so2"],
                        "pm2_5": pm25,
                        "pm10": entry["components"]["pm10"],
                        "nh3": entry["components"]["nh3"],
                        "timestamp": datetime.utcfromtimestamp(entry["dt"]).strftime("%Y-%m-%d %H:%M:%S")
                    }
                    aqi_data.append(aqi_record)
            else:
                logger.warning(f"No data returned for {current_date.strftime('%Y-%m-%d')}")
        except requests.RequestException as e:
            logger.error(f"Error fetching data for {current_date.strftime('%Y-%m-%d')}: {e}")
        
        current_date += timedelta(days=1)
        time.sleep(1)  # Avoid overwhelming API
    
    if not aqi_data:
        logger.error("No AQI data collected.")
        raise ValueError("No AQI data collected.")
    
    df = pd.DataFrame(aqi_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    logger.info(f"Fetched {len(df)} records. AQI range: {df['aqi'].min()} to {df['aqi'].max()}")
    logger.info(f"Sample data:\n{df[['timestamp', 'aqi', 'pm2_5']].head(5).to_string()}")
    return df

def save_to_hopsworks(df):
    logger.info("Connecting to Hopsworks...")
    try:
        project = hopsworks.login(
            host="c.app.hopsworks.ai",
            api_key_value="gLSSl3rhxHuDtcgF.0qpN3LM2nlJREY39NBkp0pUGNg4a4mwh1mfgEZSdvvZ5Vx2i5LdwHOSzVN18Dse9"
        )
        fs = project.get_feature_store(name="aqi_islamabad_featurestore")
        
        fg = fs.get_or_create_feature_group(
            name="isb_aqi_history",
            version=1,
            primary_key=["timestamp"],
            description="Historical AQI data for Islamabad",
            event_time="timestamp"
        )
        
        logger.info("Inserting data into feature group...")
        fg.insert(df, write_options={"wait_for_job": True})
        logger.info("Data successfully inserted into Hopsworks.")
    except Exception as e:
        logger.error(f"Error saving to Hopsworks: {e}")
        raise

def main():
    try:
        df = fetch_historical_aqi()
        save_to_hopsworks(df)
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise

if __name__ == "__main__":
    main()