import streamlit as st
import pandas as pd
import numpy as np
import hopsworks
import xgboost as xgb
import joblib
from datetime import datetime, timedelta
import requests
import logging
import plotly.express as px

st.set_page_config(page_title="AQI App", page_icon="🌍", layout="wide")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("aqi_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Custom CSS
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(45deg, #1e1e1e, #333333, #1e1e1e);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #ffffff;
    }
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .section {
        transition: all 1s ease-in-out;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Hardcoded for Lahore
CITY = "Islamabad"
LAT = 33.6995  
LON = 73.0363 

# Title and description
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown(
        """
        <h1 style="text-align: center; margin-bottom: 0;">
            Islamabad Air Quality Forecasting App
        </h1>
        <h4 style="text-align: center; margin-top: 0;">
            Real-time AQI data and 3-day forecast for Lahore using OpenWeatherMap and XGBoost.
        </h4>
        """,
        unsafe_allow_html=True
    )
with col3:
    st.write("")
    st.write("")
    if st.button("Refresh AQI Data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# Constants
OPENWEATHERMAP_API_KEY = "bd4f84998dd0e6d4c12ae08ff7d8d0df"
HOPSWORKS_API_KEY = "gLSSl3rhxHuDtcgF.0qpN3LM2nlJREY39NBkp0pUGNg4a4mwh1mfgEZSdvvZ5Vx2i5LdwHOSzVN18Dse9"
BASE_URL = "http://api.openweathermap.org/data/2.5/air_pollution"

# Hopsworks connection
@st.cache_resource
def connect_to_hopsworks():
    try:
        project = hopsworks.login(
            host="c.app.hopsworks.ai",
            api_key_value=HOPSWORKS_API_KEY
        )
        fs = project.get_feature_store(name="aqi_islamabad_featurestore")
        mr = project.get_model_registry()
        logger.info("Connected to Hopsworks successfully.")
        return fs, mr
    except Exception as e:
        logger.error(f"Failed to connect to Hopsworks: {e}")
        st.error("Failed to connect to Hopsworks. Please check your API key and connection.")
        return None, None

fs, mr = connect_to_hopsworks()

# Fetch latest AQI from feature group as fallback
def fetch_latest_historical_aqi():
    try:
        fg = fs.get_feature_group("isb_aqi_history", version=1)
        df = fg.read()
        if df.empty:
            logger.warning("Feature group 'isb_aqi_history' is empty for fallback AQI.")
            return None, None
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        latest = df.sort_values("timestamp").tail(1)
        aqi = latest["aqi"].iloc[0]
        pm25 = latest["pm2_5"].iloc[0]
        logger.info(f"Latest historical AQI: {aqi}, PM2.5: {pm25} µg/m³")
        return aqi, pm25
    except Exception as e:
        logger.error(f"Error fetching latest historical AQI: {e}")
        return None, None

# Fetch real-time AQI
@st.cache_data(ttl=300)
def fetch_realtime_aqi():
    try:
        url = f"{BASE_URL}?lat={LAT}&lon={LON}&appid={OPENWEATHERMAP_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if "list" in data:
            entry = data["list"][0]
            pm25 = entry["components"]["pm2_5"]
            def pm25_to_aqi(pm25):
                if 0 <= pm25 <= 12.0:
                    return int((50 / 12.0) * pm25)
                elif 12.1 <= pm25 <= 35.4:
                    return int(((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1) + 51)
                elif 35.5 <= pm25 <= 55.4:
                    return int(((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101)
                elif 55.5 <= pm25 <= 150.4:
                    return int(((200 - 151) / (150.4 - 55.5)) * (pm25 - 55.5) + 151)
                elif 150.5 <= pm25 <= 250.4:
                    return int(((300 - 201) / (250.4 - 150.5)) * (pm25 - 150.5) + 201)
                elif 250.5 <= pm25 <= 500.4:
                    return int(((500 - 301) / (500.4 - 250.5)) * (pm25 - 250.5) + 301)
                else:
                    return 500
            us_aqi = pm25_to_aqi(pm25)
            logger.info(f"OpenWeatherMap PM2.5: {pm25} µg/m³, AQI: {us_aqi} for {CITY}")
            
            # Validate PM2.5; fallback to historical AQI if too low
            if pm25 < 40:  # Unlikely for Lahore
                logger.warning(f"PM2.5 {pm25} µg/m³ too low for {CITY}. Fetching latest historical AQI...")
                fallback_aqi, fallback_pm25 = fetch_latest_historical_aqi()
                if fallback_aqi is not None:
                    us_aqi = int(fallback_aqi)
                    pm25 = fallback_pm25
                    logger.info(f"Using historical AQI: {us_aqi}, PM2.5: {pm25} µg/m³")
            
            aqi_record = {
                "city": CITY,
                "aqi": us_aqi,
                "pm2_5": pm25,
                "pm10": entry["components"]["pm10"],
                "no2": entry["components"]["no2"],
                "so2": entry["components"]["so2"],
                "co": entry["components"]["co"],
                "o3": entry["components"]["o3"],
                "timestamp": datetime.utcfromtimestamp(entry["dt"]).strftime("%Y-%m-%d %H:%M:%S"),
                "source": "OpenWeatherMap" if pm25 >= 40 else "Historical Fallback"
            }
            return pd.Series(aqi_record)
        else:
            logger.error("No AQI data found in OpenWeatherMap response.")
            st.error("No AQI data found in OpenWeatherMap response.")
            return None
    except Exception as e:
        logger.error(f"Error fetching real-time AQI for {CITY}: {e}")
        st.error(f"Failed to fetch real-time AQI for {CITY}. Please check the API key or connection.")
        return None

# Load trained model
@st.cache_resource
def load_model():
    try:
        models = mr.get_models("isb_aqi_model")
        if not models:
            raise ValueError("No models found with the name 'isb_aqi_model'.")
        models_sorted = sorted(models, key=lambda m: m.version, reverse=True)
        latest_model = models_sorted[0]
        model_dir = latest_model.download()
        model_path = f"{model_dir}/isb_aqi_model.pkl"
        model = joblib.load(model_path)
        logger.info(f"Loaded model 'isb_aqi_model' version {latest_model.version} successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error("Failed to load the trained model. Please check the Model Registry.")
        return None

# AQI helper functions
def get_aqi_color(aqi_value):
    if aqi_value <= 50:
        return "#00cc44"  # Good
    elif aqi_value <= 100:
        return "#C8A600"  # Moderate
    elif aqi_value <= 150:
        return "#CC5500"  # Unhealthy for Sensitive Groups
    elif aqi_value <= 200:
        return "#cd5c5c"  # Unhealthy
    elif aqi_value <= 300:
        return "#9900cc"  # Very Unhealthy
    else:
        return "#4E0068"  # Hazardous

def get_aqi_label(aqi_value):
    if aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Moderate"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi_value <= 200:
        return "Unhealthy"
    elif aqi_value <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def get_aqi_message(aqi_value):
    if aqi_value <= 50:
        return "Enjoy outdoor activities"
    elif aqi_value <= 100:
        return "Sensitive individuals should limit prolonged exertion"
    elif aqi_value <= 150:
        return "Sensitive groups may experience health effects"
    elif aqi_value <= 200:
        return "Everyone may begin to experience health effects"
    elif aqi_value <= 300:
        return "Significant health effects for everyone"
    else:
        return "Serious health effects for everyone"

def get_aqi_face(aqi_value):
    if aqi_value <= 50:
        return "😀"
    elif aqi_value <= 100:
        return "😐"
    elif aqi_value <= 150:
        return "🤧"
    else:
        return "😷"

# Forecast next 3 days
def forecast_next_days(model, last_records, days=3, max_lag=3):
    try:
        if len(last_records) < max_lag:
            logger.warning(f"Insufficient records for forecasting: {len(last_records)} rows, need at least {max_lag}.")
            return None
        aqi_predictions = []
        pm25_values = []
        pm10_values = []
        df = last_records.sort_values("timestamp").tail(max_lag).copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        for _ in range(days):
            for lag in range(1, max_lag + 1):
                df[f"aqi_lag{lag}"] = df["aqi"].shift(lag)
            latest = df.iloc[-1:].copy()
            latest = latest.dropna()
            if latest.empty:
                raise ValueError("Not enough data to create lag features.")
            
            features = [f"aqi_lag{lag}" for lag in range(1, max_lag + 1)]
            X = latest[features].values
            pred_aqi = int(round(model.predict(X)[0]))
            aqi_predictions.append(pred_aqi)
            pm25_values.append(latest["pm2_5"].iloc[0])
            pm10_values.append(latest["pm10"].iloc[0])
            
            new_record = {
                "city": CITY,
                "aqi": pred_aqi,
                "pm2_5": latest["pm2_5"].iloc[0],
                "pm10": latest["pm10"].iloc[0],
                "no2": latest["no2"].iloc[0],
                "so2": latest["so2"].iloc[0],
                "co": latest["co"].iloc[0],
                "o3": latest["o3"].iloc[0],
                "timestamp": latest["timestamp"].iloc[0] + timedelta(days=1)
            }
            df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
        
        logger.info(f"Forecasted AQI for next {days} days: {aqi_predictions}")
        return aqi_predictions, pm25_values, pm10_values
    except Exception as e:
        logger.error(f"Error forecasting AQI: {e}")
        st.error("Failed to forecast AQI. Please check the model and input data.")
        return None

# Fetch last records for forecasting
@st.cache_data(ttl=300)
def fetch_last_records():
    try:
        fg = fs.get_feature_group("isb_aqi_history", version=1)
        df = fg.read()
        if df.empty:
            logger.warning("Feature group 'isb_aqi_history' is empty.")
            st.warning(f"No data in feature group for forecasting in {CITY}.")
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        logger.info(f"Fetched {len(df)} records for forecasting.")
        logger.info(f"Sample records:\n{df[['timestamp', 'aqi']].tail(5).to_string()}")
        return df
    except Exception as e:
        logger.error(f"Error fetching last records: {e}")
        st.error(f"Failed to fetch last records for {CITY}. Please check the Feature Store data.")
        return None

# Main logic
current_aqi = None
next_three_days_pred = None

if fs and mr:
    current_aqi_record = fetch_realtime_aqi()
    if current_aqi_record is not None:
        current_aqi = current_aqi_record["aqi"]
    
    model = load_model()
    if model is not None:
        last_records = fetch_last_records()
        if last_records is not None and len(last_records) >= 3:
            forecast_results = forecast_next_days(model, last_records, days=3, max_lag=3)
            if forecast_results is not None:
                aqi_preds, pm25_vals, pm10_vals = forecast_results
                next_three_days_pred = {
                    "aqi": aqi_preds,
                    "pm25": pm25_vals,
                    "pm10": pm10_vals
                }

# Display current AQI
st.write("")
st.write("")
if current_aqi is not None and current_aqi_record is not None:
    current_pm25 = current_aqi_record["pm2_5"]
    current_pm10 = current_aqi_record["pm10"]
    aqi_label = get_aqi_label(current_aqi)
    current_aqi_bg_color = get_aqi_color(current_aqi)
    aqi_message = get_aqi_message(current_aqi)
    aqi_face = get_aqi_face(current_aqi)
    source = current_aqi_record["source"]

    st.markdown(
        f"""
        <div style="
            background-color: {current_aqi_bg_color};
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            color: #fff;
            display: flex;
            justify-content: space-between;
            align-items: center;
        " class="section">
            <div style="flex: 1;">
                <h2 style="margin: 0;">Today's AQI: {current_aqi} - {aqi_label} ({source})</h2>
                <p style="margin: 0; font-size: 1.5rem;"><strong>{aqi_message}</strong></p>
            </div>
            <div style="text-align: right;">
                <p style="font-size: 4rem; margin: 0;">{aqi_face}</p>
                <p style="margin: 0; font-size: 1.2rem;"><strong>Main Pollutant: PM2.5</strong></p>
                <p style="margin: 0; font-size: 1.2rem;">
                    <strong>PM2.5: {current_pm25} µg/m³ | PM10: {current_pm10} µg/m³</strong>
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning(f"Unable to fetch current AQI for {CITY}. Please check the API connection.")

# Pollutant breakdown chart
st.write("")
st.write("")
if current_aqi_record is not None:
    st.subheader("Today's Pollutant Breakdown")
    pollutant_data = {
        "Pollutant": ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"],
        "Value": [
            current_aqi_record.get("pm2_5", 0),
            current_aqi_record.get("pm10", 0),
            current_aqi_record.get("no2", 0),
            current_aqi_record.get("so2", 0),
            current_aqi_record.get("co", 0),
            current_aqi_record.get("o3", 0)
        ]
    }
    df_pollutant = pd.DataFrame(pollutant_data)
    fig_breakdown = px.bar(
        df_pollutant,
        y="Pollutant",
        x="Value",
        color="Pollutant",
        orientation="h",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_breakdown.update_layout(
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#1e1e1e",
        font_color="#fff",
        hovermode="y unified"
    )
    fig_breakdown.update_traces(
        hovertemplate="<b>%{y}</b>: %{x:.2f} µg/m³<extra></extra>"
    )
    st.plotly_chart(fig_breakdown, use_container_width=True)

# Next 3 days forecast
st.write("")
st.write("")
st.write("")
st.subheader(f"Next 3 Days AQI Forecast for {CITY}")
st.write("")
if next_three_days_pred is not None:
    aqi_preds = next_three_days_pred["aqi"]
    pm25_preds = next_three_days_pred["pm25"]
    pm10_preds = next_three_days_pred["pm10"]
    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]
    for i in range(len(aqi_preds)):
        day_aqi = aqi_preds[i]
        day_pm25 = pm25_preds[i]
        day_pm10 = pm10_preds[i]
        day_label = get_aqi_label(day_aqi)
        day_color = get_aqi_color(day_aqi)
        day_face = get_aqi_face(day_aqi)
        with columns[i]:
            st.markdown(
                f"""
                <div style="
                    background-color: {day_color};
                    border-radius: 10px;
                    padding: 1rem;
                    color: #fff;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                " class="section">
                    <div style="flex: 1;">
                        <h3 style="margin-top: 0;">Day {i+1}</h3>
                        <p style="font-size: 1.5rem; margin: 0;"><strong>AQI: {day_aqi}</strong></p>
                        <p style="margin: 0; font-size: 1.2rem;"><strong>{day_label}</strong></p>
                        <p style="margin: 0; font-size: 1.1rem;">
                           <strong>PM2.5: {day_pm25:.2f} µg/m³ | PM10: {day_pm10:.2f} µg/m³</strong>
                        </p>
                    </div>
                    <div style="margin-left: 1rem; font-size: 3rem;">
                        {day_face}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
else:
    st.warning(f"Unable to predict AQI for the next 3 days for {CITY}. Please check the model and data.")

# Historical data (last 60 days)
st.write("")
st.write("")
st.write("")
st.subheader(f"{CITY} Historical Air Quality Data (Last 90 Days)")
st.write("")
@st.cache_data
def fetch_historical_data():
    try:
        fg = fs.get_feature_group("isb_aqi_history", version=1)
        df = fg.read()
        if df.empty:
            logger.warning("Feature group 'isb_aqi_history' is empty.")
            st.warning(f"No historical data found for the last 90 days for {CITY}.")
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cutoff = pd.to_datetime("today") - pd.Timedelta(days=90)
        df = df[df["timestamp"] >= cutoff]
        
        if df.empty:
            logger.warning("No data within the last 90 days.")
            st.warning(f"No historical data found for the last 90 days for {CITY}.")
            return None
        
        df["date"] = df["timestamp"].dt.date
        df = df.groupby("date").mean(numeric_only=True).reset_index()
        df["aqi_label"] = df["aqi"].apply(get_aqi_label)
        df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%a, %b %d")
        logger.info(f"Fetched {len(df)} days of historical data.")
        logger.info(f"Sample historical data:\n{df[['date', 'aqi']].head(5).to_string()}")
        return df
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        st.error(f"Failed to fetch historical data for {CITY}. Please check the Feature Store data.")
        return None

historical_data = fetch_historical_data()
if historical_data is not None and not historical_data.empty:
    aqi_color_map = {
        "Good": "#00cc44",
        "Moderate": "#C8A600",
        "Unhealthy for Sensitive Groups": "#CC5500",
        "Unhealthy": "#cd5c5c",
        "Very Unhealthy": "#9900cc",
        "Hazardous": "#4E0068"
    }
    fig_hist = px.bar(
        historical_data,
        x="date",
        y="aqi",
        color="aqi_label",
        color_discrete_map=aqi_color_map,
        template="plotly_dark",
        labels={"aqi": "AQI", "date": "Date", "aqi_label": "Category"},
        custom_data=["aqi", "aqi_label", "date_str"]
    )
    fig_hist.update_layout(
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font_color='#fff',
        hovermode="x unified",
        legend=dict(itemclick='toggleothers', itemdoubleclick='toggle')
    )
    fig_hist.update_traces(
        hovertemplate="<b>%{customdata[0]} AQI</b> %{customdata[1]}<br>%{customdata[2]}<extra></extra>"
    )
    st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.warning(f"No historical data found for the last 90 days for {CITY}.")
