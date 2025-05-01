# app.py

from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
import pandas as pd
import numpy as np
import hopsworks
import joblib
import requests
import logging
from datetime import datetime, timedelta, timezone
import plotly.express as px

# ─── Configuration ───────────────────────────────────────────────────────────────
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
HOPSWORKS_API_KEY      = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_HOST         = os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai")

CITY = os.getenv("CITY_NAME", "Islamabad")
LAT, LON = float(os.getenv("LATITUDE", 33.6995)), float(os.getenv("LONGITUDE", 73.0363))
BASE_URL = "http://api.openweathermap.org/data/2.5/air_pollution"

FEATURESTORE_NAME = os.getenv("FEATURESTORE_NAME", "aqi_islamabad_featurestore")
FEATUREGROUP_NAME = os.getenv("FEATUREGROUP_NAME", "isb_aqi_history")
FG_VERSION        = int(os.getenv("FG_VERSION", "1"))

MODEL_REGISTRY_NAME = os.getenv("MODEL_REGISTRY_NAME", "isb_aqi_model")
MODEL_FILE          = os.getenv("MODEL_FILE", "model.pkl")

# ─── Logging ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("aqi_app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ─── Streamlit Page ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AQI App", page_icon="🌍", layout="wide")

st.markdown(
    """
    <style>
      body { background: linear-gradient(45deg,#1e1e1e,#333,#1e1e1e); color:#fff; }
      .section { transition: all 1s ease-in-out; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌍 Islamabad Air Quality Forecasting")

if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.experimental_rerun()

# ─── AQI Helpers ─────────────────────────────────────────────────────────────────
BREAKPOINTS = {
    "pm2_5": [(0.0,12.0,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
              (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,500.4,301,500)],
    "pm10":  [(0,54,0,50),(55,154,51,100),(155,254,101,150),
              (255,354,151,200),(355,424,201,300),(425,604,301,500)]
}

def compute_individual_aqi(cp, breakpoints):
    if cp is None: return None
    for Cl, Ch, Il, Ih in breakpoints:
        if Cl <= cp <= Ch:
            return int(((Ih - Il)/(Ch - Cl))*(cp - Cl) + Il)
    return 500

def get_aqi_color(aqi):
    if aqi<=50: return "#00cc44"
    if aqi<=100: return "#C8A600"
    if aqi<=150: return "#CC5500"
    if aqi<=200: return "#cd5c5c"
    if aqi<=300: return "#9900cc"
    return "#4E0068"

def get_aqi_label(aqi):
    if aqi<=50: return "Good"
    if aqi<=100: return "Moderate"
    if aqi<=150: return "Unhealthy for Sensitive Groups"
    if aqi<=200: return "Unhealthy"
    if aqi<=300: return "Very Unhealthy"
    return "Hazardous"

def get_aqi_message(aqi):
    if aqi<=50: return "Enjoy outdoor activities"
    if aqi<=100: return "Sensitive individuals should limit exertion"
    if aqi<=150: return "Health effects for sensitive groups"
    if aqi<=200: return "Everyone may experience effects"
    if aqi<=300: return "Significant health risks"
    return "Serious health risks"

def get_aqi_face(aqi):
    if aqi<=50: return "😀"
    if aqi<=100: return "😐"
    if aqi<=150: return "🤧"
    return "😷"

# ─── Connect to Hopsworks ─────────────────────────────────────────────────────────
@st.cache_resource
def connect_to_hopsworks():
    proj = hopsworks.login(host=HOPSWORKS_HOST, api_key_value=HOPSWORKS_API_KEY)
    fs = proj.get_feature_store(name=FEATURESTORE_NAME)
    mr = proj.get_model_registry()
    return fs, mr

fs, mr = connect_to_hopsworks()

# ─── Fetch real-time AQI ─────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_realtime_aqi():
    try:
        url = f"{BASE_URL}?lat={LAT}&lon={LON}&appid={OPENWEATHERMAP_API_KEY}"
        resp = requests.get(url); resp.raise_for_status()
        entry = resp.json()["list"][0]
        pm25 = entry["components"]["pm2_5"]
        aqi  = compute_individual_aqi(pm25, BREAKPOINTS["pm2_5"])
        return {
            "aqi": aqi,
            "pm2_5": pm25,
            "pm10": entry["components"]["pm10"],
            "no2": entry["components"]["no2"],
            "so2": entry["components"]["so2"],
            "co": entry["components"]["co"],
            "o3": entry["components"]["o3"],
            "timestamp": datetime.fromtimestamp(entry["dt"], tz=timezone.utc)
        }
    except Exception as e:
        logger.error("Failed real-time fetch: %s", e)
        return None

# ─── Load latest model ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    models = mr.get_models(MODEL_REGISTRY_NAME)
    latest = sorted(models, key=lambda m: m.version, reverse=True)[0]
    dirpath = latest.download()
    return joblib.load(os.path.join(dirpath, MODEL_FILE))

model = load_model()

# ─── Fetch historical FG data ────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_history():
    fg = fs.get_feature_group(FEATUREGROUP_NAME, version=FG_VERSION)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)

history = fetch_history()

# ─── Forecast next 3 days ───────────────────────────────────────────────────────
def forecast_next_days(model, df, days=3, max_lag=3):
    df = df.sort_values("timestamp").tail(max_lag).copy()
    # engineer sub-indices
    if "pm2_5_idx" not in df: 
        df["pm2_5_idx"] = df["pm2_5"].map(lambda v: compute_individual_aqi(v, BREAKPOINTS["pm2_5"]))
    if "pm10_idx" not in df:
        df["pm10_idx"] = df["pm10"].map(lambda v: compute_individual_aqi(v, BREAKPOINTS["pm10"]))
    preds = []
    for _ in range(days):
        # create lags
        for col in ["pm2_5_idx","pm10_idx","aqi"]:
            for lag in range(1, max_lag+1):
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
        row = df.dropna().iloc[[-1]]
        feat_cols = [f"{col}_lag{lag}" 
                     for col in ("pm2_5_idx","pm10_idx","aqi") 
                     for lag in range(1, max_lag+1)]
        pred = int(round(model.predict(row[feat_cols])[0]))
        preds.append(pred)

        # append dummy next‐day
        new = {c: df[c].iloc[-1] for c in ["pm2_5","pm10","no2","so2","co","o3"]}
        new["aqi"] = pred
        new["timestamp"] = df["timestamp"].iloc[-1] + timedelta(days=1)
        df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
        # recompute sub-indices
        df["pm2_5_idx"] = df["pm2_5"].map(lambda v: compute_individual_aqi(v, BREAKPOINTS["pm2_5"]))
        df["pm10_idx"] = df["pm10"].map(lambda v: compute_individual_aqi(v, BREAKPOINTS["pm10"]))
    return preds

# ─── Display ─────────────────────────────────────────────────────────────────────
realtime = fetch_realtime_aqi()
if realtime:
    color = get_aqi_color(realtime["aqi"])
    label = get_aqi_label(realtime["aqi"])
    msg   = get_aqi_message(realtime["aqi"])
    face  = get_aqi_face(realtime["aqi"])

    st.markdown(f"""
    <div style="background:{color};padding:1rem;border-radius:8px;color:#fff">
      <h2>Current AQI: {realtime['aqi']} ({label}) {face}</h2>
      <p><strong>{msg}</strong> • PM2.5: {realtime['pm2_5']} µg/m³ | PM10: {realtime['pm10']} µg/m³</p>
    </div>
    """, unsafe_allow_html=True)

    # pollutant breakdown
    df_p = pd.DataFrame({
      "Pollutant":["PM2.5","PM10","NO2","SO2","CO","O3"],
      "Value":[realtime["pm2_5"], realtime["pm10"],
               realtime["no2"], realtime["so2"],
               realtime["co"], realtime["o3"]]
    })
    fig = px.bar(df_p, y="Pollutant", x="Value", orientation="h",
                 template="plotly_dark", color="Pollutant")
    fig.update_layout(paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e", font_color="#fff")
    st.subheader("Today's Pollutant Breakdown")
    st.plotly_chart(fig, use_container_width=True)

# 3-day forecast
if realtime and model and history is not None:
    preds = forecast_next_days(model, history, days=3)
    st.subheader("⏭️ Next 3 Days AQI Forecast")
    cols = st.columns(3)
    for i, p in enumerate(preds):
        c = cols[i]
        col = get_aqi_color(p)
        lbl = get_aqi_label(p)
        face = get_aqi_face(p)
        c.markdown(f"""
          <div style="background:{col};padding:1rem;border-radius:8px;color:#fff">
            <h3>Day {i+1}: {p} {face}</h3>
            <p><strong>{lbl}</strong></p>
          </div>
        """, unsafe_allow_html=True)

# historical chart
if history is not None:
    df90 = history.copy()
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=90)
    df90 = df90[df90["timestamp"] >= cutoff]
    df90["date"] = df90["timestamp"].dt.date
    df90 = df90.groupby("date").mean(numeric_only=True).reset_index()
    df90["label"] = df90["aqi"].map(get_aqi_label)
    st.subheader("📊 Last 90 Days AQI")
    fig2 = px.bar(df90, x="date", y="aqi", color="label",
                  template="plotly_dark",
                  color_discrete_map={
                     "Good":"#00cc44","Moderate":"#C8A600",
                     "Unhealthy for Sensitive Groups":"#CC5500",
                     "Unhealthy":"#cd5c5c","Very Unhealthy":"#9900cc",
                     "Hazardous":"#4E0068"
                  })
    fig2.update_layout(paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e",
                       font_color="#fff", hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)
