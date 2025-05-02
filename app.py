# app.py
"""
Streamlit dashboard for real‑time and 3‑day AQI forecasts for Islamabad.

Key changes
───────────
1. Loads the *model bundle* (dict of {'model', 'feat_order'}) saved by
   trainmodel.py, so feature‑name drift can’t bite.
2. Forecast loop now **recomputes rolling means each day**, instead of
   carrying yesterday’s values forward – no more flat 500s!
3. Points to feature‑group version 3 (unit‑fixed data).
"""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import os, logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import joblib
import hopsworks
import streamlit as st
import plotly.express as px

# ─── Config ────────────────────────────────────────────────────────────────
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY   = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_HOST      = os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai")

FEATURESTORE_NAME   = os.getenv("FEATURESTORE_NAME", "aqi_islamabad_featurestore")
FEATUREGROUP_NAME   = os.getenv("FEATUREGROUP_NAME", "isb_aqi_history")
FG_VERSION          = int(os.getenv("FG_VERSION", "3"))   # ← new version

MODEL_REGISTRY_NAME = os.getenv("MODEL_REGISTRY_NAME", "isb_aqi_model")

CITY               = "Islamabad"
LAT, LON           = 33.6995, 73.0363
REALTIME_URL       = "http://api.openweathermap.org/data/2.5/air_pollution"
HIST_DAYS          = 90

# ─── Streamlit page setup ─────────────────────────────────────────────────
st.set_page_config("Islamabad AQI Forecast", "🌍", layout="wide")
st.markdown(
    """
    <style>
        body { background:#1e1e1e; color:#fff; }
        .section { transition: all .6s ease-in-out; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("aqi_app.log")],
)
log = logging.getLogger(__name__)

# ─── AQI helpers (same break‑points as train/fetch) ───────────────────────
BREAKPOINTS = {
    "pm2_5": [(0.0,12.0,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
              (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,500.4,301,500)],
    "pm10":  [(0,54,0,50),(55,154,51,100),(155,254,101,150),
              (255,354,151,200),(355,424,201,300),(425,604,301,500)],
}
def compute_individual_aqi(cp, bps):
    if cp is None or np.isnan(cp): return None
    for Cl,Ch,Il,Ih in bps:
        if Cl <= cp <= Ch:
            return int(((Ih-Il)/(Ch-Cl))*(cp-Cl) + Il)
    return 500

def aqi_color(aqi):
    if aqi<=50:   return "#00cc44"
    if aqi<=100:  return "#C8A600"
    if aqi<=150:  return "#CC5500"
    if aqi<=200:  return "#cd5c5c"
    if aqi<=300:  return "#9900cc"
    return "#4E0068"

def aqi_label(aqi):
    if aqi<=50:   return "Good"
    if aqi<=100:  return "Moderate"
    if aqi<=150:  return "Unhealthy for Sensitive Groups"
    if aqi<=200:  return "Unhealthy"
    if aqi<=300:  return "Very Unhealthy"
    return "Hazardous"

def aqi_message(aqi):
    if aqi<=50:   return "Enjoy outdoor activities"
    if aqi<=100:  return "Limit prolonged exertion if sensitive"
    if aqi<=150:  return "Sensitive groups may experience effects"
    if aqi<=200:  return "Everyone may begin to experience effects"
    if aqi<=300:  return "Significant health effects likely"
    return "Serious health effects very likely"

def aqi_face(aqi): return "😀" if aqi<=100 else ("😐" if aqi<=150 else "😷")

# ─── Connect to Hopsworks ────────────────────────────────────────────────
@st.cache_resource
def _connect():
    proj = hopsworks.login(host=HOPSWORKS_HOST, api_key_value=HOPSWORKS_API_KEY)
    return (
        proj.get_feature_store(name=FEATURESTORE_NAME),
        proj.get_model_registry()
    )
fs, mr = _connect()

# ─── Real‑time AQI (OpenWeather) ──────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_realtime():
    r = requests.get(
        f"{REALTIME_URL}?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}",
        timeout=10,
    )
    r.raise_for_status()
    d  = r.json()["list"][0]
    pm = d["components"]["pm2_5"]
    return {
        "aqi": compute_individual_aqi(pm, BREAKPOINTS["pm2_5"]),
        "timestamp": datetime.fromtimestamp(d["dt"], tz=timezone.utc),
        **d["components"],
    }

# ─── Load latest model bundle ─────────────────────────────────────────────
@st.cache_resource
def load_model_bundle():
    latest = sorted(
        mr.get_models(MODEL_REGISTRY_NAME), key=lambda m: m.version, reverse=True
    )[0]
    path   = latest.download()
    bundle = joblib.load(os.path.join(path, "model.pkl"))
    return bundle            # {'model': XGBRegressor, 'feat_order': [...]}

# ─── Historical feature‑group ↔ DataFrame ────────────────────────────────
@st.cache_data(ttl=300)
def fetch_history():
    fg = fs.get_feature_group(name=FEATUREGROUP_NAME, version=FG_VERSION)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)

def add_rolls(df: pd.DataFrame) -> pd.DataFrame:
    for w in (3,7,14):
        df[f"aqi_roll{w}"] = df["aqi"].rolling(w).mean()
    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    base = [f"aqi_roll{w}" for w in (3,7,14)] + ["aqi"]
    df = df.copy()
    for c in base:
        df[f"{c}_lag1"] = df[c].shift(1)
    return df.dropna().reset_index(drop=True)

# ─── Forecast function (re‑computes rolls each day) ───────────────────────
def forecast_days(bundle, raw: pd.DataFrame, n=3):
    model      = bundle["model"]
    feat_order = bundle["feat_order"]

    hist = raw.copy()
    preds = []
    for _ in range(n):
        feat_df  = prepare_features(hist)
        X_last   = feat_df[feat_order].iloc[[-1]]
        pred     = int(round(model.predict(X_last)[0]))
        preds.append(pred)

        # append new day and recompute rolling means
        next_day = hist.iloc[-1].copy()
        next_day["timestamp"] += timedelta(days=1)
        next_day["aqi"]        = pred
        hist = pd.concat([hist, pd.DataFrame([next_day])], ignore_index=True)
        hist = add_rolls(hist)
    return preds

# ─── Page actions ─────────────────────────────────────────────────────────
if st.button("🔄 Refresh"):
    st.cache_data.clear(); st.cache_resource.clear()
    st.experimental_rerun()

try:
    current = fetch_realtime()
    bundle  = load_model_bundle()
    history = fetch_history()
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# ─── Current AQI panel ────────────────────────────────────────────────────
c_aqi   = current["aqi"]
st.markdown(
    f"""
    <div class="section" style="background:{aqi_color(c_aqi)};padding:1rem;border-radius:8px;">
      <h2>Current AQI — {c_aqi} ({aqi_label(c_aqi)}) {aqi_face(c_aqi)}</h2>
      <p><strong>{aqi_message(c_aqi)}</strong>
         • PM2.5: {current["pm2_5"]:.1f} µg/m³
         • PM10: {current["pm10"]:.1f} µg/m³
      </p>
      <p>As of {current["timestamp"].strftime('%Y-%m-%d %H:%M UTC')}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Pollutant bars
polls = ["pm2_5","pm10","no2","so2","co","o3"]
vals  = [current[p] for p in polls]
poll_df = pd.DataFrame({"Pollutant":polls,"Value":vals})
fig = px.bar(poll_df, y="Pollutant", x="Value", orientation="h",
             template="plotly_dark", labels={"Value":"µg/m³"})
fig.update_layout(height=300, margin=dict(l=20,r=20,t=20,b=20))
st.subheader("Today's Pollutant Breakdown")
st.plotly_chart(fig, use_container_width=True)

# ─── 3‑day forecast cards ────────────────────────────────────────────────
st.subheader("Next 3‑Day AQI Forecast")
preds = forecast_days(bundle, history, 3)
for i, p in enumerate(preds, 1):
    st.metric(f"Day {i}", p, aqi_label(p), delta_color="off")

# ─── 90‑day history plot ─────────────────────────────────────────────────
st.subheader(f"Historical AQI (last {HIST_DAYS} days)")
cutoff = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(days=HIST_DAYS)
hist = history[history["timestamp"] >= cutoff]
if hist.empty:
    st.warning("No historical data available.")
else:
    hist["date"] = hist["timestamp"].dt.date
    plot_df = hist.groupby("date")["aqi"].mean().reset_index()
    plot_df["cat"] = plot_df["aqi"].apply(aqi_label)
    cmap = {
        "Good":"#00cc44","Moderate":"#C8A600",
        "Unhealthy for Sensitive Groups":"#CC5500","Unhealthy":"#cd5c5c",
        "Very Unhealthy":"#9900cc","Hazardous":"#4E0068",
    }
    fig2 = px.bar(plot_df, x="date", y="aqi", color="cat",
                  color_discrete_map=cmap,
                  template="plotly_dark",
                  labels={"aqi":"AQI","date":"Date","cat":"Category"})
    fig2.update_layout(height=300, margin=dict(l=20,r=20,t=30,b=20))
    st.plotly_chart(fig2, use_container_width=True)
