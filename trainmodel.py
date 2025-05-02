#!/usr/bin/env python3
"""
Train an XGBoost regressor on the daily‑AQI feature group,
then register the model plus its feature order in the Hopsworks
Model Registry.

All text fields are ASCII‑only to avoid MySQL “incorrect string value”
errors in the registry.
"""
from __future__ import annotations

import os, logging, math
from datetime import datetime

import joblib
import hopsworks
import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ─── Env / constants ─────────────────────────────────────────────────────
load_dotenv()
HOPSWORKS_API_KEY   = os.environ["HOPSWORKS_API_KEY"]
HOPSWORKS_HOST      = os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai")
FEATURESTORE_NAME   = os.getenv("FEATURESTORE_NAME", "aqi_islamabad_featurestore")
FEATUREGROUP_NAME   = os.getenv("FEATUREGROUP_NAME", "isb_aqi_history")
FG_VERSION          = int(os.getenv("FG_VERSION", "10"))   # same as fetch.py
MODEL_REGISTRY_NAME = os.getenv("MODEL_REGISTRY_NAME", "isb_aqi_model")
MODEL_DIR           = os.getenv("MODEL_DIR", "isb_aqi_model_dir")
MODEL_FILE          = os.getenv("MODEL_FILE", "model.pkl")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("train_model.log")]
)
log = logging.getLogger(__name__)

retry_registry = retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    reraise=True,
)

# ─── Load feature‑group ──────────────────────────────────────────────────
def load_featuregroup() -> tuple[pd.DataFrame, hopsworks.Project]:
    project = hopsworks.login(host=HOPSWORKS_HOST, api_key_value=HOPSWORKS_API_KEY)
    fs      = project.get_feature_store(name=FEATURESTORE_NAME)
    fg      = fs.get_feature_group(name=FEATUREGROUP_NAME, version=FG_VERSION)
    df      = fg.read()
    if df.empty:
        raise RuntimeError("Feature group is empty – cannot train.")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    log.info("Loaded %d rows (%s -> %s)",
             len(df), df["timestamp"].iloc[0].date(), df["timestamp"].iloc[-1].date())
    return df, project

# ─── Feature engineering ────────────────────────────────────────────────
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    base = [f"aqi_roll{w}" for w in (3, 7, 14)] + ["aqi"]
    for col in base:
        df[f"{col}_lag1"] = df[col].shift(1)
    df = df.dropna(subset=[f"{c}_lag1" if not c.endswith("_lag1") else c for c in base])
    log.info("After lags: %d rows × %d columns", *df.shape)
    return df.reset_index(drop=True)

# ─── Train & evaluate ───────────────────────────────────────────────────
def train_model(df: pd.DataFrame) -> tuple[xgb.XGBRegressor, float, list[str]]:
    X = df.filter(like="_lag1")
    y = df["aqi"]

    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth":    [3, 5, 7],
        "learning_rate":[0.05, 0.1],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(
        model, param_grid, cv=tscv,
        scoring="neg_root_mean_squared_error", verbose=1, n_jobs=-1
    )
    grid.fit(X, y)
    best = grid.best_estimator_
    log.info("Best params: %s", grid.best_params_)

    # rolling‑origin validation
    preds, actuals = [], []
    for i in range(tscv.n_splits, len(X)):
        best.fit(X.iloc[:i], y.iloc[:i])
        preds.append(best.predict(X.iloc[i:i+1])[0])
        actuals.append(y.iloc[i])
    rmse = math.sqrt(mean_squared_error(actuals, preds))
    log.info("Rolling origin RMSE = %.2f", rmse)
    return best, rmse, X.columns.tolist()

# ─── Register model ─────────────────────────────────────────────────────
@retry_registry
def register_model(project: hopsworks.Project,
                   model: xgb.XGBRegressor,
                   feat_order: list[str],
                   rmse: float) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    bundle_path = os.path.join(MODEL_DIR, MODEL_FILE)
    joblib.dump({"model": model, "feat_order": feat_order}, bundle_path)
    log.info("Model bundle written -> %s", bundle_path)

    registry = project.get_model_registry()
    py_model = registry.python.create_model(
        name        = MODEL_REGISTRY_NAME,
        metrics     = {"rmse": rmse},
        description = "XGBoost daily-AQI predictor (unit fixed data)",
    )
    py_model.save(MODEL_DIR)
    log.info("Registered model '%s' v%d", MODEL_REGISTRY_NAME, py_model.version)

# ─── Main ───────────────────────────────────────────────────────────────
def main() -> None:
    df_raw, project = load_featuregroup()
    df_feat         = prepare_features(df_raw)
    model, rmse, feats = train_model(df_feat)
    register_model(project, model, feats, rmse)
    log.info("Finished – RMSE %.2f", rmse)

if __name__ == "__main__":
    main()
