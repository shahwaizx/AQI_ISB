#!/usr/bin/env python3
import os
import logging
from datetime import datetime
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
import joblib
import hopsworks
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv
import math

# ----------------------
# Load environment variables
# ----------------------
load_dotenv()

HOPSWORKS_API_KEY   = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_HOST      = os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai")
FEATURESTORE_NAME   = os.getenv("FEATURESTORE_NAME", "aqi_islamabad_featurestore")
FEATUREGROUP_NAME   = os.getenv("FEATUREGROUP_NAME", "isb_aqi_history")
FG_VERSION          = int(os.getenv("FG_VERSION", "1"))

MODEL_REGISTRY_NAME = os.getenv("MODEL_REGISTRY_NAME", "isb_aqi_model")
MODEL_DIR           = os.getenv("MODEL_DIR", "isb_aqi_model_dir")
MODEL_FILE          = os.getenv("MODEL_FILE", "model.pkl")

# ----------------------
# Logging setup
# ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("train_model.log")]
)
logger = logging.getLogger(__name__)

# ----------------------
# Retry policy for registry calls
# ----------------------
retry_registry = retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    reraise=True,
)

# ----------------------
# AQI Breakpoints for sub-index calculation
# ----------------------
BREAKPOINTS = {
    "pm2_5": [(0.0,12.0,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
              (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,500.4,301,500)],
    "pm10":  [(0,54,0,50),(55,154,51,100),(155,254,101,150),
              (255,354,151,200),(355,424,201,300),(425,604,301,500)]
}

def compute_individual_aqi(cp: float, breakpoints: list) -> int:
    if cp is None:
        return None
    for Cl, Ch, Il, Ih in breakpoints:
        if Cl <= cp <= Ch:
            return int(((Ih - Il) / (Ch - Cl)) * (cp - Cl) + Il)
    return 500

# ----------------------
# Load data from Hopsworks
# ----------------------
def load_featuregroup():
    logger.info("Logging into Hopsworks at %s...", HOPSWORKS_HOST)
    project = hopsworks.login(host=HOPSWORKS_HOST, api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store(name=FEATURESTORE_NAME)
    fg = fs.get_feature_group(name=FEATUREGROUP_NAME, version=FG_VERSION)
    df = fg.read()
    if df.empty:
        raise RuntimeError(f"Feature group {FEATUREGROUP_NAME} is empty.")
    logger.info("Loaded %d rows from %s v%d", len(df), FEATUREGROUP_NAME, FG_VERSION)
    return df, project

# ----------------------
# Engineer sub-indices if missing
# ----------------------
def engineer_subindices(df: pd.DataFrame) -> pd.DataFrame:
    if "pm2_5_idx" not in df.columns and "pm2_5" in df.columns:
        df["pm2_5_idx"] = df["pm2_5"].map(lambda v: compute_individual_aqi(v, BREAKPOINTS["pm2_5"]))
    if "pm10_idx" not in df.columns and "pm10" in df.columns:
        df["pm10_idx"] = df["pm10"].map(lambda v: compute_individual_aqi(v, BREAKPOINTS["pm10"]))
    return df

# ----------------------
# Prepare features and lags
# ----------------------
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preparing and lagging features...")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = engineer_subindices(df)

    idx_cols = [c for c in df.columns if c.endswith("_idx")] + ["aqi"]
    for col in idx_cols:
        for lag in (1, 2, 3):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    df = df.dropna().reset_index(drop=True)
    logger.info("After lagging: %d rows, %d cols", df.shape[0], df.shape[1])
    return df

# ----------------------
# Train and evaluate
# ----------------------
def train_and_evaluate(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if any(c.endswith(f"_lag{l}") for l in (1,2,3))]
    X = df[feature_cols]
    y = df["aqi"]
    logger.info("Training on %d samples with %d features", len(X), X.shape[1])

    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1]
    }
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(
        model, param_grid, cv=tscv,
        scoring="neg_root_mean_squared_error", verbose=2
    )
    grid.fit(X, y)
    best = grid.best_estimator_
    logger.info("Best params: %s", grid.best_params_)

    preds, actuals = [], []
    for i in range(tscv.n_splits, len(X)):
        X_train, y_train = X.iloc[:i], y.iloc[:i]
        X_val, y_val = X.iloc[i:i+1], y.iloc[i]
        best.fit(X_train, y_train)
        preds.append(best.predict(X_val)[0])
        actuals.append(y_val)

    mse = mean_squared_error(actuals, preds)
    rmse = math.sqrt(mse)
    logger.info("Rolling RMSE: %.3f", rmse)
    return best, float(rmse)

# ----------------------
# Save locally & register with correct API call
# ----------------------
def register_model(project, model, rmse: float):
    # persist model locally
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, MODEL_FILE)
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)

    # register in Hopsworks
    registry = project.get_model_registry()
    py_model = registry.python.create_model(
        name=MODEL_REGISTRY_NAME,
        metrics={"rmse": rmse},
        description="XGBoost-based AQI predictor"
    )
    # pass directory as positional arg
    py_model.save(MODEL_DIR)
    logger.info("Registered model '%s' version %d", MODEL_REGISTRY_NAME, py_model.version)

# ----------------------
# Main
# ----------------------
def main():
    df, project = load_featuregroup()
    df_feat = prepare_features(df)
    model, rmse = train_and_evaluate(df_feat)
    register_model(project, model, rmse)
    logger.info("Training and registration pipeline completed.")

if __name__ == "__main__":
    main()
