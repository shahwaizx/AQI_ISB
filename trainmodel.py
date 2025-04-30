import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
import joblib
import hopsworks
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("train_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data_from_featurestore():
    logger.info("Logging into Hopsworks and loading feature group data...")
    try:
        project = hopsworks.login(
            host="c.app.hopsworks.ai",
            api_key_value="gLSSl3rhxHuDtcgF.0qpN3LM2nlJREY39NBkp0pUGNg4a4mwh1mfgEZSdvvZ5Vx2i5LdwHOSzVN18Dse9"
        )
        fs = project.get_feature_store(name="aqi_islamabad_featurestore")
        fg = fs.get_feature_group(name="isb_aqi_history", version=1)
        df = fg.read()
        if df.empty:
            logger.error("Feature group 'isb_aqi_history' is empty.")
            raise ValueError("Feature group 'isb_aqi_history' is empty.")
        logger.info(f"Loaded {len(df)} records from feature group.")
        logger.info(f"Sample data:\n{df[['timestamp', 'aqi']].head(5).to_string()}")
        logger.info(f"AQI range: {df['aqi'].min()} to {df['aqi'].max()}")
        return df, fs
    except Exception as e:
        logger.error(f"Error loading feature group: {e}")
        raise

def prepare_data(df):
    logger.info("Preparing and sorting data by timestamp...")
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[["timestamp", "aqi"]].sort_values("timestamp").reset_index(drop=True)
        if len(df) < 10:
            logger.error(f"Insufficient data: {len(df)} rows.")
            raise ValueError(f"Insufficient data: {len(df)} rows.")
        return df
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        raise

def create_lag_features(df, target_col="aqi", lags=[1, 2, 3]):
    logger.info("Creating lag features for AQI...")
    try:
        for lag in lags:
            df[f"aqi_lag{lag}"] = df[target_col].shift(lag)
        df = df.dropna().reset_index(drop=True)
        if len(df) < 10:
            logger.warning(f"Insufficient data after creating lag features: {len(df)} rows.")
            raise ValueError(f"Insufficient data after creating lag features: {len(df)} rows.")
        logger.info(f"Created lag features, {len(df)} rows remaining.")
        return df
    except Exception as e:
        logger.error(f"Error creating lag features: {e}")
        raise

def train_model(X, y):
    logger.info("Training XGBoost model with GridSearchCV...")
    try:
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
            "learning_rate": [0.01, 0.1]
        }
        tscv = TimeSeriesSplit(n_splits=3)
        grid = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_root_mean_squared_error', verbose=1)
        grid.fit(X, y)
        logger.info("Best params: %s", grid.best_params_)
        return grid.best_estimator_
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

def evaluate_model(model, X, y, max_lag=3):
    logger.info("Evaluating model with rolling forecast...")
    try:
        min_rows = max_lag + 5
        if len(X) < min_rows:
            logger.warning(f"Insufficient data for evaluation: {len(X)} rows, need {min_rows}.")
            return None
        
        test_size = max(10, int(len(X) * 0.2))
        train_size = len(X) - test_size
        if train_size < max_lag:
            logger.warning("Training set too small for rolling forecast.")
            return None

        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        model.fit(X_train, y_train)
        
        preds = []
        actuals = []
        for i in range(max_lag, len(X_test)):
            try:
                X_row = X_test.iloc[i - max_lag:i].tail(1)
                if X_row.empty:
                    continue
                pred = model.predict(X_row.values)[0]
                preds.append(pred)
                actuals.append(y_test.iloc[i])
            except Exception as e:
                logger.warning(f"Error in rolling forecast iteration {i}: {e}")
                continue
        
        if not preds:
            logger.warning("No valid predictions generated.")
            return None
        
        rmse = mean_squared_error(actuals, preds) ** 0.5
        logger.info("Rolling RMSE: %.3f", rmse)
        return rmse
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return None

def save_and_register_model(fs, model, filename="isb_aqi_model.pkl", model_name="isb_aqi_model", rmse=None):
    logger.info("Saving and uploading model to Hopsworks...")
    try:
        model_dir = "isb_aqi_model_dir"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, filename)
        
        joblib.dump(model, model_path)
        logger.info("Model saved to %s", model_path)

    except Exception as e:
        logger.error(f"Error saving/registering model: {e}")
        raise

def main():
    try:
        df, fs = load_data_from_featurestore()
        df = prepare_data(df)
        df = create_lag_features(df, target_col="aqi")
        features = [col for col in df.columns if col.startswith("aqi_lag")]
        X = df[features]
        y = df["aqi"]
        logger.info(f"Training with {len(X)} samples, {len(features)} features.")
        model = train_model(X, y)
        rmse = evaluate_model(model, X, y)
        save_and_register_model(fs, model, rmse=rmse)
        logger.info("Training pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise

if __name__ == "__main__":
    main()