import hopsworks
import os
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upload_model_to_hopsworks(model_dir, filename, model_name, rmse):
    # Login to Hopsworks
    logger.info("Logging in to Hopsworks...")
    project = hopsworks.login(
        host="c.app.hopsworks.ai",
        api_key_value="gLSSl3rhxHuDtcgF.0qpN3LM2nlJREY39NBkp0pUGNg4a4mwh1mfgEZSdvvZ5Vx2i5LdwHOSzVN18Dse9"
    )

    # Get model registry
    mr = project.get_model_registry()

    # Create and register the model
    logger.info("Creating model entry in Hopsworks...")
    model_obj = mr.python.create_model(
        name=model_name,
        metrics={"RMSE": rmse},
        description="Uploaded from separate script"
    )

    logger.info("Uploading model directory to Hopsworks...")
    model_obj.save(model_dir)

    logger.info("✅ Model successfully uploaded to Hopsworks registry.")

if __name__ == "__main__":
    MODEL_DIR = "isb_aqi_model_dir"
    FILENAME = "isb_aqi_model.pkl"
    MODEL_NAME = "isb_aqi_model"
    RMSE = 45.226 # Replace with your actual value if known

    upload_model_to_hopsworks(MODEL_DIR, FILENAME, MODEL_NAME, RMSE)
