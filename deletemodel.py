import hopsworks

# Connect to your Hopsworks project
project = hopsworks.login(
        host="c.app.hopsworks.ai",
        api_key_value="gLSSl3rhxHuDtcgF.0qpN3LM2nlJREY39NBkp0pUGNg4a4mwh1mfgEZSdvvZ5Vx2i5LdwHOSzVN18Dse9"
    )
model_registry = project.get_model_registry()

# Replace with your model's name and version
model_name = "lahore_aqi_model"
model_version = 1  # or whatever version you want to delete

# Get the model
model = model_registry.get_model(model_name, version=model_version)

# Delete the model
model.delete()
print(f"Model '{model_name}' version {model_version} deleted successfully.")
