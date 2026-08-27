import os
import sys
import traceback
import joblib
import mlflow
import mlflow.sklearn

# Configure MLflow SQLite tracking backend matching train.py
DB_PATH = "sqlite:///mlflow.db"
MODEL_REGISTRY_URI = "models:/fraudguard360-detector/Staging"
LOCAL_FALLBACK_PATH = "models/best_model.joblib"

# Set module-level tracking URI
mlflow.set_tracking_uri(DB_PATH)


def load_production_model():
    """
    Loads the production-ready model for FraudGuard 360.
    
    Attempts to fetch the model version currently in 'Staging' from the MLflow Model Registry.
    If the registry is unavailable or loading fails, falls back gracefully to loading
    models/best_model.joblib locally.

    Returns
    -------
    tuple
        (model_object, source_string) where source_string is 'registry' or 'local_fallback'.
    """
    # 1. Explicitly set tracking URI at start of load_production_model
    mlflow.set_tracking_uri(DB_PATH)
    active_tracking_uri = mlflow.get_tracking_uri()
    
    print(f"[model_loader] Active MLflow Tracking URI: {active_tracking_uri}")
    print(f"[model_loader] Attempting to load model from MLflow Registry: '{MODEL_REGISTRY_URI}'...")

    # 2. Try registry load with full exception reporting and trusted type handling
    try:
        trusted_types = [
            "sklearn.calibration._CalibratedClassifier",
            "sklearn.calibration._SigmoidCalibration"
        ]
        
        try:
            loaded_model = mlflow.sklearn.load_model(MODEL_REGISTRY_URI, skops_trusted_types=trusted_types)
        except TypeError:
            # Fallback if load_model signature doesn't take skops_trusted_types in this version
            loaded_model = mlflow.sklearn.load_model(MODEL_REGISTRY_URI)

        print(f"[model_loader] Successfully loaded model from MLflow Registry ('Staging' stage).")
        return loaded_model, "registry"
    except Exception as err:
        print(f"[WARNING] Failed to load model from MLflow Model Registry ('{MODEL_REGISTRY_URI}').")
        print(f"[WARNING] Exception Type: {type(err).__name__}")
        print(f"[WARNING] Full Exception Message:\n{err}")
        print("[WARNING] Stack Trace:")
        traceback.print_exc(file=sys.stdout)
        print(f"[model_loader] Falling back to local model artifact at '{LOCAL_FALLBACK_PATH}'...")

        if not os.path.exists(LOCAL_FALLBACK_PATH):
            raise FileNotFoundError(
                f"Neither MLflow Registry model ('{MODEL_REGISTRY_URI}') nor local fallback file "
                f"at '{LOCAL_FALLBACK_PATH}' could be loaded."
            )

        loaded_model = joblib.load(LOCAL_FALLBACK_PATH)
        print(f"[model_loader] Successfully loaded model from local fallback file '{LOCAL_FALLBACK_PATH}'.")
        return loaded_model, "local_fallback"


if __name__ == "__main__":
    model_obj, source = load_production_model()
    print(f"Loaded model type: {type(model_obj).__name__}, Source: {source}")
