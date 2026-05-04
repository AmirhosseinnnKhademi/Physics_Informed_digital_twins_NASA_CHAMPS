"""Helper that configures MLflow with DagsHub credentials from .env."""
import os
from dotenv import load_dotenv
import mlflow

def setup_mlflow(experiment_name: str) -> None:
    """Set tracking URI with Basic-Auth and activate experiment."""
    load_dotenv()
    tracking_uri = os.getenv("MLflow_tracking_uri", "mlruns")
    user = os.getenv("DAGSHUB_USER", "")
    token = os.getenv("DAGSHUB_TOKEN", "")
    if user and token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    mlflow.set_tracking_uri(tracking_uri)
    try:
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        print(f"[mlflow] Remote tracking unavailable ({e}), falling back to local mlruns/")
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment(experiment_name)
