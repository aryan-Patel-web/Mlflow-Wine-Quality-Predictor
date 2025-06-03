# Wine Quality Prediction with MLflow & DAGsHub
# Dataset source: http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# Authors: P. Cortez et al., Decision Support Systems, Elsevier, 2009

import os
import warnings
import sys
import logging
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn

# ------------------- DAGsHub Setup -------------------
from dagshub.common.init import init as dagshub_init

# Set your credentials (if needed in non-interactive environments)
os.environ["DAGSHUB_USER"] = "your-username"  # optional
os.environ["DAGSHUB_TOKEN"] = "your-access-token"  # optional

dagshub_init(repo_owner='aryan-Patel-web', repo_name='Mlflow-Wine-Quality-Predictor', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/aryan-Patel-web/Mlflow-Wine-Quality-Predictor.mlflow")

# ------------------- Logging Setup -------------------
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# ------------------- Evaluation Metrics -------------------
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# ------------------- Main Script -------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Load dataset
    csv_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception("Failed to download dataset. Error: %s", e)
        sys.exit(1)

    # Split dataset
    train, test = train_test_split(data, test_size=0.25, random_state=42)
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # Hyperparameters
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # Start MLflow run
    with mlflow.start_run():
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(train_x, train_y)
        predictions = model.predict(test_x)

        rmse, mae, r2 = eval_metrics(test_y, predictions)

        print(f"ElasticNet Model (alpha={alpha}, l1_ratio={l1_ratio}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        # Log hyperparameters and metrics
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", float(rmse))
        mlflow.log_metric("mae", float(mae))
        mlflow.log_metric("r2", float(r2))

        # Log model
        tracking_type = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_type != "file":
            mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetWineModel") # type: ignore
        else:
            mlflow.sklearn.log_model(model, "model") # type: ignore
