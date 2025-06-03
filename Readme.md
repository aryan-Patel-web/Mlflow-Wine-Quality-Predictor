
## ML FLOW Experiments

import dagshub
dagshub.init(repo_owner='aryan-Patel-web', repo_name='Mlflow-Wine-Quality-Predictor', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)

 MLFLOW_TRACKING_URI = https://dagshub.com/aryan-Patel-web/Mlflow-Wine-Quality-Predictor.mlflow

MLFLOW_TRACKING_USERNAME=aryan-Patel-web
MLFLOW_TRACKING_PASSWORD=