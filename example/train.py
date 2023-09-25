#!/usr/bin/env python
import os
import shutil

import pandas as pd

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from dotenv import load_dotenv
import mlflow
from mlflow.environment_variables import (
    MLFLOW_S3_ENDPOINT_URL,
)

from model import Model


def main():
    load_dotenv()

    MLFLOW_S3_ENDPOINT_URL.set("http://localhost:9000")

    mlflow.set_tracking_uri("http://localhost:5000")
    experiment = mlflow.set_experiment("Default")
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        X, y = make_regression(1000, 3)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        model = Model()
        model.fit(X_train, y_train)

        y_pred = model.predict(model_input=X_test, context=None)
        rf_score = r2_score(y_test, y_pred)

        mlflow.log_metric("r2", rf_score)

        model_path = os.path.join("models", "model")
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        mlflow.pyfunc.save_model(
            path=model_path, python_model=model, extra_pip_requirements=["pymysql"]
        )
        mlflow.pyfunc.log_model(
            model_path,
            python_model=model,
            code_path=[
                "example/model.py",
                "example/features.py",
            ],
        )
        print(f"run_id: {run.info}")


if __name__ == "__main__":
    main()
