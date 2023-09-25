import os

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from mlflow.pyfunc import PythonModel

from features import FeatureStore


def feature_augmenter(features_in: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    result = features_in.copy() if copy else features_in
    conn_string = f"mysql+pymysql://mlflow_user:mlflow@localhost:3306/mlflow_database"
    fs = FeatureStore(conn_string)
    result["market"] = fs.get("market")
    result["answer"] = fs.get("answer")
    result["holidays"] = fs.get("holidays")

    print(result)
    return result


def feature_names_augmentor(feature_names_in: list[str]) -> list[str]:
    feature_names_out = feature_names_in.copy()
    feature_names_out.extend(
        [
            "market",
            "answer",
            "holidays",
        ]
    )


class Model(PythonModel):
    def __init__(self):
        self._conn_string = (
            f"mysql+pymysql://mlflow_user:mlflow@localhost:3306/mlflow_database"
        )
        self._pipe = Pipeline(
            steps=[
                (
                    "feat_augmentor",
                    FunctionTransformer(
                        feature_augmenter,
                        check_inverse=False,
                        feature_names_out=feature_names_augmentor,
                    ),
                ),
                ("regressor", RandomForestRegressor()),
            ]
        )

    def fit(self, X, y) -> "Model":
        self._pipe.fit(X, y)
        return self

    def predict(self, context, model_input, params=None):
        return self._pipe.predict(model_input)
