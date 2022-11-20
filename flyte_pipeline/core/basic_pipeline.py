import os
import inspect
import typing
import joblib
import mlflow
import boto3
import requests
import flytekitplugins.pandera
import pandas as pd
import seaborn as sns

from cmath import log
from os import pathconf
from random import randint
from datetime import datetime
from flytekit import task, workflow, Resources
from flytekit.types.file import JoblibSerializedFile
from pandera.typing import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from mlflow import log_metric, log_param
from superwise import Superwise
from superwise.models.model import Model
from superwise.models.version import Version
from superwise.resources.superwise_enums import DataEntityRole, DatasetType
from superwise.models.dataset import Dataset

from core.validations import RawData, TargetSerie, TrainData, PredictionData
from core.utils.metrics import eval_metrics

TIMESTAMP = datetime.now().strftime('%Y%m%d%H%M%S')
MODEL_NAME = "diamonds_predictor"
LOCAL_IP = "172.16.80.54"
SUPERWISE_PROJECT_ID = 4

ModelPerformance = typing.NamedTuple("ModelPerformance", performance_metrics=typing.Dict,
                                     test_dataset_with_prediction=DataFrame[PredictionData])
DataSplits = typing.NamedTuple("DataSplit", x_train=DataFrame[TrainData], y_train=DataFrame[TargetSerie],
                               x_test=DataFrame[TrainData], y_test=DataFrame[TargetSerie])
mlflow.set_tracking_uri(f"http://{LOCAL_IP}:80")
experiment = mlflow.set_experiment("webinar-diamonds-training-serving-pipeline")


@task(cache=True, cache_version="1.0")
def extract_data(price_threshold: int) -> DataFrame[RawData]:
    data_url = "https://www.openml.org/data/get_csv/21792853/dataset"
    print(f"Reading dataset from {data_url}")
    df = pd.read_csv(data_url)
    print(f"Dataset read successfully, shape {df.shape}")
    if price_threshold:
        print(f"filtering diamonds prices exceeding {price_threshold}$")
        df = df[df['price'] < price_threshold]
    return df


@task(cache=True, cache_version="1.0")
def validate_data(df: DataFrame[RawData]) -> DataFrame[RawData]:
    BINARY_FEATURES = []
    # List all column names for numeric features
    NUMERIC_FEATURES = ["carat", "depth", "table", "x", "y", "z"]
    # List all column names for categorical features
    CATEGORICAL_FEATURES = ["cut", "color", "clarity"]
    # ID column - needed to support predict() over numpy arrays
    ID = ["record_id"]
    TARGET = "price"
    ALL_COLUMNS = ID + BINARY_FEATURES + NUMERIC_FEATURES + CATEGORICAL_FEATURES
    # define the column name for the target
    df = df.reset_index().rename(columns={"index": "record_id"})
    for n in NUMERIC_FEATURES:
        df[n] = pd.to_numeric(df[n], errors="coerce")

    df = df.fillna(df.mean(numeric_only=True))

    def data_selection(df: pd.DataFrame, selected_columns: typing.List[str]):
        selected_columns.append(TARGET)
        data = df.loc[:, selected_columns]
        return data

    ## Feature selection
    df = data_selection(df, ALL_COLUMNS)
    return df


@task(cache=True, cache_version="1.0")
def prepare_data(df: DataFrame[RawData]) -> DataSplits:
    target = "price"
    X, y = df.drop(columns=[target]), df[target]
    X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train_data, y_train_data.to_frame(), X_test_data, y_test_data.to_frame()


@task(cache=True, cache_version="1.0", timeout=1600, requests=Resources(cpu="2", mem="1Gi"))
def train_model(x_train: DataFrame[TrainData], y_train: DataFrame[TargetSerie]) -> JoblibSerializedFile:
    print("Trainning model started")
    # List all column names for numeric features
    NUMERIC_FEATURES = ["carat", "depth", "table", "x", "y", "z"]
    # List all column names for categorical features
    CATEGORICAL_FEATURES = ["cut", "color", "clarity"]
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("cat", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        n_jobs=-1,
    )
    # We now create a full pipeline, for preprocessing and training
    # for training we selected a RandomForestRegressor
    estimators = randint(450, 550)
    model_params = {"max_features": "auto",
                    "n_estimators": estimators,
                    "max_depth": 9,
                    "random_state": 42}
    regressor = RandomForestRegressor()
    regressor.set_params(**model_params)
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", regressor)]
    )
    # For Workshop time efficiency we will use 2-fold cross validation
    k = 2

    score = cross_val_score(
        pipeline, x_train, y_train['price'], cv=k, scoring="neg_root_mean_squared_error", n_jobs=2
    ).mean()

    print("finished cross val")
    # Now we fit all our data to the classifier.
    pipeline.fit(x_train, y_train['price'])
    print(f"score: {score}, pipeline: {pipeline}")

    model_name = f"model_{TIMESTAMP}"
    model_fp = f"{model_name}.joblib"

    os.environ['AWS_PROFILE'] = "mlflow"
    boto3.client('s3')
    with mlflow.start_run(run_name=inspect.currentframe().f_code.co_name):
        mlflow.log_param("n_estimators", estimators)
        mlflow.log_metric(f"{k}_fold_cross_val_score", score)
        mlflow.sklearn.log_model(pipeline, "trained_model")

    joblib.dump(pipeline, model_fp)
    return JoblibSerializedFile(path=model_fp)


@task(requests=Resources(cpu="1", mem="1Gi"))
def evaluate_model(model: JoblibSerializedFile, x_test: DataFrame[TrainData],
                   y_test: DataFrame[TargetSerie]) -> ModelPerformance:
    model = joblib.load(model)

    model_performance = {}
    os.environ['AWS_PROFILE'] = "mlflow"
    boto3.client('s3')

    with mlflow.start_run(run_name=inspect.currentframe().f_code.co_name):
        y_pred = model.predict(x_test)
        rmse, mae, r2 = eval_metrics(y_test['price'], y_pred)
        model_performance["rmse"] = rmse
        model_performance["mae"] = mae
        model_performance["r2"] = r2
        mlflow.log_metric(f"rmse", rmse)
        mlflow.log_metric(f"mae", mae)
        mlflow.log_metric(f"r2", r2)

        df = pd.DataFrame({"predicted Price(USD)": y_pred, "actual Price(USD)": y_test["price"]})
        scatter_plot = sns.scatterplot(data=df, x="predicted Price(USD)", y="actual Price(USD)")
        mlflow.log_figure(scatter_plot.get_figure(), "prices_scatterplot.png")

    test_dataset = x_test.copy()
    test_dataset["price"] = y_test["price"]
    test_dataset["prediction"] = y_pred

    return model_performance, test_dataset


@task(requests=Resources(cpu="1", mem="1Gi"))
def validate_model(model_metrics: typing.Dict, new_model: JoblibSerializedFile,
                   dataset: DataFrame[RawData]) -> typing.Dict:
    deploy_dict = {}
    target = "price"
    X, y = dataset.drop(columns=[target]), dataset[target]
    model_candidate = joblib.load(new_model)

    os.environ['AWS_PROFILE'] = "mlflow"
    boto3.client('s3')

    with mlflow.start_run(run_name=inspect.currentframe().f_code.co_name):
        y_pred = model_candidate.predict(X)
        rmse, mae, r2 = eval_metrics(y, y_pred)
        mlflow.log_metric("train_rmse", model_metrics['rmse'])
        mlflow.log_metric("train_mae", model_metrics['mae'])
        mlflow.log_metric("train_r2", model_metrics['r2'])
        mlflow.log_metric("validation_rmse", rmse)
        mlflow.log_metric("validation_mae", mae)
        mlflow.log_metric("validation_r2", r2)
        mlflow.sklearn.log_model(model_candidate, MODEL_NAME)
        run_id = mlflow.active_run().info.run_uuid

    print(f"new model train rmse: {model_metrics['rmse']}")
    print(f"new model train r2: {model_metrics['r2']}")
    print(f"new model validation rmse: {rmse}")
    print(f"new model validation r2: {r2}")

    if (
            rmse <= model_metrics["rmse"]
            and model_metrics["r2"] >= 0.95
            and abs(model_metrics['rmse']) < 1000
    ):
        mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
        deploy_dict["run_id"] = run_id
        deploy_dict["to_deploy"] = 1
    else:
        deploy_dict["to_deploy"] = 0

    return deploy_dict


@task
def register_to_superwise(deploy_dict: dict,
                          model_name: str,
                          test_dataset_with_prediction: DataFrame[PredictionData],
                          ) -> dict:
    to_deploy = deploy_dict["to_deploy"]
    print("Is model ready for deployment?")
    print(f"*** {to_deploy == 1} ***")

    if not to_deploy:
        return to_deploy

    sw = Superwise()

    print("Register the new model to Superwise")

    first_version = False
    # Check if model exists
    models = sw.model.get_by_name(model_name)
    if len(models) == 0:
        print(f"Registering new model {model_name} to Superwise")
        diamond_model = Model(project_id=SUPERWISE_PROJECT_ID, name=model_name, description="Predicting Diamond Prices")
        new_model = sw.model.create(diamond_model)
        model_id = new_model.id
        first_version = True
    else:
        print(f"Model {model_name} already exists in Superwise")
        model_id = models[0].id

    # adding timestamp field for superwise
    test_dataset = test_dataset_with_prediction.assign(
        ts=pd.Timestamp.now() - pd.Timedelta(30, "d")
    )

    new_version_name = f"v_{TIMESTAMP}"
    # writing df into disk for superwise's dataset creation
    train_dataset_path = "train_dataset.csv"
    test_dataset.to_csv(train_dataset_path)

    roles = {
        DataEntityRole.LABEL.value: ["price"],
        DataEntityRole.PREDICTION_VALUE.value: ["prediction"],
        DataEntityRole.TIMESTAMP.value: "ts",
        DataEntityRole.ID.value: "record_id"
    }

    dataset = Dataset(name=f"{new_version_name}_training", files=[train_dataset_path], project_id=SUPERWISE_PROJECT_ID,
                      type=DatasetType.TRAIN, roles=roles)
    dataset = sw.dataset.create(dataset)

    if not first_version:
        model_versions = sw.version.get({"model_id": model_id})
        print(
            f"Model already has the following versions: {[v.name for v in model_versions]}"
        )

    # create new version for model in Superwise
    diamond_version = Version(
        model_id=model_id,
        name=new_version_name,
        dataset_id=dataset.id,
    )
    new_version = sw.version.create(diamond_version)
    # activate the new version for monitoring
    sw.version.activate(new_version.id)
    deploy_dict.update({"model_id": model_id, "version_id": new_version.id})
    return deploy_dict


@task
def deploy_model(deployment_dict: typing.Dict) -> typing.Dict:
    if not deployment_dict["to_deploy"]:
        return

    model_uri = f"{mlflow.get_run(deployment_dict['run_id']).to_dictionary()['info']['artifact_uri']}/{MODEL_NAME}/model.pkl"
    print(f"Deploying model from {model_uri}")
    res = requests.post(f"http://{LOCAL_IP}:5050/diamonds/v1/update-model",
                        json={"model_uri": model_uri,
                              "model_id": deployment_dict["model_id"],
                              "version_id": deployment_dict["version_id"]})
    res.raise_for_status()
    return {"status_code": res.status_code}


@workflow
def ml_pipeline(diamonds_price_threshold: int = 0):
    raw_data = extract_data(price_threshold=diamonds_price_threshold)
    validated_data = validate_data(df=raw_data)
    x_train, y_train, x_test, y_test = prepare_data(df=validated_data)
    trained_model = train_model(x_train=x_train, y_train=y_train)
    performance_metrics, test_dataset_with_prediction = evaluate_model(model=trained_model, x_test=x_test, y_test=y_test)
    deployment_dict = validate_model(model_metrics=performance_metrics, new_model=trained_model, dataset=validated_data)

    superwise_deployment_dict = register_to_superwise(
        deploy_dict=deployment_dict,
        model_name="Diamonds Price Predictor 3",
        test_dataset_with_prediction=test_dataset_with_prediction,
    )

    deploy_model(deployment_dict=superwise_deployment_dict)


def main():
    ml_pipeline()


if __name__ == "__main__":
    main()
