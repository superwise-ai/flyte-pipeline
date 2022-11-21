import json
import logging
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import boto3
import pandas as pd
from superwise import Superwise

from app.exceptions import EmptyServingModelException

CLIENT_ID = os.getenv("SUPERWISE_CLIENT_ID")
SECRET = os.getenv("SUPERWISE_SECRET")
SUPERWISE_MODEL_ID = os.getenv("SUPERWISE_MODEL_ID")
SUPERWISE_VERSION_ID = os.getenv("SUPERWISE_VERSION_ID")

logger = logging.getLogger("gunicorn.error")


@dataclass
class ServingModelMetadata:
    model_aws_path: str
    model_id: int
    version_id: int


class DiamondPricePredictor:
    persistent_model_path_file = '/tmp/model_path.json'

    def __init__(self):
        self._model = None
        self._load_initial_model_metadata()
        self._sw = Superwise(
            client_id=os.environ["SUPERWISE_CLIENT_ID"],
            secret=os.environ["SUPERWISE_SECRET"]
        )

    def _send_monitor_data(self, predictions):
        """
        send predictions and input data to Superwise

        :param pd.Serie prediction
        :return str transaction_id
        """
        logger.info(f"Logging predictions to Superwise")
        transaction_id = self._sw.transaction.log_records(
            model_id=int(self._superwise_model_id),
            version_id=int(self._superwise_version_id),
            records=predictions
        )
        return transaction_id

    def _get_current_model_metadata(self) -> Optional[ServingModelMetadata]:
        try:
            f = open(self.persistent_model_path_file, 'r')
        except FileNotFoundError:
            # still no serving model was set
            return None
        else:
            with f:
                model = json.load(f)
                return ServingModelMetadata(**model)

    def _load_initial_model_metadata(self):
        model_metadata = self._get_current_model_metadata()
        if model_metadata:
            self._set_model(
                model_aws_path=model_metadata.model_aws_path,
                model_id=model_metadata.model_id,
                version_id=model_metadata.version_id,
            )

    def _persist_model_metadata(self, model_aws_path, model_id, version_id):
        with open(self.persistent_model_path_file, 'w') as f:
            metadata = ServingModelMetadata(
                model_aws_path=model_aws_path,
                model_id=model_id,
                version_id=version_id
            )
            json.dump(metadata.__dict__, f)

    def _set_model(self, model_aws_path, model_id=1, version_id=1):
        """
        download file from s3 to temp file and deserialize it to sklearn object

        :param str model_s3_path: Path to s3 file
        :return sklearn.Pipeline model: Deserialized pipeline ready for production
        """
        logger.info(f"Update Superwise model-id ({model_id}) and version-id ({version_id})")
        self._superwise_model_id = model_id
        self._superwise_version_id = version_id

        logger.info(f"Loading model from {model_aws_path}...")
        s3 = boto3.resource('s3')
        bucket_name = os.environ["BUCKET_NAME"]
        if f"s3://{bucket_name}/" in model_aws_path:
            model_aws_path = model_aws_path.replace(f"s3://{bucket_name}/", "")
        self._model = pickle.loads(s3.Bucket(bucket_name).Object(model_aws_path).get()['Body'].read())
        logger.info(f"Model has been successfully loaded !")

    def update_model(self, model_aws_path, model_id=1, version_id=1):
        self._set_model(model_aws_path, model_id, version_id)
        self._persist_model_metadata(model_aws_path, model_id, version_id)

    def predict(self, instances):
        """
        apply predictions on instances and log predictions to Superwise

        :param list instances: [{record1}, {record2} ... {record-N}]
        :return dict api_output: {[predicted_prices: prediction, transaction_id: str]}
        """
        if not self._model:
            raise EmptyServingModelException("no trained model was found")
        logger.info(f"Predicting prices for {len(instances)}")
        input_df = pd.DataFrame(instances)
        # Add timestamp to prediction
        input_df["prediction"] = self._model.predict(input_df)
        # Send data to Superwise
        transaction_id = self._send_monitor_data(input_df)
        api_output = {
            "transaction_id": transaction_id,
            "predicted_prices": input_df["prediction"].values.tolist(),
        }
        return api_output
