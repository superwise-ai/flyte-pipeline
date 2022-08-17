import logging
import os
import pickle
import pandas as pd
import boto3

from superwise import Superwise

CLIENT_ID = os.getenv("SUPERWISE_CLIENT_ID")
SECRET = os.getenv("SUPERWISE_SECRET")
SUPERWISE_MODEL_ID = os.getenv("SUPERWISE_MODEL_ID")
SUPERWISE_VERSION_ID = os.getenv("SUPERWISE_VERSION_ID")

logger = logging.getLogger("gunicorn.error")

class DiamondPricePredictor:
    def __init__(self, model_s3_path):
        self._model = self.set_model(model_s3_path)
        self._sw = Superwise(
           client_id=os.getenv("SUPERWISE_CLIENT_ID"),
           secret=os.getenv("SUPERWISE_SECRET")
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

    def set_model(self, model_aws_path, model_id=1, version_id=1):
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
        model = pickle.loads(s3.Bucket(bucket_name).Object(model_aws_path).get()['Body'].read()) 
        logger.info(f"Model has been successfully loaded !")
        return model

    def predict(self, instances):
        """
        apply predictions on instances and log predictions to Superwise

        :param list instances: [{record1}, {record2} ... {record-N}]
        :return dict api_output: {[predicted_prices: prediction, transaction_id: str]}
        """
        logger.info(f"Predicting prices for {len(instances)}")
        input_df = pd.DataFrame(instances)
        # Add timestamp to prediction
        input_df["predictions"] = self._model.predict(input_df)
        # Send data to Superwise
        transaction_id = self._send_monitor_data(input_df)
        api_output = {
            "transaction_id": transaction_id,
            "predicted_prices": input_df["predictions"].values.tolist(),
        }
        return api_output
