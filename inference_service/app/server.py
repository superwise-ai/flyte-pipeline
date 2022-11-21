import logging

from flask import Flask, jsonify, request
from flytekit.configuration import Config
from flytekit.remote import FlyteRemote

from app.predictor import DiamondPricePredictor
from app.exceptions import EmptyServingModelException

app = Flask("DiamondPricePredictor")
gunicorn_logger = logging.getLogger("gunicorn.debug")
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)
diamonds_predictor = DiamondPricePredictor()


@app.route("/diamonds/v1/predict", methods=["POST"])
def predict():
    """
    Handle the Endpoint predict request
    expected in the following format:
        {
            "instances": [
            {
                "carat" : 1.42, "clarity" : "VVS1", "color" : "F", "cut" : "Ideal", "depth" : 60.8, "record_id" : 27671, "table" : 56, "x" : 7.25, "y" : 7.32, "z" : 4.43
            },
            {
                "carat" : 2.03, "clarity" : "VS2", "color" : "G", "cut" : "Premium", "depth" : 59.6, "record_id" : 27670, "table" : 60, "x" : 8.27, "y" : 8.21, "z" : 4.91
            }
            ]
        }
    """
    try:
        predictions = diamonds_predictor.predict(request.json)
    except EmptyServingModelException as e:
        return str(e), 404
    else:
        return jsonify(
            {
                "predictions": predictions["predicted_prices"],
                "transaction_id": predictions["transaction_id"],
            }
        )


@app.route("/diamonds/v1/update-model", methods=["POST"])
def update_model():
    """
    Update the model
    """
    diamonds_predictor.update_model(request.json["model_uri"], request.json["model_id"], request.json["version_id"])
    return jsonify(status_code=200)
    

@app.route("/diamonds/v1/trigger-pipeline", methods=["POST"])
def trigger_pipeline():
    """
    Trigger training-serving pipeline
    """
    remote = FlyteRemote(config=Config.for_sandbox(), default_project="flytesnacks", default_domain="development")
    flyte_workflow = remote.fetch_workflow(name="core.basic_pipeline.ml_pipeline")
    execution = remote.execute(flyte_workflow, inputs={}, execution_name="retrain", wait=False)
    return jsonify(status_code=200)


@app.route("/diamonds/v1/healthcheck", methods=["GET"])
def healthcheck():
    """
    Validate service health
    """
    resp = jsonify(health="Diamonds Prediction Service is Alive!")
    resp.status_code = 200
    return resp


if __name__ == "__main__":
    app.run(host="localhost")
