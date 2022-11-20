# SKLearn Model Orchestration
Basic Flask App that demonstrates how to deploy sklearn model and log predictions into Superwise's platform

***

<p align="center" width="100%">
<b>🚧 THIS IS FOR DEMONSTRATION PURPOSE ONLY 🚧</b>
<p align="center" width="100%">
<b>🚧 DO NOT USE IT IN PRODUCTION 🚧</b>

***

# Usage
- `python3 -m venv venv && pip install -r requirements.txt`

- Build image:
  ```
  gunicorn --bind 0.0.0.0:5050 predictor.server:app --timeout 100 -w 1 -e BUCKET_NAME=$BUCKET_NAME -e SUPERWISE_CLIENT_ID=$SUPERWISE_CLIENT_ID -e SUPERWISE_SECRET=$SUPERWISE_SECRET
  ```

- Health check: 
  
  `curl -X GET http://localhost:5050/diamonds/v1/healthcheck`

- Prediction:
  
  `
  curl -X POST 'localhost:5050/diamonds/v1/predict' -H "Content-Type: application/json" -d '[{"carat" : 1.42, "clarity" : "VVS1", "color" : "F", "cut" : "Ideal", "depth" : 60.8, "record_id" : 27671, "table" : 56, "x" : 7.25, "y" : 7.32, "z" : 4.43, "ts": "2022-01-01 00:00:00"}]'
  `

- update model:

  `
  curl -X POST 'localhost:5050/diamonds/v1/update-model' -H "Content-Type: application/json" -d '{"model_uri": "mlflow/1/04a18e09ffa64f9da0b818e1074b01bf/artifacts/validated_model/model.pkl"}'
  `

- trigger pipeline:

  `
  curl -X POST 'localhost:5050/diamonds/v1/trigger-pipeline'
  `