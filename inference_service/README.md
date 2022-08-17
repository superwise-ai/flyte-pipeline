# SKLearn Model Orchestration
Basic Flask App that demonstrates how to deploy sklearn model and log predictions into Superwise's platform

***

<p align="center" width="100%">
<b>🚧 THIS IS FOR DEMONSTRATION PURPOSE ONLY 🚧</b>
<p align="center" width="100%">
<b>🚧 DO NOT USE IT IN PRODUCTION 🚧</b>

***
# Usage

- override resources/envs.env file with relevant keys

- `python3 -m venv venv && pip install -r requirements.txt`


***

# Local Check
- Build image:
  ```
  gunicorn --bind 0.0.0.0:5050 predictor.server:app --timeout 100 -w 1
  ```

- Health check: 
  
  `curl -X GET http://localhost:5050/diamonds/v1/healthcheck`

- Prediction:
  
  `
  curl -X POST 'localhost:5050/diamonds/v1/predict' -H "Content-Type: application/json" -d '{"instances": [{"carat" : 1.42, "clarity" : "VVS1", "color" : "F", "cut" : "Ideal", "depth" : 60.8, "record_id" : 27671, "table" : 56, "x" : 7.25, "y" : 7.32, "z" : 4.43}]}'
  `

- update model:

  `
  curl -X POST 'localhost:5050/diamonds/v1/update-model' -H "Content-Type: application/json" -d '{"model_uri": "mlflow/1/04a18e09ffa64f9da0b818e1074b01bf/artifacts/validated_model/model.pkl"}'
  `

- trigger pipeline:

  `
  curl -X POST 'localhost:5050/diamonds/v1/trigger-pipeline'
  `