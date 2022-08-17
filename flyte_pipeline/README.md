# SKLearn Model Orchestration
Basic Flask App that demonstrates how to deploy sklearn model and log predictions into Superwise's platform

***

<p align="center" width="100%">
<b>🚧 THIS IS FOR DEMONSTRATION PURPOSE ONLY 🚧</b>
<p align="center" width="100%">
<b>🚧 DO NOT USE IT IN PRODUCTION 🚧</b>

***
# Prerequisites:


- `brew install flyteorg/homebrew-tap/flytectl` (Assure docker is installed and running)

- `pip install flytekit`

- `pip install mlflow`

- update `.env` file ( SUPERWISE and AWS secrets)

- update `.aws_credentials` file ( AWS secrets )

- Create docker image repository and build image as following:

  `
  docker build --tag=<YOUR_REPOSITORY>/basic_pipeline_img:$(date "+%s") --build-arg MODEL_PATH='<PATH TO DEFAULT MODEL>' --build-arg SUPERWISE_CLIENT_ID=$SUPERWISE_CLIENT_ID --build-arg SUPERWISE_SECRET=$SUPERWISE_SECRET --build-arg TAG='basic_pipeline' -f Dockerfile .
 `

- `docker push <YOUR_REPOSITORY>/basic_pipeline_img`


***
# Usage:

- `flytectl demo start` - do spin-up k3s cluster within a docker container
  
- run `pyflyte run --remote --image <YOUR_REPOSITORY>/basic_pipeline_img:lastest core/basic_pipeline.py ml_pipeline --diamonds_price_threshold=10000` - execute the workflow