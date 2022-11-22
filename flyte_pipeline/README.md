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
- `python3 -m venv venv && pip install -r core/requirements.txt`

- update `.aws_credentials` file ( AWS secrets )

- Create docker image repository and build image as following:

```
docker build --tag=<YOUR_REPOSITORY>:$(date "+%s") --build-arg SUPERWISE_CLIENT_ID=$SUPERWISE_CLIENT_ID --build-arg SUPERWISE_SECRET=$SUPERWISE_SECRET  --build-arg TAG='basic_pipeline' -f Dockerfile .
```

- `docker push <YOUR_REPOSITORY>:<tag>`

***

# Usage:

- set `LOCAL_IP` - you can run `ip addr | grep en0` in order to retrieve it

- `flytectl demo start` - do spin-up k3s cluster within a docker container

-
run `pyflyte run --remote --image <YOUR_REPOSITORY>:<tag> core/basic_pipeline.py ml_pipeline --diamonds_price_threshold=10000` in order to execute the workflow