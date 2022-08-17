# Deploy MLFlow locally

This is an example how to deploy MLFlow locally, using MySQL for metadata store and s3 as a MODEL Registry.

All using simple docker compose.

Taken from: https://towardsdatascience.com/deploy-mlflow-with-docker-compose-8059f16b6039

***

<p align="center" width="100%">
<b>🚧 THIS IS FOR DEMONSTRATION PURPOSE ONLY 🚧</b>
<p align="center" width="100%">
<b>🚧 DO NOT USE IT IN PRODUCTION 🚧</b>

***
# Usage

- override envs.env file with relevant keys

***

# Local Check

`
docker-compose --env-file envs.env -f ./docker-compose.yaml up -d --build
`