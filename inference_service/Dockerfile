FROM python:3.7


WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . ./

ARG MODEL_PATH
ARG SUPERWISE_CLIENT_ID
ARG SUPERWISE_SECRET
ARG SUPERWISE_MODEL_ID
ARG SUPERWISE_VERSION_ID
ARG BUCKET_NAME

ENV SUPERWISE_CLIENT_ID=${SUPERWISE_CLIENT_ID}
ENV SUPERWISE_SECRET=${SUPERWISE_SECRET}
ENV BUCKET_NAME=${BUCKET_NAME}
ENV MODEL_PATH=${MODEL_PATH}
ENV SUPERWISE_MODEL_ID=${SUPERWISE_MODEL_ID}
ENV SUPERWISE_VERSION_ID=${SUPERWISE_VERSION_ID}
ENV FLASK_APP /app/server.py

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:5050", "predictor.server:app", "--timeout", "1000", "-w", "4"]

EXPOSE 5050
