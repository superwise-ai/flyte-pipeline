FROM python:3.7-slim

RUN apt-get update && apt-get upgrade -y && apt-get install -y git

WORKDIR /root
# Virtual environment
ENV VENV /opt/venv
RUN python3 -m venv ${VENV}
ENV PATH="${VENV}/bin:$PATH"

# Install Python dependencies
COPY core/requirements.txt /root
RUN ${VENV}/bin/pip install -r /root/requirements.txt


ARG MODEL_PATH
ARG SUPERWISE_CLIENT_ID
ARG SUPERWISE_SECRET

ENV SUPERWISE_CLIENT_ID=${SUPERWISE_CLIENT_ID}
ENV SUPERWISE_SECRET=${SUPERWISE_SECRET}

COPY in_container.mk /root/Makefile
COPY core/sandbox.config .
RUN mkdir /root/.aws/
COPY .aws_credentials /root/.aws/credentials

# Copy the actual code
COPY core/ core/
ENV PYTHONPATH="/root"

# This tag is supplied by the build script and will be used to determine the version
# when registering tasks, workflows, and launch plans
ARG TAG
ENV FLYTE_INTERNAL_IMAGE $TAG

