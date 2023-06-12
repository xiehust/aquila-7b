FROM ghcr.io/allenai/pytorch:1.13.1-cuda11.7-python3.10-v1.2.2

RUN apt update
RUN pip install bminf flagai fastapi uvicorn

RUN mkdir -p /opt/ml/code

COPY . /opt/ml/code
COPY serve /opt/ml/code

RUN mkdir -p /opt/ml/code/checkpoints_in

WORKDIR /opt/ml/code
