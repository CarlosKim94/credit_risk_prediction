FROM python:3.12.12-bookworm

RUN pip install uv

WORKDIR /app
COPY ["pyproject.toml", "uv.lock", "./"]

RUN uv sync

COPY ["predict.py", "model/model_depth_25_estimator_60_0.858.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["uvicorn", "--bind=0.0.0.0:9696", "predict:app"]