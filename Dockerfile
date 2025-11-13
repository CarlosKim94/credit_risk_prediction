FROM python:3.12.12-bookworm

RUN pip install uv

WORKDIR /app
COPY ["pyproject.toml", "uv.lock", "./"]

RUN uv sync

COPY ["predict.py", "./"]
COPY ["model/", "./model/"]

EXPOSE 9696

ENTRYPOINT ["/app/.venv/bin/uvicorn", "--host=0.0.0.0", "--port=9696", "predict:app"]