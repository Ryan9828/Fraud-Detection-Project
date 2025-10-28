FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code AND artifacts into the image
COPY service_raw.py ./service_raw.py
COPY artifacts_fraud_detec/ ./artifacts_fraud_detec/

# (If you have other modules/files they must be copied too)
# COPY utils.py ./utils.py
# COPY some_config.json ./some_config.json

ENV ART_DIR=artifacts_fraud_detec \
    PORT=8000 \
    HOST=0.0.0.0

EXPOSE 8000

# Healthcheck can hit 127.0.0.1 inside the container
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s CMD \
  curl -fsS http://127.0.0.1:${PORT}/health || exit 1

CMD ["uvicorn", "service_raw:app", "--host", "0.0.0.0", "--port", "8000"]
