# syntax=docker/dockerfile:1

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*

# Copy project (relies on .dockerignore to trim context)
COPY . .

# Install deps (pyproject preferred, else requirements.txt)
RUN bash -c 'if [ -f pyproject.toml ]; then pip install --upgrade pip && pip install .; \
    elif [ -f requirements.txt ]; then pip install -r requirements.txt; \
    else echo "WARNING: no pyproject.toml or requirements.txt found"; fi'

# Build tiny sample index (idempotent)
RUN python -m src.ingestion.build_kb

ENV INDEX_DIR=models/index/kb-faiss-bge \
    RETRIEVAL_DEVICE=gpu \
    PORT_API=8000 \
    PORT_UI=8501

# Default = API (compose overrides for UI)
CMD ["bash","-c","uvicorn src.interface.api:app --host 0.0.0.0 --port ${PORT_API}"]