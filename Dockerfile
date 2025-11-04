# syntax=docker/dockerfile:1.6
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies required by Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for better security
RUN groupadd --system app \
    && useradd --system --create-home --gid app app

WORKDIR /app

# Install dependencies first to leverage Docker layer caching
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Copy application code
COPY . /app

# Ensure runtime directories exist and are writable
RUN mkdir -p data/docs data/memory \
    && chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD sh -c 'curl -fsS -H "X-Internal-Key: ${MICROSERVICE_INTERNAL_KEY}" http://127.0.0.1:8000/healthz || exit 1'

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
