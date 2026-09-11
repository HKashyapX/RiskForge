FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
COPY src/ /app/src/
COPY config/ /app/config/

WORKDIR /app
RUN pip install --no-cache-dir --prefix=/install .


FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY --from=builder /app/src /app/src
COPY --from=builder /app/config /app/config

RUN groupadd -r riskforge && useradd -r -g riskforge -d /app -s /sbin/nologin riskforge

WORKDIR /app
RUN chown -R riskforge:riskforge /app

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

USER riskforge

CMD ["uvicorn", "riskforge.runtime.production:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
