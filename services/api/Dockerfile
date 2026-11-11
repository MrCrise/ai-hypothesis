# syntax=docker/dockerfile:1.7
FROM python:3.12-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock* ./
COPY packages/common/pyproject.toml ./packages/common/pyproject.toml
COPY packages/ingestion/pyproject.toml ./packages/ingestion/pyproject.toml
COPY services/api/pyproject.toml ./services/api/pyproject.toml
COPY services/orchestrator/pyproject.toml ./services/orchestrator/pyproject.toml
COPY eval/pyproject.toml ./eval/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-workspace --no-dev --package api
COPY packages ./packages
COPY services/api ./services/api
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --package api

FROM python:3.12-slim-bookworm AS runtime
WORKDIR /app
RUN groupadd -r appuser && useradd -r -g appuser appuser
COPY --from=builder --chown=appuser:appuser /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER appuser
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
