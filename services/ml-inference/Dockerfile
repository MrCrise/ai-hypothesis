# syntax=docker/dockerfile:1.7
FROM python:3.12-slim-bookworm AS builder

ARG TORCH_BACKEND=cpu
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

COPY pyproject.toml uv.lock* ./
COPY packages/common/pyproject.toml ./packages/common/pyproject.toml
COPY packages/ingestion/pyproject.toml ./packages/ingestion/pyproject.toml
COPY services/api/pyproject.toml ./services/api/pyproject.toml
COPY services/orchestrator/pyproject.toml ./services/orchestrator/pyproject.toml
COPY services/ml-inference/pyproject.toml ./services/ml-inference/pyproject.toml
COPY eval/pyproject.toml ./eval/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-workspace --no-dev --package ml-inference --extra ${TORCH_BACKEND}

COPY packages/common ./packages/common
COPY services/ml-inference ./services/ml-inference

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package ml-inference --extra ${TORCH_BACKEND}

FROM python:3.12-slim-bookworm AS runtime
WORKDIR /app
RUN groupadd -r appuser && useradd -r -g appuser appuser
COPY --from=builder --chown=appuser:appuser /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
USER appuser
EXPOSE 8081
CMD ["uvicorn", "ml_inference.main:app", "--host", "0.0.0.0", "--port", "8081"]
