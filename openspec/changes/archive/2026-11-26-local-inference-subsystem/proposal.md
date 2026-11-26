# Proposal: Local Inference Subsystem and Unified Client

## Why

Система анализа научной литературы и генерации металлургических гипотез должна работать в строго изолированном контуре (без внешних облачных LLM API) в соответствии со стандартами проекта (K-13, Closed Contour). Для обеспечения высокопроизводительного и надежного инференса на локальном оборудовании (GPU NVIDIA RTX 3080 Ti 12 GB + CPU хоста в dev-режиме, многопроцессорный GPU-сервер в prod) требуется развернуть оптимизированный стек инференса: сервер `llama.cpp` для больших языковых моделей Qwen и микросервис `services/ml-inference` на FastAPI/PyTorch (`transformers>=4.51.0`) для моделей `Qwen3-Embedding` и `Qwen3-Reranker`, а также предоставить надежный асинхронный клиент `InferenceClient` с приоритизацией запросов (SLA чата <= 30с), клиентским батчингом, аудитом и адаптерами под LangGraph и RAGAS.

## What Changes

- **llm-server**: Переход в `docker-compose.yml` на актуальный образ `ghcr.io/ggml-org/llama.cpp:${LLAMA_CPP_IMAGE_TAG:-server-cuda}` с возможностью точного пина билда (`server-cuda-bXXXXX` с поддержкой Qwen3), внутренним портом `8080`, `start_period: 120s` в healthcheck, флагами `--flash-attn on`, `--jinja`, `--reasoning-format deepseek`, `--no-context-shift`, continuous batching, квантованием KV-кэша (`q8_0`) и унифицированной переменной `LLM_GPU_LAYERS`.
  - Для dev-окружения на 12 GB VRAM предусмотрена конфигурация под `Qwen3-14B` (Q4_K_M) или MoE `Qwen3-30B-A3B` с опциональной env-переменной `LLAMA_ARG_CPU_MOE` / `LLAMA_ARG_N_CPU_MOE`, не ломающей запуск плотных моделей (27B/32B) в prod.
- **services/ml-inference**: Новый микросервис на FastAPI и PyTorch (`transformers>=4.51.0`, порт `8081`):
  - 100% чистый OpenAI-совместимый эндпоинт `/v1/embeddings` для `Qwen3-Embedding` с `padding_side="left"`, last-token pooling, обязательной L2-нормализацией и обрезкой по MRL-размерности (`EMBEDDING_DIM`, например 1024 под `pgvector`).
  - Cohere-совместимый эндпоинт `/v1/rerank` (поле `relevance_score`, опциональные `instruction` с металлургическим дефолтом и `return_documents: bool = False`) на базе `Qwen3-Reranker` (`AutoModelForCausalLM`, `padding_side="left"`).
  - Официальный промпт Qwen3-Reranker с раздельной токенизацией `prefix_ids + body_ids (truncated) + suffix_ids`, гарантирующей неразрывность суффикса `<think>\n\n</think>\n\n` и расчет вероятности через `softmax` логитов `convert_tokens_to_ids("yes")` и `convert_tokens_to_ids("no")`.
  - Конфигурируемые лимиты микро-батчера (`max_batch_size`, `max_wait_ms`, `max_length`) и общий `asyncio.Lock` между Embedding и Reranker при `ML_INFERENCE_DEVICE=cuda` для предотвращения CUDA OOM.
  - Управление зависимостями через UV: `[[tool.uv.index]]` (`pytorch-cpu`, `pytorch-cu124`) и `[tool.uv] conflicts` в корневом `pyproject.toml`, `optional-dependencies` в сервисе, синхронизация через `uv sync --package ml-inference --extra cpu` (в Dockerfile: `uv sync --frozen --package ml-inference --extra ${TORCH_BACKEND} --no-dev`).
  - Структурированное хранилище моделей: `./models/gguf/<file>.gguf` и `./models/hf/<model-dir>` через `snapshot_download(local_dir=...)` с гарантированным оффлайн-режимом `HF_HUB_OFFLINE=1` и `local_files_only=True`.
- **packages/common (InferenceClient & Adapters)**:
  - Асинхронный клиент `InferenceClient` с приоритизацией `priority: Literal["high", "background"] = "background"` и семафором `asyncio.Semaphore(llm_bg_max_concurrency)` (по умолчанию `max(1, llm_parallel_slots - 1)`) для защиты SLA чата (<= 30с).
  - Обязательное клиентское разбиение больших списков текстов на чанки (`client_batch_size=32`) в `embed_documents()` для исключения таймаутов на 60с.
  - Формирование асимметричного префикса `Instruct: ...\nQuery: ...` на стороне клиента в `embed_queries()`.
  - Метаданные аудита в `ChatResponse` (`content`, `reasoning_content`, токены, `latency_ms` для ФТ-26).
  - Поддержка `enable_thinking` и двухшагового structured output при рассуждениях.
  - Адаптеры LangChain (`QwenChatModel`, `QwenEmbeddings`) с поддержкой `with_structured_output` и изоляцией тестов Ragas в пакете `eval`.
- **scripts/download_models.py**: Скрипт скачивания моделей в строгую структуру `./models/gguf` и `./models/hf` без устаревших параметров `local_dir_use_symlinks`.
- **Интеграция в конфигурацию**: Обновление `Settings` в `packages/common/src/common/config.py` и `.env.example`.

## Capabilities

### New Capabilities
- `local-inference`: Локальный инференс LLM (llama.cpp порт 8080) и микросервис `services/ml-inference` (порт 8081, transformers>=4.51.0) для Qwen3-Embedding и Qwen3-Reranker (CausalLM с защитой суффикса и общим GPU-lock) с чистым OpenAI/Cohere API, детерминированным UV-стеком (CPU/CUDA) и строгим Closed Contour.
- `inference-client`: Унифицированный асинхронный Python-клиент с приоритизацией запросов (SLA чата <= 30с), клиентским чанкированием батчей, асимметрией эмбеддингов, структурированным выводом и адаптерами для LangGraph и RAGAS.

### Modified Capabilities
*Нет существующих спецификаций в openspec/specs/*

## Impact

- `docker-compose.yml`: Сервисы `llm-server` (порт 8080, `ghcr.io/ggml-org/llama.cpp:${LLAMA_CPP_IMAGE_TAG:-server-cuda}`, `start_period: 120s`, `LLM_GPU_LAYERS`, опциональный `LLAMA_ARG_CPU_MOE`) и `ml-inference` (порт 8081, build-arg `TORCH_BACKEND`).
- `pyproject.toml` (корень): Размещение `[[tool.uv.index]]` и `[tool.uv] conflicts`.
- `services/ml-inference`: Новый сервис с чистым `/v1/embeddings` (last-token pooling, L2, MRL) и `/v1/rerank` (`AutoModelForCausalLM`, раздельная токенизация префикса/тела/суффикса, `relevance_score`, `instruction`, `return_documents`).
- `packages/common`: `InferenceClient` с приоритизацией, клиентским батчингом, асимметрией запросов и аудитом, без тяжелых зависимостей PyTorch и Ragas.
- `.env.example`: Параметры портов (8080, 8081), тегов, переменной `LLM_GPU_LAYERS`, `LLM_BG_MAX_CONCURRENCY`, `ML_BATCH_MAX_SIZE`, `ML_BATCH_MAX_WAIT_MS` и профилей моделей.
