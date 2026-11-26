# Tasks: Local Inference Subsystem and Unified Client

## 1. Конфигурация и рабочее пространство UV

- [x] 1.1 Добавить `services/ml-inference` в `tool.uv.workspace.members` в корневом `pyproject.toml`, настроить корневые `[[tool.uv.index]]` (`pytorch-cpu`, `pytorch-cu124`) и `[tool.uv] conflicts`, создать `services/ml-inference/pyproject.toml` с `optional-dependencies` (`cpu`, `cu124` с `transformers>=4.51.0`, `torch>=2.5.0`, `accelerate>=1.2.0`). Верифицировать успешной синхронизацией: `uv sync --package ml-inference --extra cpu`.
- [x] 1.2 Расширить `packages/common/src/common/config.py` и `.env.example` параметрами инференса (`llm_base_url` на порт 8080, `ml_inference_base_url` на порт 8081, `ml_inference_device`, `embedding_model_name`, `embedding_dim`, `reranker_model_name`, `llama_cpp_image_tag`, `llm_ctx_size`, `llm_gpu_layers`, `llm_parallel_slots`, `llm_bg_max_concurrency`, `ml_batch_max_size`, `ml_batch_max_wait_ms`, `torch_num_threads`). Документировать профили моделей для dev (12 GB VRAM: Qwen3-14B / MoE Qwen3-30B-A3B c `LLAMA_ARG_CPU_MOE`) и prod (27B/32B). Верифицировать тестом валидации `Settings`.
- [x] 1.3 Добавить в `packages/common/pyproject.toml` легковесные зависимости `httpx>=0.28.0`, `tenacity>=9.0.0`, `langchain-core>=0.3.0`. Верифицировать проверкой лицензий K-13 для runtime-зависимостей пакета: `uv run pip-licenses --packages common --fail-on="GPL;AGPL"`.

## 2. Асинхронный InferenceClient и адаптеры LangChain/RAGAS

- [x] 2.1 Реализовать класс `InferenceClient` в `packages/common/src/common/inference/client.py`:
  - Приоритизация `priority: Literal["high", "background"] = "background"` с семафором `asyncio.Semaphore(llm_bg_max_concurrency)` (по умолчанию `max(1, llm_parallel_slots - 1)`) для защиты SLA чата (<= 30с).
  - Метод `chat` с возвратом `ChatResponse` (`content`, `reasoning_content`, токены, `latency_ms` для журнала ФТ-26) и `chat_stream`.
  - Метод `chat_structured` с поддержкой `enable_thinking` и автоматическим двухшаговым режимом во избежание конфликта GBNF с `<think>`.
  - Раздельные методы `embed_documents` (с обязательным клиентским чанкированием списков по `client_batch_size=32` для исключения таймаутов на 60с) и `embed_queries` (формирование префикса `Instruct: ...\nQuery: ...` на клиенте) с отправкой на чистый `/v1/embeddings`.
  - Метод `rerank` с параметрами `instruction` (с металлургическим дефолтом), `return_documents: bool = False` и возвратом поля `relevance_score`.
  - Верифицировать модульными тестами с моками HTTP-ответов (`pytest packages/common/tests`).
- [x] 2.2 Реализовать LangChain-совместимый чат-адаптер `QwenChatModel(BaseChatModel)` с поддержкой `with_structured_output`, `_agenerate`, `_astream` и явным флагом `enable_thinking=False` для работы в графах и Ragas. Верифицировать асинхронными тестами генерации и структурированного вывода.
- [x] 2.3 Реализовать LangChain-совместимый адаптер эмбеддингов `QwenEmbeddings(Embeddings)` с разделением вызовов `aembed_documents` (с клиентским чанкированием) и `aembed_query` (с инструкцией). Верифицировать тестом генерации векторов.
- [x] 2.4 Экспортировать клиент и адаптеры в `packages/common/src/common/inference/__init__.py`. Реализовать интеграционный тест совместимости с Ragas (`LangchainLLMWrapper`, `LangchainEmbeddingsWrapper`) в workspace-пакете `eval` (через dev-зависимости), без добавления `ragas` в runtime-зависимости `packages/common`. Верифицировать запуском `pytest eval/tests`.

## 3. Микросервис инференса services/ml-inference

- [x] 3.1 Реализовать модуль загрузки моделей и инференса в `services/ml-inference/src/ml_inference/engine.py` (`transformers>=4.51.0`):
  - `Qwen3-Embedding`: токенизация с `padding_side="left"`, last-token pooling, обрезка по `EMBEDDING_DIM`, обязательная L2-нормализация.
  - `Qwen3-Reranker`: загрузка строго через `AutoModelForCausalLM` с `padding_side="left"`, официальный шаблон промпта, раздельная токенизация `prefix_ids + body_ids (truncated) + suffix_ids` для гарантированного сохранения суффикса `<think>\n\n</think>\n\n`, расчет калиброванной вероятности `relevance_score` через `softmax` логитов токенов `yes`/`no`.
  - Общий `asyncio.Lock` между Embedding и Reranker при `ML_INFERENCE_DEVICE=cuda` для исключения ошибок CUDA OOM.
  - Загрузка из каталогов `./models/hf/<model-name>` с параметрами `HF_HUB_OFFLINE=1` и `local_files_only=True`.
  - Поддержка параметров `ML_INFERENCE_DEVICE` (`cpu`/`cuda`) и лимита потоков `torch.set_num_threads`.
  - Верифицировать тестом вычисления эмбеддингов и реранкинга на тестовых тензорах.
- [x] 3.2 Реализовать механизм динамического микро-батчинга (`BatchingQueueWorker`) через `asyncio.Queue` и `asyncio.to_thread` в `services/ml-inference/src/ml_inference/batcher.py` с конфигурируемыми лимитами `max_batch_size`, `max_wait_ms`, `max_length`. Верифицировать тестом конкурентного объединения запросов.
- [x] 3.3 Реализовать FastAPI приложение в `services/ml-inference/src/ml_inference/main.py` на порту 8081 с эндпоинтами `/health`, чистым `/v1/embeddings` (OpenAI format) и `/v1/rerank` (строгий Cohere format с полями `relevance_score`, `instruction`, `return_documents`). Верифицировать тестами FastAPI с `httpx.AsyncClient` и `ASGITransport`.
- [x] 3.4 Создать `services/ml-inference/Dockerfile` с `ARG TORCH_BACKEND=cpu`, установкой через `uv sync --frozen --package ml-inference --extra ${TORCH_BACKEND} --no-dev`, multi-stage сборкой и non-root пользователем `appuser` для работы в `internal_net`. Верифицировать валидацией синтаксиса Dockerfile.

## 4. Контейнеризация, docker-compose и скрипты подготовки весов

- [x] 4.1 Создать CLI-скрипт `scripts/download_models.py` на базе `huggingface_hub` для скачивания GGUF (в `./models/gguf/<file>.gguf`) и весов `Qwen3-Embedding` и `Qwen3-Reranker` (в `./models/hf/<model-dir>` через `snapshot_download(local_dir=...)` без устаревших параметров `local_dir_use_symlinks`). Верифицировать тестом запуска скрипта с ключом `--help` и проверкой структуры целевых директорий.
- [x] 4.2 Обновить `docker-compose.yml`:
  - `llm-server`: образ `ghcr.io/ggml-org/llama.cpp:${LLAMA_CPP_IMAGE_TAG:-server-cuda}`, порт `8080`, healthcheck с `start_period: 120s`, флаги `--flash-attn on`, `--jinja`, `--reasoning-format deepseek`, `--no-context-shift`, `-ctk q8_0`, `-ctv q8_0`, `-ngl ${LLM_GPU_LAYERS:-28}`, опциональный `${LLAMA_ARG_CPU_MOE:-}`, путь `-m /models/gguf/${LLM_MODEL_FILE}`.
  - `ml-inference`: порт `8081`, build-arg `TORCH_BACKEND`, переменная `ML_INFERENCE_DEVICE`, монтирование `./models:/models:ro`.
  - Верифицировать командой `docker compose config`.

## 5. Сквозная верификация и соответствие стандартам (SDD Quality Gate)

- [x] 5.1 Запустить полный набор тестов проекта: `uv run pytest`. Верифицировать прохождение всех существующих и новых тестов.
- [x] 5.2 Выполнить проверку линтера и форматирования: `uv run ruff check .` и `uv run ruff format . --check`.
- [x] 5.3 Выполнить статическую проверку типов: `uv run mypy packages services`.
- [x] 5.4 Выполнить лицензионный аудит runtime-зависимостей: `uv run pip-licenses --packages common api orchestrator ml-inference --fail-on="GPL;AGPL"`.

## Workflow follow-up

- Выполнить команду `openspec validate local-inference-subsystem --strict` для подтверждения целостности артефактов.
- После подтверждения пользователем запустить этап применения: `openspec apply local-inference-subsystem`.
