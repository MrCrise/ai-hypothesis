# Design: Local Inference Subsystem and Unified Client

## Context

См. [proposal.md](proposal.md) для целей и предпосылок изменения.
Проект разрабатывается в рамках UV-монорепозитория (`packages/common`, `packages/ingestion`, `services/api`, `services/orchestrator`, `eval`).
Ключевые ограничения архитектуры:
- Стандарты проекта требуют изоляции зависимостей и запрета GPL/AGPL библиотек (K-13).
- Закрытый контур (Closed Contour): модели монтируются локально `./models:/models:ro`, доступ в интернет из Docker-сети `internal_net` отсутствует.
- Целевое аппаратное обеспечение разработчика: NVIDIA GeForce RTX 3080 Ti с 12 ГБ VRAM, многоядерный CPU и системная память RAM (32+ ГБ). На продакшн-сервере заказчика доступен полноразмерный GPU-кластер (CUDA).
- Требование запуска большой языковой модели семейства Qwen и специализированных моделей `Qwen3-Embedding` и `Qwen3-Reranker`. Для поддержки архитектуры `qwen3` необходим `transformers>=4.51.0` (в более ранних версиях возникает `KeyError: 'qwen3'`).

## Goals / Non-Goals

**Goals:**
- Обеспечить надежную работу LLM на 12 ГБ VRAM без ошибок CUDA OOM за счет выбора оптимального профиля модели: `Qwen3-14B` (Q4_K_M) или MoE `Qwen3-30B-A3B` (с опциональными переменными `LLAMA_ARG_CPU_MOE` / `LLAMA_ARG_N_CPU_MOE`) для локального dev-окружения, и плотных моделей 27B/32B для продакшн GPU.
- Развернуть сервер `llama.cpp` на внутреннем порту `8080` (образ `ghcr.io/ggml-org/llama.cpp:${LLAMA_CPP_IMAGE_TAG:-server-cuda}` с поддержкой пина билда `server-cuda-bXXXXX`), флагами `--flash-attn on`, `--jinja`, `--reasoning-format deepseek`, `--no-context-shift`, унифицированной переменной `LLM_GPU_LAYERS` и `start_period: 120s`.
- Развернуть микросервис `services/ml-inference` на внутреннем порту `8081` (`transformers>=4.51.0`):
  - `Qwen3-Embedding`: чистый 100% OpenAI-совместимый эндпоинт `/v1/embeddings`, `padding_side="left"`, last-token pooling, обязательная L2-нормализация и обрезка по MRL-размерности (`EMBEDDING_DIM`).
  - `Qwen3-Reranker`: загрузка строго через `AutoModelForCausalLM`, `padding_side="left"`, официальный шаблон промпта, раздельная токенизация `prefix_ids + body_ids (truncated) + suffix_ids` для гарантированного сохранения суффикса `<think>\n\n</think>\n\n`, расчет софтмакса по логитам токенов `convert_tokens_to_ids("yes")` и `convert_tokens_to_ids("no")`, строгий Cohere-формат с полем `relevance_score`, опциональной `instruction` и флагом `return_documents`.
  - Динамический микро-батчинг (`max_batch_size`, `max_wait_ms`, `max_length`) и общий `asyncio.Lock` между Embedding и Reranker при `ML_INFERENCE_DEVICE=cuda` для предотвращения CUDA OOM.
  - Поддержка CPU/CUDA через UV workspace паттерн: `[[tool.uv.index]]` и `[tool.uv] conflicts` в корневом `pyproject.toml`, `optional-dependencies` в сервисе, сборка Dockerfile через `uv sync --frozen --package ml-inference --extra ${TORCH_BACKEND} --no-dev`.
  - Структурированное оффлайн-хранилище: `./models/gguf/<file>.gguf` и `./models/hf/<model-dir>` через `snapshot_download(local_dir=...)` с `HF_HUB_OFFLINE=1` и `local_files_only=True`.
- Разработать `InferenceClient` в `packages/common`:
  - Приоритизация запросов `priority: Literal["high", "background"] = "background"` и семафор `asyncio.Semaphore(settings.llm_bg_max_concurrency)` (по умолчанию `max(1, llm_parallel_slots - 1)`) для гарантированного SLA интерактивного чата (<= 30с).
  - Обязательное клиентское разбиение больших списков текстов на батчи (`client_batch_size=32`) в `embed_documents()` для исключения таймаутов на 60с.
  - Формирование асимметричного префикса `Instruct: ...\nQuery: ...` на клиенте в `embed_queries()`.
  - Аудит-метаданные в `ChatResponse` (`content`, `reasoning_content`, токены, `latency_ms` для ФТ-26).
  - Управление рассуждениями (`enable_thinking`) и Structured Output (двухшаговый режим при рассуждениях).
  - LangChain-адаптеры `QwenChatModel` (с `with_structured_output`) и `QwenEmbeddings`.
- Изолировать зависимости: `packages/common` без PyTorch и Ragas; тесты интеграции Ragas запускаются в `eval`.

**Non-Goals:**
- Многоузловой распределенный инференс (vLLM Ray cluster).
- Использование коммерческих закрытых облачных API в рантайме.
- Полная реализация логики всех узлов LangGraph пайплайна (будет выполняться в последующих изменениях поверх созданного клиента).

## Decisions

### 1. Топология портов, образ и конфигурация llama.cpp server
- **Образ и порт:** Образ `ghcr.io/ggml-org/llama.cpp:${LLAMA_CPP_IMAGE_TAG:-server-cuda}` на внутреннем порту `8080`. Это предотвращает конфликт с бэкендом `services/api` (порт `8000`).
- **Healthcheck:** Добавляется `start_period: 120s` для предотвращения ложного падения контейнера во время прогрева и загрузки весов в память.
- **Профили моделей для dev и prod:**
  - **Dev (RTX 3080 Ti 12 GB):** Рекомендуется `Qwen3-14B` (квантизация Q4_K_M) либо MoE-архитектура `Qwen3-30B-A3B`. Для MoE запуск параметризуется опциональными env-переменными (например, `${LLAMA_ARG_CPU_MOE}` или `${LLAMA_ARG_N_CPU_MOE}`), чтобы флаг не ломал запуск плотных моделей в prod.
  - **Prod (GPU-сервер):** Плотная модель `Qwen3-27B` или `Qwen3-32B` с полным размещением слоев в VRAM.
- **Ключевые флаги запуска `llama-server`:**
  - `-m /models/gguf/${LLM_MODEL_FILE}`: путь к GGUF файлу.
  - `--host 0.0.0.0 --port 8080`.
  - `-c ${LLM_CTX_SIZE:-32768}`: размер контекстного окна.
  - `-ngl ${LLM_GPU_LAYERS:-28}`: число слоев для GPU offload (унифицированное имя переменной).
  - `--flash-attn on`: явное включение FlashAttention.
  - `--jinja`: нативная сборка промптов через Jinja.
  - `--reasoning-format deepseek`: выделение reasoning-контента в API.
  - `--no-context-shift`: отключение неявного сдвига контекста при длинных RAG-промптах.
  - `--cont-batching`: непрерывный батчинг.
  - `-ctk q8_0 -ctv q8_0`: 8-битное квантование KV-кэша.
  - `-np ${LLM_PARALLEL_SLOTS:-4}`: 4 параллельных слота генерации.

### 2. Специфика инференса Qwen3-Embedding и Qwen3-Reranker в services/ml-inference
- **Порт и библиотека:** `services/ml-inference` слушает внутренний порт `8081` и использует `transformers>=4.51.0`.
- **Токенизация:** Токенизаторы обеих моделей конфигурируются строго с `padding_side="left"`.
- **Qwen3-Embedding:**
  - **Эндпоинт:** 100% чистый OpenAI-совместимый `/v1/embeddings`, принимающий `input: list[str]`.
  - **Пулинг:** Last-token pooling по `attention_mask` (извлечение скрытого состояния последнего значащего токена).
  - **MRL (Matryoshka Representation Learning):** Если задан `EMBEDDING_DIM` (например, 1024 под `Vector(1024)` в `pgvector`), вектор обрезается: `embedding[:, :embedding_dim]`.
  - **Нормализация:** Обязательная L2-нормализация: `F.normalize(embedding, p=2, dim=-1)`.
  - **Клиентская асимметрия:** Асимметричный префикс `Instruct: {instruction}\nQuery: {query}` формируется на стороне `InferenceClient.embed_queries()`. Сервер обрабатывает любые тексты единообразно, что кардинально упрощает динамический микро-батчинг.
- **Qwen3-Reranker:**
  - **Тип модели:** Загружается строго как `AutoModelForCausalLM` (не `SequenceClassification`).
  - **Точный шаблон промпта:**
    - system: `Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".`
    - user: `<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}`
    - assistant suffix: `<think>\n\n</think>\n\n`
  - **Защита от обрезки суффикса:** Выполняется раздельная токенизация:
    1. `prefix_ids` = токенизированный системный промпт + `<Instruct>: ...\n<Query>: ...\n<Document>: `
    2. `suffix_ids` = токенизированный `<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`
    3. `body_ids` = токенизированный текст документа, усеченный до `max_length - len(prefix_ids) - len(suffix_ids)`
    4. Итоговые токены: `prefix_ids + body_ids + suffix_ids`. Это гарантирует, что суффикс `<think>\n\n</think>\n\n` никогда не отсекается на длинных документах.
  - **Расчет вероятности:** Индексы токенов определяются строго через `tokenizer.convert_tokens_to_ids("yes")` и `tokenizer.convert_tokens_to_ids("no")`. Калиброванная вероятность `relevance_score` вычисляется как:
    $$\text{relevance\_score} = \text{softmax}([\text{logit}_{\text{no}}, \text{logit}_{\text{yes}}])[1]$$
  - **Формат ответа:** Строгий Cohere-формат с полем `relevance_score`, поддержкой кастомной `instruction` (дефолт: металлургический поиск) и флагом `return_documents`.

### 3. UV Workspace паттерн для PyTorch CPU/CUDA и Closed Contour
- **Размещение настроек UV Workspace:**
  - Согласно спецификации `uv` workspace, секции `[[tool.uv.index]]` и `[tool.uv] conflicts` размещаются в **корневом `pyproject.toml`**, чтобы глобальный `uv.lock` резолвил зависимости рабочего пространства без конфликтов:
    ```toml
    [tool.uv]
    conflicts = [
      [
        { extra = "cpu", package = "ml-inference" },
        { extra = "cu124", package = "ml-inference" },
      ],
    ]

    [[tool.uv.index]]
    name = "pytorch-cpu"
    url = "https://download.pytorch.org/whl/cpu"
    explicit = true

    [[tool.uv.index]]
    name = "pytorch-cu124"
    url = "https://download.pytorch.org/whl/cu124"
    explicit = true
    ```
  - В `services/ml-inference/pyproject.toml` объявляются `optional-dependencies`:
    ```toml
    [project.optional-dependencies]
    cpu = ["torch>=2.5.0", "transformers>=4.51.0", "accelerate>=1.2.0"]
    cu124 = ["torch>=2.5.0", "transformers>=4.51.0", "accelerate>=1.2.0"]

    [tool.uv.sources]
    torch = [
      { index = "pytorch-cpu", extra = "cpu" },
      { index = "pytorch-cu124", extra = "cu124" },
    ]
    ```
  - Команда синхронизации для локальной разработки:
    `uv sync --package ml-inference --extra cpu`
  - В `services/ml-inference/Dockerfile` сборка использует команду:
    ```dockerfile
    ARG TORCH_BACKEND=cpu
    RUN uv sync --frozen --package ml-inference --extra ${TORCH_BACKEND} --no-dev
    ```
- **Структура директории `./models`:**
  - `./models/gguf/<model-file>.gguf`
  - `./models/hf/<model-name>/` (содержит `config.json`, `tokenizer.json`, `model.safetensors`).
  - Скачивание через `scripts/download_models.py` выполняется методом `snapshot_download(repo_id, local_dir=...)` без устаревшего параметра `local_dir_use_symlinks`.
  - В рантайме сервиса `ml-inference` задаются `HF_HUB_OFFLINE=1` и `local_files_only=True`.

### 4. Динамический батчинг, защита от CUDA OOM и приоритизация в InferenceClient
- **Лимиты микро-батчера:** В `services/ml-inference` настраиваются параметры `max_batch_size` (32), `max_wait_ms` (10) и `max_length` (8192 для эмбеддингов, 4096 для реранкера).
- **Защита от CUDA OOM:** При `ML_INFERENCE_DEVICE=cuda` между пайплайнами эмбеддингов и реранкера используется общий `asyncio.Lock()`, исключающий параллельное выделение тензоров на GPU.
- **Клиентское чанкирование:** В `InferenceClient.embed_documents()` списки документов автоматически разбиваются на клиентские чанки по `client_batch_size=32` для предотвращения превышения 60-секундного HTTP-таймаута при индексации сотен фрагментов.
- **Приоритизация запросов и защита SLA интерактивного чата (<= 30с):**
  - Запросы чата выполняются с `priority="high"`.
  - Фоновые задачи оркестратора выполняются с `priority="background"`.
  - Фоновые задачи ограничиваются асинхронным семафором:
    `bg_semaphore = asyncio.Semaphore(settings.llm_bg_max_concurrency)` (по умолчанию `max(1, settings.llm_parallel_slots - 1)`).
  - Высокоприоритетные запросы чата проходят без ожидания семафора, гарантируя доступность слота `llama.cpp` и соблюдение НФТ отклика <= 30 секунд.

### 5. Архитектура InferenceClient и адаптеров в packages/common
- **`InferenceClient`**:
  - `chat(messages, priority="background", ...)` -> `ChatResponse` (поля `content`, `reasoning_content`, `model`, `prompt_tokens`, `completion_tokens`, `latency_ms`).
  - `chat_stream(messages, priority="high", ...)` -> асинхронный генератор SSE.
  - `chat_structured(messages, response_model, priority="background", enable_thinking=True)`: двухшаговый сценарий при active thinking, исключающий конфликт GBNF с `<think>`.
  - `embed_documents(texts: list[str])` -> `list[list[float]]` (клиентское чанкирование по 32 элемента, отправка на `/v1/embeddings`).
  - `embed_queries(queries: list[str], instruction: str | None = None)` -> `list[list[float]]` (формирование префикса `Instruct: ...\nQuery: ...`).
  - `rerank(query: str, documents: list[str], instruction: str | None = None, return_documents: bool = False, top_n: int | None = None)` -> `list[RerankResult]`.
- **LangChain / RAGAS совместимость**:
  - `QwenChatModel(BaseChatModel)`: нативная реализация `with_structured_output`, флаг `enable_thinking=False`.
  - `QwenEmbeddings(Embeddings)`: разделение `aembed_documents` и `aembed_query`.
  - Интеграционные тесты с `ragas` запускаются исключительно в пакете `eval` через его dev-зависимости, не затрагивая `packages/common`.

## Risks / Trade-offs

- **[Risk]** Конфликт грамматик GBNF и генерации рассуждений `<think>...</think>`.
  - **Mitigation:** Двухшаговый вызов `chat_structured` при `enable_thinking=True` либо принудительное отключение рассуждений `enable_thinking=False` для узлов строгой фильтрации.
- **[Risk]** Монополизация слотов `llama-server` фоновыми задачами пайплайна гипотез.
  - **Mitigation:** Семафор `bg_semaphore = asyncio.Semaphore(settings.llm_bg_max_concurrency)` и приоритизация `priority="high"` для чата.
  - *Примечание:* `asyncio.Semaphore` действует в пределах одного процесса воркера; при масштабировании оркестратора на несколько реплик лимит делится на число реплик или переносится в распределенный Redis-семафор.
- **[Risk]** OOM на 12 GB VRAM при использовании тяжелых моделей.
  - **Mitigation:** Использование профиля `Qwen3-14B` Q4_K_M или `Qwen3-30B-A3B` (`--n-cpu-moe`) для dev-режима; общий `asyncio.Lock` между моделями при `ML_INFERENCE_DEVICE=cuda`.

## Migration Plan

1. Добавить `services/ml-inference` в корень `pyproject.toml` (`tool.uv.workspace.members`) и настроить корневые `[tool.uv.index]` / `[tool.uv] conflicts`.
2. Добавить конфигурационные параметры в `packages/common/src/common/config.py` и `.env.example` (порты 8080/8081, теги, `LLM_GPU_LAYERS`, `LLM_BG_MAX_CONCURRENCY`, `ML_BATCH_MAX_SIZE`, `ML_BATCH_MAX_WAIT_MS`).
3. Реализовать клиент `InferenceClient` (приоритизация, клиентское чанкирование, асимметрия, аудит) и LangChain-адаптеры в `packages/common/src/common/inference/`.
4. Реализовать сервис `services/ml-inference` на порту 8081 с чистым `/v1/embeddings`, CausalLM `/v1/rerank` (защита суффикса, общий GPU-lock) и Dockerfile (`uv sync --frozen --package ml-inference --extra ${TORCH_BACKEND} --no-dev`).
5. Обновить `docker-compose.yml`: настроить `llm-server` (порт 8080, `start_period: 120s`, `LLM_GPU_LAYERS`, опциональный `LLAMA_ARG_CPU_MOE`) и `ml-inference` (порт 8081).
6. Реализовать `scripts/download_models.py` с сохранением в `./models/gguf` и `./models/hf` без устаревших параметров symlinks.
7. Запустить проверки качества (`ruff`, `mypy`, `pytest`, `pip-licenses`).
