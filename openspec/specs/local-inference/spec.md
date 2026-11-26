# local-inference Specification

## Purpose

Предоставляет автономную инфраструктуру локального инференса языковых моделей (LLM) и моделей векторных представлений (Embeddings и Reranker) без обращения к внешним облачным сервисам.

## Requirements

### Requirement: Local LLM Server Execution
Система SHALL запускать HTTP-сервер llama.cpp из репозитория ggml-org на внутреннем порту 8080 с поддержкой FlashAttention, Jinja-шаблонов, формата reasoning, переменной LLM_GPU_LAYERS и флага no-context-shift.

#### Scenario: Successful Chat Completion
- **WHEN** клиент отправляет валидный POST-запрос на `http://llm-server:8080/v1/chat/completions`
- **THEN** сервер возвращает сгенерированный ответ в стандартном формате OpenAI Chat Completion с кодом 200

#### Scenario: Streaming Chat Completion
- **WHEN** клиент запрашивает генерацию с флагом `stream: true`
- **THEN** сервер передает токены в реальном времени через Server-Sent Events (SSE) до завершающего маркера `[DONE]`

#### Scenario: Long Context RAG Request
- **WHEN** клиент отправляет длинный контекст статьи в пределах лимита окна контекста
- **THEN** флаг `--no-context-shift` гарантирует отсутствие неявного сдвига контекста и отбрасывания начальных данных

### Requirement: Clean OpenAI Embeddings Inference Service
Микросервис инференса на порту 8081 SHALL предоставлять чистый OpenAI-совместимый эндпоинт `/v1/embeddings` для Qwen3-Embedding с last-token pooling, обязательной L2-нормализацией, MRL-обрезкой размерности и лимитами микро-батчера.

#### Scenario: Batch Text Embeddings
- **WHEN** поступает запрос со списком текстов на `/v1/embeddings`
- **THEN** сервис токенизирует батч с `padding_side="left"`, извлекает вектор последнего значащего токена, обрезает по `EMBEDDING_DIM` и применяет L2-нормализацию

#### Scenario: Dynamic Batching Under Concurrency
- **WHEN** несколько параллельных запросов поступают одновременно в течение окна ожидания микро-батчинга `max_wait_ms`
- **THEN** сервис объединяет их в единый проход модели до `max_batch_size` без блокировки event loop и возвращает корректные эмбеддинги

### Requirement: CausalLM Reranker Inference Service
Микросервис инференса на порту 8081 SHALL предоставлять эндпоинт `/v1/rerank` на базе Qwen3-Reranker как CausalLM с раздельной токенизацией суффикса, токенами yes/no, опциональной инструкцией и флагом return_documents.

#### Scenario: Document Reranking with Strict Cohere Format
- **WHEN** поступает POST-запрос на `/v1/rerank` с запросом, документами, опциональной `instruction` и `return_documents`
- **THEN** сервис вычисляет `relevance_score` через softmax логитов токенов `yes`/`no` и возвращает результат в Cohere-формате

#### Scenario: Suffix Protection on Long Documents
- **WHEN** длина документа превышает лимит контекста модели
- **THEN** сервис обрезает только тело документа `body_ids`, гарантируя полное сохранение токенов суффикса `<think>\n\n</think>\n\n`

### Requirement: Closed Contour, Storage Structure, and Concurrency Lock
Микросервис инференса SHALL загружать веса моделей из каталогов ./models/gguf и ./models/hf в режиме offline и использовать общий lock для предотвращения CUDA OOM.

#### Scenario: Offline Initialization from Structured Models Directory
- **WHEN** сервис инициализирует модели из локально смонтированных каталогов `/models/hf`
- **THEN** загрузка выполняется с параметрами `HF_HUB_OFFLINE=1` и `local_files_only=True` без обращения во внешнюю сеть

#### Scenario: CUDA Concurrency Lock
- **WHEN** сервис запущен на GPU (`ML_INFERENCE_DEVICE=cuda`) и запросы на эмбеддинги и реранкинг приходят одновременно
- **THEN** общий `asyncio.Lock` сериализует выполнение прямого прохода на GPU, исключая ошибку CUDA Out of Memory

### Requirement: Service Health and Readiness Probes
Серверы инференса SHALL предоставлять эндпоинты `/health` для проверки доступности и готовности с периодом прогрева start_period 120s.

#### Scenario: Health Check When Ready
- **WHEN** сервис завершил прогрев моделей в течение `start_period: 120s`
- **THEN** GET-запрос на `/health` возвращает статус 200 OK со статусом готовности
