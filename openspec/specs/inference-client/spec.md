# inference-client Specification

## Purpose

Предоставляет асинхронный унифицированный клиент и адаптеры для вызова локальных сервисов инференса из пайплайнов LangGraph и фреймворка оценки Ragas.

## Requirements

### Requirement: Unified Async Inference Client
Клиентская библиотека SHALL предоставлять единый асинхронный интерфейс `InferenceClient` для взаимодействия с сервисами генерации текста (порт 8080), эмбеддингов и реранкера (порт 8081).

#### Scenario: Generate Chat Response with Audit Metadata
- **WHEN** вызывается асинхронный метод `chat()`
- **THEN** клиент возвращает `ChatResponse` содержащий `content`, `reasoning_content`, `model`, `prompt_tokens`, `completion_tokens` и `latency_ms`

#### Scenario: Asymmetric Embeddings with Client-Side Chunking
- **WHEN** вызывается `embed_documents()` с большим списком текстов или `embed_queries()` с поисковыми запросами
- **THEN** клиент разбивает документы на батчи по 32 элемента для исключения таймаутов, а для запросов формирует асимметричный префикс `Instruct: ...\nQuery: ...` перед отправкой на чистый `/v1/embeddings`

#### Scenario: Document Reranking with Strict Cohere Format
- **WHEN** вызывается асинхронный метод `rerank()` с опциональной инструкцией и флагом `return_documents`
- **THEN** клиент возвращает результаты с полями `index`, `relevance_score` и исходными документами при запросе

### Requirement: Request Prioritization and SLA Protection
Клиент SHALL поддерживать приоритизацию запросов и семафор фоновой нагрузки llm_bg_max_concurrency для защиты времени отклика интерактивного чата (SLA <= 30с).

#### Scenario: High Priority Chat Request Execution
- **WHEN** поступает запрос чата с `priority="high"` на фоне выполняющихся задач оркестратора с `priority="background"`
- **THEN** клиент резервирует слоты генерации и обрабатывает запрос чата без ожидания фонового семафора, гарантируя время ответа не более 30с

### Requirement: Structured Output and Reasoning Control
Клиент SHALL поддерживать управление рассуждениями `enable_thinking` и валидированный структурированный вывод с двухшаговым режимом при активных рассуждениях.

#### Scenario: Structured Output Generation
- **WHEN** клиент вызывает генерацию со структурированной JSON-схемой (например, Pydantic-схема 10-польных гипотез)
- **THEN** клиент передает `response_format` серверу и валидирует полученный JSON в целевую структуру данных

#### Scenario: Structured Output with Thinking Enabled
- **WHEN** запрашивается структурированный вывод при `enable_thinking=True`
- **THEN** клиент выполняет двухшаговый вызов (шаг 1: генерация рассуждений и первичного ответа, шаг 2: извлечение чистого JSON), исключая конфликт грамматик GBNF с тегами рассуждений

### Requirement: Resilient HTTP Execution and Retries
Клиент SHALL обрабатывать временные сбои сети и сервисов инференса с использованием экспоненциального бэкоффа и раздельных таймаутов.

#### Scenario: Retry On Transient Network Failure
- **WHEN** запрос к сервису инференса завершается сетевым сбоем или статусом 503 Service Unavailable
- **THEN** клиент выполняет повторные попытки с экспоненциальной задержкой до исчерпания лимита попыток

#### Scenario: Timeout On Hanging Generation
- **WHEN** время ответа сервера превышает сконфигурированный таймаут (300с для LLM, 60с для Embeddings)
- **THEN** клиент прерывает соединение и выбрасывает специализированное исключение таймаута

### Requirement: Streaming Response Iteration
Клиент SHALL поддерживать асинхронную потоковую передачу токенов генерации для снижения задержки отклика интерфейса.

#### Scenario: Stream Generation Chunks
- **WHEN** вызывается метод `chat_stream()`
- **THEN** клиент возвращает асинхронный генератор, отдающий текстовые фрагменты по мере их поступления от сервера

### Requirement: LangGraph and LangChain Compatibility
Клиент SHALL предоставлять обертки `QwenChatModel` и `QwenEmbeddings` из `langchain-core` с поддержкой `with_structured_output` и отключения рассуждений.

#### Scenario: Structured Output in LangGraph
- **WHEN** узел графа оркестратора вызывает `chat_model.with_structured_output(Schema)` с `enable_thinking=False`
- **THEN** адаптер выполняет вызов и возвращает типизированный объект Pydantic без конфликтов с грамматикой

#### Scenario: Embeddings Integration
- **WHEN** оркестратор вызывает `aembed_documents()` и `aembed_query()`
- **THEN** адаптер делегирует вызовы в соответствующие методы `InferenceClient`

### Requirement: Ragas Evaluation Compatibility
Клиент SHALL поддерживать интеграцию с метриками оценки RAGAS через LangChain-обертки с изоляцией зависимостей внутри пакета eval.

#### Scenario: Evaluation Run in Eval Package
- **WHEN** адаптер инференса передается в оцениватель RAGAS внутри тестов пакета `eval`
- **THEN** метрики вычисляются с использованием локальных моделей без добавления `ragas` в зависимости `packages/common`
