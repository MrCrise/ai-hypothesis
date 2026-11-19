# Metallurgy R&D AI Assistant

Интеллектуальный ассистент для поиска научной литературы, анализа релевантности и автоматизированной генерации проверяемых гипотез в металлургии.

## Структура репозитория

- `packages/` — библиотеки рабочего пространства (UV workspace):
  - `common/` — модели базы данных SQLAlchemy (`pgvector`), схемы Pydantic, конфигурация (`pydantic-settings`), клиенты S3/LLM.
  - `ingestion/` — парсинг PDF (клиент Docling), коннекторы академических источников, дедупликация и индексация.
- `services/` — микросервисы системы:
  - `api/` — FastAPI сервис (эндпоинты `/health`, `/api/v1/topics`, `/corpus`, `/chat`, `/hypotheses`, `/reports`).
  - `orchestrator/` — фоновый воркер многостадийного пайплайна генерации гипотез на базе LangGraph.
  - `egress-gateway/` — Squid proxy шлюз с белым списком доменов для доступа к внешним академическим API.
- `eval/` — инструменты оценки качества поиска и генерации (Ragas, SciPy).
- `openspec/` — артефакты Spec-Driven Development (`specs/`, `changes/`, `config.yaml`).
- `.agents/` — база знаний для агентов:
  - `rules/project-standards.md` — стандарты архитектуры, лицензий, безопасности и разработки.
  - `skills/` — сценарии и навыки для агентов (включая SDD-аудит `sdd-audit`).

## Команды разработки

Все команды выполняются через `uv` из корня проекта:

```bash
# Синхронизация зависимостей рабочего пространства
uv sync

# Добавление зависимости в конкретный пакет/сервис
uv add <package_name> --package <common|ingestion|api|orchestrator|eval>

# Запуск тестов
uv run pytest

# Линтинг, форматирование и типизация
uv run ruff check .
uv run ruff format . --check    # Проверка стиля (uv run ruff format . для автоформатирования)
uv run mypy packages services

# Проверка лицензий зависимостей на соответствие K-13
uv run pip-licenses --summary
uv run pip-licenses --fail-on="GPL;AGPL"

# Управление локальным окружением в Docker
docker compose up --build -d                  # Базовые сервисы (postgres, seaweedfs, api, orchestrator, egress-gateway)
docker compose --profile ml up --build -d     # Запуск с ML-сервисами (docling/pdf-parser, llm-server)
docker compose down
```

## Стандарты и SDD-разработка

- Технические стандарты, архитектурные ограничения и правила работы с кодом: [.agents/rules/project-standards.md](.agents/rules/project-standards.md).
- Любые изменения и новые функции реализуются строго по SDD-протоколу через спецификации в директории [openspec/](openspec/).
