"""Клиент docling-serve (API v1, асинхронный режим) и точка входа `parse_document` (ФТ-11, ФТ-13)."""

from __future__ import annotations

import asyncio
import time
from pathlib import PurePath
from typing import Any

import httpx
from common.config import settings
from common.schemas.parsed import DocumentFormat, ParsedDocument, ParseMeta
from pydantic import ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from ingestion.parser.docling_adapter import from_docling_json
from ingestion.parser.text_layer import needs_ocr

_MIME: dict[DocumentFormat, str] = {
    DocumentFormat.PDF: "application/pdf",
    DocumentFormat.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
_PENDING = frozenset({"pending", "started"})
_EXT: dict[str, DocumentFormat] = {".pdf": DocumentFormat.PDF, ".docx": DocumentFormat.DOCX}


class ParseError(RuntimeError):
    def __init__(self, filename: str, reason: str) -> None:
        super().__init__(f"{filename}: {reason}")
        self.filename = filename
        self.reason = reason


class UnsupportedFormatError(ParseError):
    pass


def detect_format(filename: str) -> DocumentFormat:
    fmt = _EXT.get(PurePath(filename).suffix.lower())
    if fmt is None:
        raise UnsupportedFormatError(filename, "неподдерживаемый формат (нужен PDF или DOCX)")
    return fmt


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500


class DoclingServeClient:
    """`POST /v1/convert/file/async` → `GET /v1/status/poll/{id}` → `GET /v1/result/{id}`."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        http_timeout: float | None = None,
        task_timeout: float | None = None,
        poll_interval: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        attempts: int = 3,
    ) -> None:
        self.base_url = (base_url or settings.pdf_parser_url).rstrip("/")
        self.http_timeout = (
            http_timeout if http_timeout is not None else settings.parser_http_timeout
        )
        self.task_timeout = (
            task_timeout if task_timeout is not None else settings.parser_task_timeout
        )
        self.poll_interval = (
            poll_interval if poll_interval is not None else settings.parser_poll_interval
        )
        self.transport = transport
        self.attempts = attempts

    def _client(self) -> httpx.AsyncClient:
        # Сервис во внутренней сети: прокси egress-gateway не нужен.
        return httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.http_timeout,
            transport=self.transport,
            trust_env=False,
        )

    async def _request(self, client: httpx.AsyncClient, method: str, url: str, **kw: Any) -> Any:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.attempts),
            wait=wait_exponential(multiplier=0.5, max=8),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        ):
            with attempt:
                resp = await client.request(method, url, **kw)
                resp.raise_for_status()
                return resp.json()
        raise AssertionError("unreachable")  # pragma: no cover

    async def _failure_details(self, client: httpx.AsyncClient, task_id: str) -> str:
        """Пробует забрать ошибки упавшей задачи из /v1/result (формат при failure не гарантирован)."""
        try:
            resp = await client.get(f"/v1/result/{task_id}")
            body = resp.json()
        except (httpx.HTTPError, ValueError):
            return ""
        errors = body.get("errors") if isinstance(body, dict) else None
        return f": {errors}" if errors else ""

    async def convert(
        self, data: bytes, filename: str, fmt: DocumentFormat, *, do_ocr: bool
    ) -> dict[str, Any]:
        form = {
            "to_formats": "json",
            "do_ocr": str(do_ocr).lower(),
            "image_export_mode": "placeholder",
        }
        files = {"files": (filename, data, _MIME[fmt])}
        deadline = time.monotonic() + self.task_timeout
        async with self._client() as client:
            try:
                task = await self._request(
                    client, "POST", "/v1/convert/file/async", data=form, files=files
                )
                task_id = task["task_id"]
                status = task.get("task_status")
                while status in _PENDING:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ParseError(
                            filename, f"превышено время ожидания {self.task_timeout} с"
                        )
                    await asyncio.sleep(min(self.poll_interval, remaining))
                    task = await self._request(client, "GET", f"/v1/status/poll/{task_id}")
                    status = task.get("task_status")
                if status != "success":
                    raise ParseError(
                        filename,
                        f"docling-serve: задача {task_id} завершилась со статусом {status}"
                        f"{await self._failure_details(client, task_id)}",
                    )
                result = await self._request(client, "GET", f"/v1/result/{task_id}")
            except httpx.HTTPError as exc:
                raise ParseError(filename, f"ошибка обращения к docling-serve: {exc}") from exc
            except (ValueError, KeyError, TypeError) as exc:  # не-JSON или не тот формат ответа
                raise ParseError(filename, f"неожиданный ответ docling-serve: {exc!r}") from exc
        if not isinstance(result, dict):
            raise ParseError(filename, "неожиданный ответ docling-serve: не объект JSON")
        if result.get("status") not in ("success", "partial_success"):
            raise ParseError(
                filename, f"docling-serve: {result.get('status')}: {result.get('errors')}"
            )
        return result


async def parse_document(
    data: bytes, filename: str, *, client: DoclingServeClient | None = None
) -> ParsedDocument:
    """Разбирает PDF/DOCX в ParsedDocument. OCR — только если в PDF есть страница без текста."""
    fmt = detect_format(filename)
    client = client or DoclingServeClient()
    do_ocr, num_pages = False, None
    if fmt is DocumentFormat.PDF:
        try:
            do_ocr, num_pages = await asyncio.to_thread(
                needs_ocr, data, settings.parser_min_chars_per_page
            )
        except Exception as exc:  # повреждённый PDF — не отправляем в сервис
            raise ParseError(filename, f"не удалось прочитать PDF: {exc}") from exc
    result = await client.convert(data, filename, fmt, do_ocr=do_ocr)
    json_content = (result.get("document") or {}).get("json_content")
    if not json_content:
        raise ParseError(filename, "docling-serve не вернул json_content")
    meta = ParseMeta(
        filename=filename,
        format=fmt,
        num_pages=num_pages,
        ocr_applied=do_ocr,
        processing_time=result.get("processing_time"),
    )
    try:
        return from_docling_json(json_content, meta)
    except ValidationError as exc:
        raise ParseError(filename, f"json_content не соответствует DoclingDocument: {exc}") from exc
