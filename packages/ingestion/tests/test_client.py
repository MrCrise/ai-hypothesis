"""Требования «OCR только для страниц без текстового слоя», «Поддержка DOCX»,
«Ошибки и устойчивость сервиса разбора»."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pypdfium2 as pdfium  # type: ignore[import-untyped]
import pytest
from docling_core.types.doc import DoclingDocument
from ingestion.parser import DoclingServeClient, ParseError, UnsupportedFormatError, parse_document
from ingestion.parser.text_layer import chars_per_page


def text_pdf(texts: list[str]) -> bytes:
    """Минимальный PDF: по странице на элемент, пустая строка — страница без текстового слоя."""
    objs: list[bytes] = []
    kids = []
    font_id = 3
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(b"")  # Pages — заполним после
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for t in texts:
        stream = f"BT /F1 12 Tf 50 700 Td ({t}) Tj ET".encode() if t else b""
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        content_id = len(objs)
        objs.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 800] /Contents %d 0 R "
            b"/Resources << /Font << /F1 %d 0 R >> >> >>" % (content_id, font_id)
        )
        kids.append(len(objs))
    objs[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (
        b" ".join(b"%d 0 R" % k for k in kids),
        len(kids),
    )
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


LONG = "Hydrogen induced cracking of line pipe steels in sour service environments."


def test_text_layer_counts() -> None:
    counts = chars_per_page(text_pdf([LONG, ""]))
    assert counts[0] > 50 and counts[1] == 0
    assert len(pdfium.PdfDocument(text_pdf([LONG]))) == 1


class FakeServe:
    """Имитация docling-serve v1 async: записывает запросы, отдаёт заданные ответы."""

    def __init__(
        self, doc: DoclingDocument, *, final: str = "success", fail_first: int = 0
    ) -> None:
        self.doc = doc
        self.final = final
        self.fail_first = fail_first
        self.requests: list[httpx.Request] = []
        self.polls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/v1/convert/file/async":
            if self.fail_first:
                self.fail_first -= 1
                return httpx.Response(503)
            return httpx.Response(200, json={"task_id": "t1", "task_status": "pending"})
        if path == "/v1/status/poll/t1":
            self.polls += 1
            status = "started" if self.polls < 2 else self.final
            return httpx.Response(200, json={"task_id": "t1", "task_status": status})
        if path == "/v1/result/t1":
            body: dict[str, Any] = {
                "document": {"json_content": self.doc.export_to_dict()},
                "status": "success",
                "processing_time": 1.5,
                "errors": [],
            }
            return httpx.Response(200, json=body)
        return httpx.Response(404)

    def form(self) -> bytes:
        return self.requests[0].content


def make_client(fake: FakeServe, **kw: Any) -> DoclingServeClient:
    return DoclingServeClient(
        "http://pdf-parser:5001", poll_interval=0, transport=httpx.MockTransport(fake), **kw
    )


async def test_pdf_with_text_layer_no_ocr(article: DoclingDocument) -> None:
    fake = FakeServe(article)
    doc = await parse_document(text_pdf([LONG, LONG]), "scan.pdf", client=make_client(fake))
    assert doc.meta.ocr_applied is False
    assert doc.meta.num_pages == 2 and doc.meta.processing_time == 1.5
    assert b'name="do_ocr"\r\n\r\nfalse' in fake.form()
    assert b'name="to_formats"\r\n\r\njson' in fake.form()
    assert doc.tables[0].caption.startswith("Table 4")


async def test_pdf_with_empty_page_uses_ocr(article: DoclingDocument) -> None:
    fake = FakeServe(article)
    doc = await parse_document(text_pdf([LONG, ""]), "mixed.pdf", client=make_client(fake))
    assert doc.meta.ocr_applied is True
    assert b'name="do_ocr"\r\n\r\ntrue' in fake.form()


async def test_docx_same_structure(article: DoclingDocument) -> None:
    fake = FakeServe(article)
    doc = await parse_document(b"PK\x03\x04fake-docx", "report.docx", client=make_client(fake))
    assert doc.meta.format == "docx" and doc.meta.ocr_applied is False
    assert b"wordprocessingml" in fake.form()
    # В DOCX нет эвристики «первый заголовок — название»: все заголовки остаются разделами.
    assert [s.title for s in doc.sections] == [
        "Reoxidation Behavior of Molten Steel in Tundish",
        "1. Introduction",
        "2. Experimental",
    ]


async def test_unsupported_format_no_request(article: DoclingDocument) -> None:
    fake = FakeServe(article)
    with pytest.raises(UnsupportedFormatError):
        await parse_document(b"MZ", "virus.exe", client=make_client(fake))
    assert fake.requests == []


async def test_failure_status_raises(article: DoclingDocument) -> None:
    fake = FakeServe(article, final="failure")
    with pytest.raises(ParseError) as exc:
        await parse_document(text_pdf([LONG]), "bad.pdf", client=make_client(fake))
    assert exc.value.filename == "bad.pdf" and "failure" in exc.value.reason


async def test_retry_on_503(article: DoclingDocument) -> None:
    fake = FakeServe(article, fail_first=2)
    doc = await parse_document(text_pdf([LONG]), "a.pdf", client=make_client(fake))
    assert doc.sections
    assert [r.url.path for r in fake.requests].count("/v1/convert/file/async") == 3


async def test_task_timeout(article: DoclingDocument) -> None:
    fake = FakeServe(article, final="started")  # задача никогда не завершается
    client = make_client(fake, task_timeout=0.05)
    client.poll_interval = 0.01
    with pytest.raises(ParseError) as exc:
        await parse_document(text_pdf([LONG]), "slow.pdf", client=client)
    assert "время ожидания" in exc.value.reason


async def test_broken_pdf_rejected_before_service(article: DoclingDocument) -> None:
    fake = FakeServe(article)
    with pytest.raises(ParseError):
        await parse_document(b"not a pdf", "broken.pdf", client=make_client(fake))
    assert fake.requests == []


def test_fake_serve_returns_json(article: DoclingDocument) -> None:
    resp = FakeServe(article)(httpx.Request("GET", "http://x/v1/result/t1"))
    assert json.loads(resp.content)["status"] == "success"


class RawServe(FakeServe):
    """Позволяет подменить ответ на конкретный путь."""

    def __init__(self, doc: DoclingDocument, overrides: dict[str, httpx.Response]) -> None:
        super().__init__(doc)
        self.overrides = overrides

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path in self.overrides:
            self.requests.append(request)
            return self.overrides[request.url.path]
        return super().__call__(request)


@pytest.mark.parametrize(
    ("path", "response"),
    [
        ("/v1/convert/file/async", httpx.Response(200, text="<html>proxy error</html>")),
        ("/v1/convert/file/async", httpx.Response(200, json={"task_status": "pending"})),
        ("/v1/result/t1", httpx.Response(200, json=["not", "a", "dict"])),
        (
            "/v1/result/t1",
            httpx.Response(200, json={"status": "success", "document": {"json_content": {"x": 1}}}),
        ),
    ],
    ids=["not-json", "no-task-id", "result-not-object", "invalid-json-content"],
)
async def test_unexpected_responses_raise_parse_error(
    article: DoclingDocument, path: str, response: httpx.Response
) -> None:
    fake = RawServe(article, {path: response})
    with pytest.raises(ParseError):
        await parse_document(text_pdf([LONG]), "odd.pdf", client=make_client(fake))


async def test_unknown_terminal_status_fails_fast(article: DoclingDocument) -> None:
    fake = RawServe(
        article,
        {
            "/v1/status/poll/t1": httpx.Response(
                200, json={"task_id": "t1", "task_status": "revoked"}
            )
        },
    )
    client = make_client(fake, task_timeout=30)
    with pytest.raises(ParseError) as exc:
        await parse_document(text_pdf([LONG]), "r.pdf", client=client)
    assert "revoked" in exc.value.reason


async def test_failure_includes_service_errors(article: DoclingDocument) -> None:
    fake_result = httpx.Response(
        200, json={"status": "failure", "errors": ["layout model crashed"]}
    )
    serve = RawServe(article, {"/v1/result/t1": fake_result})
    serve.final = "failure"
    with pytest.raises(ParseError) as exc:
        await parse_document(text_pdf([LONG]), "bad.pdf", client=make_client(serve))
    assert "layout model crashed" in exc.value.reason
