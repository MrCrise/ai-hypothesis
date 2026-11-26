"""Проверка текстового слоя PDF: решает, нужен ли OCR (см. design.md, решение 3)."""

from __future__ import annotations

import threading

import pypdfium2 as pdfium  # type: ignore[import-untyped]

# pdfium не потокобезопасен: вызовы из asyncio.to_thread сериализуются.
_PDFIUM_LOCK = threading.Lock()


def chars_per_page(pdf_bytes: bytes) -> list[int]:
    """Число непробельных символов текстового слоя на каждой странице."""
    with _PDFIUM_LOCK:
        return _count(pdf_bytes)


def _count(pdf_bytes: bytes) -> list[int]:
    pdf = pdfium.PdfDocument(pdf_bytes)
    try:
        counts = []
        for i in range(len(pdf)):
            page = pdf[i]
            textpage = page.get_textpage()
            text = textpage.get_text_range()
            counts.append(sum(not ch.isspace() for ch in text))
            textpage.close()
            page.close()
        return counts
    finally:
        pdf.close()


def needs_ocr(pdf_bytes: bytes, min_chars_per_page: int) -> tuple[bool, int]:
    """(нужен ли OCR, число страниц). OCR нужен, если хоть одна страница беднее порога."""
    counts = chars_per_page(pdf_bytes)
    return any(c < min_chars_per_page for c in counts), len(counts)
