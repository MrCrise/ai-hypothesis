"""ParsedDocument → чанки для `chunks` (ФТ-16): текст по разделам, таблица — отдельный чанк."""

from __future__ import annotations

from common.config import settings
from common.schemas.parsed import ChunkKind, ParsedChunk, ParsedDocument, TextBlock


def _flush(doc: ParsedDocument, buf: list[TextBlock], out: list[ParsedChunk]) -> None:
    if buf:
        out.append(
            ParsedChunk(
                section=doc.section_title(buf[0].section_index),
                page=buf[0].page,
                content="\n\n".join(b.text for b in buf),
            )
        )
        buf.clear()


def chunk_document(doc: ParsedDocument, max_chars: int | None = None) -> list[ParsedChunk]:
    """Блоки раздела набираются до `max_chars`, чанк не пересекает границу раздела.
    Блок длиннее лимита становится отдельным чанком целиком (не режется посреди фразы).
    Литература в чанки не попадает."""
    limit = max_chars or settings.chunk_max_chars
    out: list[ParsedChunk] = []
    buf: list[TextBlock] = []
    size = 0
    for block in doc.blocks:
        sep = 2 if buf else 0
        if buf and (
            block.section_index != buf[0].section_index or size + sep + len(block.text) > limit
        ):
            _flush(doc, buf, out)
            size, sep = 0, 0
        buf.append(block)
        size += sep + len(block.text)
    _flush(doc, buf, out)

    for table in doc.tables:
        body = table.markdown or "\n".join(c.text for c in table.cells)
        content = f"{table.caption}\n\n{body}".strip() if table.caption else body.strip()
        if not content:
            continue
        out.append(
            ParsedChunk(
                section=doc.section_title(table.section_index),
                page=table.page,
                content=content,
                kind=ChunkKind.TABLE,
            )
        )
    return out
