"""Требование «Нарезка на чанки» (ФТ-16)."""

from common.schemas.parsed import (
    ChunkKind,
    DocumentFormat,
    ParsedDocument,
    ParseMeta,
    Section,
    TextBlock,
)
from docling_core.types.doc import DoclingDocument
from ingestion.parser import chunk_document, from_docling

META = ParseMeta(filename="a.pdf", format=DocumentFormat.PDF)


def test_table_is_own_chunk(article: DoclingDocument) -> None:
    chunks = chunk_document(from_docling(article, META), max_chars=1500)
    tables = [c for c in chunks if c.kind is ChunkKind.TABLE]
    assert len(tables) == 1
    t = tables[0]
    assert t.content.startswith("Table 4. Chemical compositions of steels in wt%.")
    assert "0.41" in t.content
    assert t.section == "2. Experimental" and t.page == 2
    texts = [c for c in chunks if c.kind is ChunkKind.TEXT and c.section == "2. Experimental"]
    assert texts and all("Table 4" not in c.content for c in texts)


def test_references_not_chunked(article: DoclingDocument) -> None:
    chunks = chunk_document(from_docling(article, META))
    assert all("Kakio" not in c.content for c in chunks)


def _doc(blocks: list[tuple[int, str]]) -> ParsedDocument:
    return ParsedDocument(
        sections=[Section(index=0, title="A"), Section(index=1, title="B")],
        blocks=[TextBlock(text=t, section_index=s, page=1) for s, t in blocks],
        meta=META,
    )


def test_chunk_respects_section_boundary() -> None:
    chunks = chunk_document(_doc([(0, "a" * 10), (1, "b" * 10)]), max_chars=1000)
    assert [c.section for c in chunks] == ["A", "B"]


def test_long_section_split_within_limit() -> None:
    chunks = chunk_document(_doc([(0, "x" * 400)] * 5), max_chars=1000)
    assert len(chunks) == 3
    assert all(len(c.content) <= 1000 for c in chunks)
    assert "\n\n".join(c.content for c in chunks).count("x") == 2000


def test_oversized_block_kept_whole() -> None:
    chunks = chunk_document(_doc([(0, "y" * 50), (0, "z" * 3000)]), max_chars=1000)
    assert [len(c.content) for c in chunks] == [50, 3000]


def test_empty_table_skipped() -> None:
    from common.schemas.parsed import Table

    doc = _doc([(0, "text")])
    doc.tables.append(Table(n_rows=0, n_cols=0))
    chunks = chunk_document(doc)
    assert all(c.content for c in chunks) and len(chunks) == 1
