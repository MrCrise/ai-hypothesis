"""Требования «Структурированный результат разбора» и «Список литературы» (ФТ-11)."""

from common.schemas.parsed import BlockKind, DocumentFormat, ParsedDocument, ParseMeta
from docling_core.types.doc import DocItemLabel, DoclingDocument
from ingestion.parser import from_docling, from_docling_json

META = ParseMeta(filename="a.pdf", format=DocumentFormat.PDF)


def test_sections_blocks_pages(article: DoclingDocument) -> None:
    doc = from_docling(article, META)
    assert doc.title == "Reoxidation Behavior of Molten Steel in Tundish"
    assert [s.title for s in doc.sections] == ["1. Introduction", "2. Experimental"]
    exp = next(b for b in doc.blocks if b.text.startswith("Experiments"))
    assert exp.page == 2
    assert doc.section_title(exp.section_index) == "2. Experimental"
    assert doc.meta.num_pages == 3


def test_page_headers_dropped(article: DoclingDocument) -> None:
    doc = from_docling(article, META)
    assert all("ISIJ International" not in b.text for b in doc.blocks)


def test_chem_table_cells(article: DoclingDocument) -> None:
    doc = from_docling(article, META)
    assert len(doc.tables) == 1
    table = doc.tables[0]
    assert table.caption == "Table 4. Chemical compositions of steels in wt%."
    assert table.page == 2
    header = {c.text: c.col for c in table.cells if c.row == 0}
    row_4340 = next(c.row for c in table.cells if c.col == 0 and c.text == "4340")
    cell = table.cell(row_4340, header["C"])
    assert cell is not None and cell.text == "0.41"
    # Подпись таблицы не дублируется обычным блоком.
    assert all(not b.text.startswith("Table 4") for b in doc.blocks)


def test_figure_caption(article: DoclingDocument) -> None:
    doc = from_docling(article, META)
    assert [f.text for f in doc.figures] == ["Fig. 1. Schematic illustration of the tundish."]
    captions = [b for b in doc.blocks if b.kind is BlockKind.CAPTION]
    assert [c.text for c in captions] == ["Fig. 1. Schematic illustration of the tundish."]


def test_references_separate(article: DoclingDocument) -> None:
    doc = from_docling(article, META)
    assert doc.references == [
        "1) Y. Kakio et al.: Tetsu-to-Hagané, 62 (1976), 1803.",
        "2) H. Tanaka et al.: Tetsu-to-Hagané, 79 (1993), 1254.",
    ]
    assert all("Kakio" not in b.text for b in doc.blocks)
    assert "REFERENCES" not in [s.title for s in doc.sections]


def test_russian_references_header() -> None:
    d = DoclingDocument(name="ru")
    d.add_heading("Отчёт о плавке")
    d.add_heading("1. Результаты")
    d.add_text(DocItemLabel.TEXT, "Текст.")
    d.add_heading("Список литературы")
    d.add_list_item("1. Иванов И. И. Сталь. 2020.")
    doc = from_docling(d, META)
    assert doc.references == ["1. Иванов И. И. Сталь. 2020."]


def test_json_roundtrip(article: DoclingDocument) -> None:
    doc = from_docling_json(article.export_to_dict(), META)
    again = ParsedDocument.model_validate(doc.model_dump(mode="json"))
    assert again == doc


def test_docx_first_heading_is_section() -> None:
    d = DoclingDocument(name="docx")
    d.add_heading("Introduction")
    d.add_text(DocItemLabel.TEXT, "Para one.")
    d.add_heading("Methods")
    d.add_text(DocItemLabel.TEXT, "Para two.")
    doc = from_docling(d, ParseMeta(filename="r.docx", format=DocumentFormat.DOCX))
    assert doc.title is None
    assert [s.title for s in doc.sections] == ["Introduction", "Methods"]
    assert doc.section_title(doc.blocks[0].section_index) == "Introduction"


def test_pdf_numbered_first_heading_is_section() -> None:
    d = DoclingDocument(name="pdf")
    d.add_heading("1. Introduction")
    d.add_text(DocItemLabel.TEXT, "Para one.")
    doc = from_docling(d, META)
    assert doc.title is None and [s.title for s in doc.sections] == ["1. Introduction"]


def test_free_standing_figure_caption_counted() -> None:
    d = DoclingDocument(name="pdf")
    d.add_heading("Title of paper")
    d.add_heading("2. Results")
    d.add_text(DocItemLabel.CAPTION, "Fig. 5. Inclusion size distribution.")
    d.add_text(DocItemLabel.CAPTION, "Рис. 2. Схема разливки.")
    d.add_text(DocItemLabel.CAPTION, "Note: values in wt%.")
    doc = from_docling(d, META)
    assert [f.text for f in doc.figures] == [
        "Fig. 5. Inclusion size distribution.",
        "Рис. 2. Схема разливки.",
    ]
    assert len([b for b in doc.blocks if b.kind is BlockKind.CAPTION]) == 3
