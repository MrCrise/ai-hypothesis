"""Результат разбора документа (ФТ-11, ФТ-13) и чанки для индекса (ФТ-16).

Схема не зависит от парсера: потребители не видят типов Docling. Сохраняется в
`PaperModel.parsed_structure` через `model_dump(mode="json")`.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class DocumentFormat(StrEnum):
    PDF = "pdf"
    DOCX = "docx"


class Section(BaseModel):
    """Раздел документа. `index` — порядковый номер, на него ссылаются блоки и таблицы."""

    index: int
    title: str
    level: int = 1


class BlockKind(StrEnum):
    TEXT = "text"
    LIST_ITEM = "list_item"
    CAPTION = "caption"
    FORMULA = "formula"
    FOOTNOTE = "footnote"


class TextBlock(BaseModel):
    text: str
    kind: BlockKind = BlockKind.TEXT
    section_index: int | None = None
    page: int | None = None


class TableCell(BaseModel):
    row: int
    col: int
    text: str
    row_span: int = 1
    col_span: int = 1
    is_header: bool = False


class Table(BaseModel):
    caption: str = ""
    section_index: int | None = None
    page: int | None = None
    n_rows: int
    n_cols: int
    cells: list[TableCell] = Field(default_factory=list)
    markdown: str = ""

    def cell(self, row: int, col: int) -> TableCell | None:
        return next((c for c in self.cells if c.row == row and c.col == col), None)


class FigureCaption(BaseModel):
    text: str
    section_index: int | None = None
    page: int | None = None


class ParseMeta(BaseModel):
    filename: str
    format: DocumentFormat
    num_pages: int | None = None
    ocr_applied: bool = False
    parser: str = "docling-serve"
    processing_time: float | None = None


class ParsedDocument(BaseModel):
    title: str | None = None
    sections: list[Section] = Field(default_factory=list)
    blocks: list[TextBlock] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    figures: list[FigureCaption] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    meta: ParseMeta

    def section_title(self, index: int | None) -> str:
        if index is None:
            return ""
        return next((s.title for s in self.sections if s.index == index), "")


class ChunkKind(StrEnum):
    TEXT = "text"
    TABLE = "table"


class ParsedChunk(BaseModel):
    """Чанк до вычисления эмбеддинга; поля совпадают с `ChunkModel` (section, page, content)."""

    section: str
    page: int | None = None
    content: str
    kind: ChunkKind = ChunkKind.TEXT
