"""DoclingDocument (ответ docling-serve, `to_formats=json`) → ParsedDocument (ФТ-11)."""

from __future__ import annotations

import re
from typing import Any

from common.schemas.parsed import (
    BlockKind,
    DocumentFormat,
    FigureCaption,
    ParsedDocument,
    ParseMeta,
    Section,
    Table,
    TableCell,
    TextBlock,
)
from docling_core.types.doc import (
    DocItem,
    DocItemLabel,
    DoclingDocument,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)

REFERENCES_HEADER = re.compile(
    r"^\s*(\d+[.)]?\s*)?(references?|literature( cited)?|bibliography|"
    r"список литературы|литература|библиографи\w*|источники)\s*$",
    re.IGNORECASE,
)

_SECTION_LIKE = re.compile(
    r"^\s*(\d+|[IVX]+)[.)\s]|^\s*(abstract|introduction|аннотация|введение)\b", re.IGNORECASE
)

_FIGURE_CAPTION = re.compile(r"^\s*(fig(ure|\.)?|рис(унок|\.)?)\s*\d+", re.IGNORECASE)

_BLOCK_KINDS: dict[DocItemLabel, BlockKind] = {
    DocItemLabel.TEXT: BlockKind.TEXT,
    DocItemLabel.PARAGRAPH: BlockKind.TEXT,
    DocItemLabel.LIST_ITEM: BlockKind.LIST_ITEM,
    DocItemLabel.CAPTION: BlockKind.CAPTION,
    DocItemLabel.FORMULA: BlockKind.FORMULA,
    DocItemLabel.FOOTNOTE: BlockKind.FOOTNOTE,
    DocItemLabel.CODE: BlockKind.TEXT,
}


def _page(item: DocItem) -> int | None:
    return item.prov[0].page_no if item.prov else None


def _table(item: TableItem, doc: DoclingDocument, section: int | None) -> Table:
    data = item.data
    cells = [
        TableCell(
            row=c.start_row_offset_idx,
            col=c.start_col_offset_idx,
            text=c.text,
            row_span=c.row_span,
            col_span=c.col_span,
            is_header=c.column_header or c.row_header,
        )
        for c in data.table_cells
    ]
    return Table(
        caption=item.caption_text(doc),
        section_index=section,
        page=_page(item),
        n_rows=data.num_rows,
        n_cols=data.num_cols,
        cells=cells,
        markdown=item.export_to_markdown(doc),
    )


def from_docling(doc: DoclingDocument, meta: ParseMeta) -> ParsedDocument:
    """Обходит документ в порядке чтения. Колонтитулы (слой furniture) Docling в обход не отдаёт."""
    result = ParsedDocument(meta=meta)
    if meta.num_pages is None and doc.pages:
        result.meta.num_pages = len(doc.pages)

    # Подписи таблиц и рисунков приходят и отдельными элементами — отмечаем, чтобы не дублировать.
    table_captions: set[str] = set()
    figure_captions: set[str] = set()
    for tb in doc.tables:
        table_captions.update(ref.cref for ref in tb.captions)
    for pic in doc.pictures:
        figure_captions.update(ref.cref for ref in pic.captions)

    section: int | None = None
    in_references = False
    for item, _level in doc.iterate_items():
        if isinstance(item, TitleItem):
            result.title = result.title or item.text.strip()
            continue
        if isinstance(item, SectionHeaderItem):
            # Docling часто размечает название статьи (PDF, стр. 1) как первый заголовок, а не title.
            # Для DOCX и для заголовков вида «1. Introduction» это не так — там это раздел.
            if (
                meta.format is DocumentFormat.PDF
                and result.title is None
                and not result.sections
                and not result.blocks
                and _page(item) in (None, 1)
                and not _SECTION_LIKE.match(item.text)
            ):
                result.title = item.text.strip()
                continue
            in_references = bool(REFERENCES_HEADER.match(item.text))
            if in_references:
                continue
            section = len(result.sections)
            result.sections.append(
                Section(index=section, title=item.text.strip(), level=item.level)
            )
            continue
        if isinstance(item, TableItem):
            result.tables.append(_table(item, doc, section))
            continue
        if isinstance(item, PictureItem):
            caption = item.caption_text(doc).strip()
            if caption:
                result.figures.append(
                    FigureCaption(text=caption, section_index=section, page=_page(item))
                )
                result.blocks.append(
                    TextBlock(
                        text=caption,
                        kind=BlockKind.CAPTION,
                        section_index=section,
                        page=_page(item),
                    )
                )
            continue
        if not isinstance(item, TextItem):
            continue
        text = item.text.strip()
        if not text or item.self_ref in table_captions or item.self_ref in figure_captions:
            continue
        if in_references or item.label == DocItemLabel.REFERENCE:
            result.references.append(text)
            continue
        kind = _BLOCK_KINDS.get(item.label)
        if kind is None:
            continue
        # Docling не всегда привязывает подпись к картинке: такая подпись приходит отдельным
        # caption-блоком. Если это подпись рисунка, учитываем её и в figures.
        if kind is BlockKind.CAPTION and _FIGURE_CAPTION.match(text):
            result.figures.append(FigureCaption(text=text, section_index=section, page=_page(item)))
        result.blocks.append(
            TextBlock(text=text, kind=kind, section_index=section, page=_page(item))
        )
    return result


def from_docling_json(json_content: dict[str, Any], meta: ParseMeta) -> ParsedDocument:
    return from_docling(DoclingDocument.model_validate(json_content), meta)
