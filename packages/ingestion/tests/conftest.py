"""Синтетические документы для тестов парсера (без чужих статей в репозитории)."""

from __future__ import annotations

import pytest
from docling_core.types.doc import (
    BoundingBox,
    ContentLayer,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)


def prov(page: int) -> ProvenanceItem:
    return ProvenanceItem(page_no=page, bbox=BoundingBox(l=0, t=0, r=10, b=10), charspan=(0, 1))


CHEM_ROWS = [
    ["Alloy", "C", "Si", "Mn"],
    ["4340", "0.41", "0.23", "0.79"],
    ["300 M", "0.4", "1.6", "0.7"],
]


def chem_table() -> TableData:
    cells = [
        TableCell(
            text=text,
            start_row_offset_idx=r,
            end_row_offset_idx=r + 1,
            start_col_offset_idx=c,
            end_col_offset_idx=c + 1,
            column_header=r == 0,
        )
        for r, row in enumerate(CHEM_ROWS)
        for c, text in enumerate(row)
    ]
    return TableData(table_cells=cells, num_rows=len(CHEM_ROWS), num_cols=len(CHEM_ROWS[0]))


def build_article() -> DoclingDocument:
    doc = DoclingDocument(name="article")
    for p in (1, 2, 3):
        doc.add_page(page_no=p, size=Size(width=600, height=800))
        doc.add_text(
            DocItemLabel.PAGE_HEADER,
            "ISIJ International, Vol. 40 (2000)",
            prov=prov(p),
            content_layer=ContentLayer.FURNITURE,
        )
    doc.add_heading("Reoxidation Behavior of Molten Steel in Tundish", prov=prov(1))
    doc.add_heading("1. Introduction", prov=prov(1))
    doc.add_text(DocItemLabel.TEXT, "Clean steel demand is increasing.", prov=prov(1))
    doc.add_heading("2. Experimental", prov=prov(2))
    doc.add_text(DocItemLabel.TEXT, "Experiments were conducted on a 270 t ladle.", prov=prov(2))
    caption = doc.add_text(
        DocItemLabel.CAPTION, "Table 4. Chemical compositions of steels in wt%.", prov=prov(2)
    )
    doc.add_table(data=chem_table(), caption=caption, prov=prov(2))
    fig_caption = doc.add_text(
        DocItemLabel.CAPTION, "Fig. 1. Schematic illustration of the tundish.", prov=prov(2)
    )
    doc.add_picture(caption=fig_caption, prov=prov(2))
    doc.add_text(DocItemLabel.TEXT, "The oxidation by air accounts for 66 %.", prov=prov(3))
    doc.add_heading("REFERENCES", prov=prov(3))
    doc.add_list_item("1) Y. Kakio et al.: Tetsu-to-Hagané, 62 (1976), 1803.", prov=prov(3))
    doc.add_list_item("2) H. Tanaka et al.: Tetsu-to-Hagané, 79 (1993), 1254.", prov=prov(3))
    return doc


@pytest.fixture
def article() -> DoclingDocument:
    return build_article()
