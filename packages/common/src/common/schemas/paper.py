from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field


class AccessStatus(StrEnum):
    FULL_TEXT = "full_text"
    ABSTRACT_ONLY = "abstract_only"
    INTERNAL = "internal"


class PaperMetadata(BaseModel):
    paper_id: UUID
    doi: str | None = None
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    source: str
    abstract: str | None = None
    access_status: AccessStatus
    s3_key: str | None = None
    is_relevant: bool | None = None
    relevance_reason: str | None = None
    manual_override: bool = False


class FunnelStats(BaseModel):
    found_total: int
    after_dedup: int
    relevant: int
    with_full_text: int
