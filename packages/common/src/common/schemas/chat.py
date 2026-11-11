from uuid import UUID

from pydantic import BaseModel, Field

from common.schemas.hypothesis import SourceCitation


class ChatRequest(BaseModel):
    topic_id: UUID
    question: str


class ChatResponse(BaseModel):
    answer: str
    has_answer_in_corpus: bool
    citations: list[SourceCitation] = Field(default_factory=list)
