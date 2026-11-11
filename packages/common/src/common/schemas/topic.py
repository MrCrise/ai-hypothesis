from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field


class TaskStage(StrEnum):
    PLANNING = "planning"
    AWAITING_PLAN_APPROVAL = "awaiting_plan_approval"
    SEARCHING = "searching"
    FILTERING = "filtering"
    INGESTING = "ingesting"
    OVERVIEW = "overview"
    HYPOTHESES = "hypotheses"
    NOVELTY_CHECK = "novelty_check"
    RANKING = "ranking"
    COMPLETED = "completed"
    FAILED = "failed"


class TopicCreateRequest(BaseModel):
    query: str
    year_from: int | None = None
    year_to: int | None = None
    languages: list[str] = Field(default_factory=lambda: ["ru", "en"])
    publication_types: list[str] = Field(default_factory=list)


class SubQuestionPlan(BaseModel):
    sub_question: str
    search_queries: list[str]
    steel_grade_analogues: list[str] = Field(default_factory=list)


class SearchPlan(BaseModel):
    topic_id: UUID
    sub_questions: list[SubQuestionPlan]
    approved: bool = False


class TopicStatusResponse(BaseModel):
    topic_id: UUID
    query: str
    stage: TaskStage
    created_at: datetime
