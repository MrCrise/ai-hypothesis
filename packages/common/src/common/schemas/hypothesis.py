from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field


class HypothesisType(StrEnum):
    PROCESS_PARAMETER = "process_parameter"
    CHEMICAL_COMPOSITION = "chemical_composition"
    THERMAL_TREATMENT = "thermal_treatment"
    KNOWN_MEASURES_COMBO = "known_measures_combo"
    CROSS_DOMAIN_TRANSFER = "cross_domain_transfer"
    CONTRADICTION_RESOLUTION = "contradiction_resolution"
    NEW_MECHANISM = "new_mechanism"


class NoveltyVerdict(StrEnum):
    PUBLISHED = "published"
    PARTIALLY_STUDIED = "partially_studied"
    NO_ANALOGUES = "no_analogues"


class SourceCitation(BaseModel):
    paper_id: UUID
    title: str
    section_or_page: str
    quote: str


class SupportingClaim(BaseModel):
    claim: str
    citation: SourceCitation


class NearestWork(BaseModel):
    title: str
    doi_or_patent_id: str | None = None
    difference_explanation: str


class HypothesisSchema(BaseModel):
    hypothesis_id: UUID
    topic_id: UUID
    statement: str
    hypothesis_type: HypothesisType
    mechanism: str
    supporting_claims: list[SupportingClaim] = Field(min_length=2, max_length=5)
    addressed_gap_id: str
    verification_method: str
    verifiability_score: int = Field(ge=1, le=5)
    verifiability_requirements: str
    novelty_verdict: NoveltyVerdict
    nearest_works: list[NearestWork] = Field(min_length=3, max_length=5)
    expected_effect: str
    risks_and_limitations: str
    rank_score: float = 0.0
    expert_rejected: bool = False
    expert_comment: str | None = None
