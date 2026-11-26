from pydantic import BaseModel, Field


class ChatResponse(BaseModel):
    content: str
    reasoning_content: str | None = None
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: str | None = None


class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    instruction: str | None = None
    return_documents: bool = False
    top_n: int | None = None


class RerankResponse(BaseModel):
    results: list[RerankResult]


class EmbeddingData(BaseModel):
    index: int
    embedding: list[float]
    object: str = "embedding"


class EmbeddingUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingRequest(BaseModel):
    input: list[str] | str
    model: str | None = None


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage = Field(default_factory=EmbeddingUsage)
