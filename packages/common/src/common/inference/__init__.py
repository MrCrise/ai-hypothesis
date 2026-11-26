"""Inference subsystem client and ecosystem adapters."""

from common.inference.client import (
    DEFAULT_METALLURGY_INSTRUCTION,
    InferenceClient,
    InferenceClientError,
    InferenceTimeoutError,
    TransientInferenceError,
)
from common.inference.langchain_adapters import QwenChatModel, QwenEmbeddings
from common.inference.schemas import (
    ChatResponse,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    RerankRequest,
    RerankResponse,
    RerankResult,
)

__all__ = [
    "DEFAULT_METALLURGY_INSTRUCTION",
    "ChatResponse",
    "EmbeddingData",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "InferenceClient",
    "InferenceClientError",
    "InferenceTimeoutError",
    "QwenChatModel",
    "QwenEmbeddings",
    "RerankRequest",
    "RerankResponse",
    "RerankResult",
    "TransientInferenceError",
]
