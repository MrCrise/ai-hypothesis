from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from common.config import settings
from common.inference.schemas import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    RerankRequest,
    RerankResponse,
)
from fastapi import FastAPI, HTTPException

from ml_inference.batcher import BatchingQueueWorker
from ml_inference.engine import MLEngine

engine = MLEngine()
batcher = BatchingQueueWorker(engine=engine)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    engine.load_models()
    batcher.start()
    yield
    await batcher.stop()


app = FastAPI(
    title="ML Inference Service",
    description="Dedicated FastAPI service for Qwen3-Embedding and Qwen3-Reranker",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "device": engine.device,
        "models_loaded": engine.models_loaded,
        "embedding_model": settings.embedding_model_name,
        "reranker_model": settings.reranker_model_name,
    }


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    if isinstance(request.input, str):
        inputs = [request.input]
    else:
        inputs = request.input

    if not inputs:
        return EmbeddingResponse(
            data=[],
            model=request.model or settings.embedding_model_name,
            usage=EmbeddingUsage(prompt_tokens=0, total_tokens=0),
        )

    try:
        embeddings = await batcher.enqueue_embed(inputs)
        data = [
            EmbeddingData(index=idx, embedding=vec, object="embedding")
            for idx, vec in enumerate(embeddings)
        ]
        return EmbeddingResponse(
            data=data,
            model=request.model or settings.embedding_model_name,
            usage=EmbeddingUsage(prompt_tokens=len(inputs), total_tokens=len(inputs)),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest) -> RerankResponse:
    if not request.documents:
        return RerankResponse(results=[])

    try:
        results = await batcher.enqueue_rerank(
            query=request.query,
            documents=request.documents,
            instruction=request.instruction,
            return_documents=request.return_documents,
            top_n=request.top_n,
        )
        return RerankResponse(results=results)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
