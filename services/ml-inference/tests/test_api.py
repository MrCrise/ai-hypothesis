import asyncio
from collections.abc import AsyncGenerator

import httpx
import pytest
from httpx import ASGITransport
from ml_inference.main import app, batcher


@pytest.fixture(autouse=True)
async def lifespan_fixture() -> AsyncGenerator[None, None]:
    batcher.start()
    yield
    await batcher.stop()


@pytest.mark.asyncio
async def test_health_endpoint() -> None:
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "device" in data


@pytest.mark.asyncio
async def test_embeddings_endpoint() -> None:
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {"input": ["alloy A", "alloy B"], "model": "Qwen3-Embedding"}
        resp = await client.post("/v1/embeddings", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 2
        assert len(data["data"][0]["embedding"]) == 1024


@pytest.mark.asyncio
async def test_rerank_endpoint() -> None:
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "query": "Ti-6Al-4V",
            "documents": ["Paper 1", "Paper 2"],
            "return_documents": True,
            "top_n": 1,
        }
        resp = await client.post("/v1/rerank", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) == 1
        assert "relevance_score" in data["results"][0]
        assert data["results"][0]["document"] == "Paper 1"


@pytest.mark.asyncio
async def test_batcher_concurrent_requests() -> None:
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Fire 5 concurrent requests
        async def call(i: int) -> httpx.Response:
            return await client.post("/v1/embeddings", json={"input": [f"query_{i}"]})

        responses = await asyncio.gather(*(call(i) for i in range(5)))
        for r in responses:
            assert r.status_code == 200
            assert len(r.json()["data"]) == 1
