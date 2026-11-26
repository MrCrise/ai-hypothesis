import json

import httpx
import pytest
from common.inference.client import InferenceClient
from common.inference.schemas import ChatResponse, RerankResult
from pydantic import BaseModel


class DummyHypothesis(BaseModel):
    title: str
    confidence: float


@pytest.mark.asyncio
async def test_client_chat_with_reasoning_and_metadata() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert "/chat/completions" in str(request.url)
        body = json.loads(request.content)
        assert body["chat_template_kwargs"]["enable_thinking"] is True
        response_data = {
            "model": "qwen3-test",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Final answer",
                        "reasoning_content": "Deep thought process",
                    }
                }
            ],
            "usage": {"prompt_tokens": 15, "completion_tokens": 25},
        }
        return httpx.Response(200, json=response_data)

    transport = httpx.MockTransport(handler)
    client = InferenceClient(llm_base_url="http://mock-llm:8080/v1")
    client._client = httpx.AsyncClient(transport=transport)

    resp = await client.chat(messages=[{"role": "user", "content": "hi"}], priority="high")
    assert isinstance(resp, ChatResponse)
    assert resp.content == "Final answer"
    assert resp.reasoning_content == "Deep thought process"
    assert resp.prompt_tokens == 15
    assert resp.completion_tokens == 25
    assert resp.latency_ms > 0
    await client.close()


@pytest.mark.asyncio
async def test_client_chat_structured_two_step() -> None:
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 1:
            # Step 1: Thinking response
            assert body["chat_template_kwargs"]["enable_thinking"] is True
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Thought out proposal for high-entropy alloys",
                                "reasoning_content": "Considering phase transitions...",
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                },
            )
        else:
            # Step 2: Extraction response
            assert body["chat_template_kwargs"]["enable_thinking"] is False
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": json.dumps({"title": "HE-Alloy", "confidence": 0.95}),
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 35, "completion_tokens": 15},
                },
            )

    transport = httpx.MockTransport(handler)
    client = InferenceClient(llm_base_url="http://mock-llm:8080/v1")
    client._client = httpx.AsyncClient(transport=transport)

    result = await client.chat_structured(
        messages=[{"role": "user", "content": "Generate alloy hypothesis"}],
        response_model=DummyHypothesis,
        enable_thinking=True,
    )
    assert len(calls) == 2
    assert isinstance(result, DummyHypothesis)
    assert result.title == "HE-Alloy"
    assert result.confidence == 0.95
    await client.close()


@pytest.mark.asyncio
async def test_client_embed_documents_chunking() -> None:
    batches_received = []

    async def handler(request: httpx.Request) -> httpx.Response:
        assert "/v1/embeddings" in str(request.url)
        body = json.loads(request.content)
        inputs = body["input"]
        batches_received.append(inputs)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "Qwen3-Embedding",
                "data": [
                    {"index": idx, "embedding": [0.1 * idx, 0.2 * idx], "object": "embedding"}
                    for idx, _ in enumerate(inputs)
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    client = InferenceClient(ml_inference_base_url="http://mock-ml:8081")
    client._client = httpx.AsyncClient(transport=transport)

    texts = [f"doc_{i}" for i in range(70)]
    embeddings = await client.embed_documents(texts, client_batch_size=32)

    # 70 items with batch_size 32 => 3 requests (32, 32, 6)
    assert len(batches_received) == 3
    assert len(batches_received[0]) == 32
    assert len(batches_received[1]) == 32
    assert len(batches_received[2]) == 6
    assert len(embeddings) == 70
    await client.close()


@pytest.mark.asyncio
async def test_client_embed_queries_asymmetry() -> None:
    captured_inputs = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_inputs.extend(body["input"])
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "Qwen3-Embedding",
                "data": [{"index": 0, "embedding": [0.1, 0.2], "object": "embedding"}],
            },
        )

    transport = httpx.MockTransport(handler)
    client = InferenceClient(ml_inference_base_url="http://mock-ml:8081")
    client._client = httpx.AsyncClient(transport=transport)

    await client.embed_queries(["austenite grain growth"])
    assert len(captured_inputs) == 1
    assert captured_inputs[0].startswith("Instruct:")
    assert "Query: austenite grain growth" in captured_inputs[0]
    await client.close()


@pytest.mark.asyncio
async def test_client_rerank_cohere_format() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert "/v1/rerank" in str(request.url)
        body = json.loads(request.content)
        assert body["query"] == "Ti-alloy phase transformation"
        assert body["return_documents"] is True
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.94, "document": "Paper B"},
                    {"index": 0, "relevance_score": 0.42, "document": "Paper A"},
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = InferenceClient(ml_inference_base_url="http://mock-ml:8081")
    client._client = httpx.AsyncClient(transport=transport)

    results = await client.rerank(
        query="Ti-alloy phase transformation",
        documents=["Paper A", "Paper B"],
        return_documents=True,
    )
    assert len(results) == 2
    assert isinstance(results[0], RerankResult)
    assert results[0].index == 1
    assert results[0].relevance_score == 0.94
    assert results[0].document == "Paper B"
    await client.close()
