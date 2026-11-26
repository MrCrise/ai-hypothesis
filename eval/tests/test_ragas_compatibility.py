import json

import httpx
import pytest
from common.inference import InferenceClient, QwenChatModel, QwenEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper


@pytest.mark.asyncio
async def test_ragas_wrapper_compatibility() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "/chat/completions" in url_str:
            return httpx.Response(
                200,
                json={
                    "model": "qwen3-chat",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Ragas evaluation response text.",
                                "reasoning_content": None,
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 8},
                },
            )
        elif "/v1/embeddings" in url_str:
            body = json.loads(request.content)
            inputs = body["input"]
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": "Qwen3-Embedding",
                    "data": [
                        {"index": i, "embedding": [0.05 * i, 0.1 * i], "object": "embedding"}
                        for i, _ in enumerate(inputs)
                    ],
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = InferenceClient(
        llm_base_url="http://mock-llm:8080/v1",
        ml_inference_base_url="http://mock-ml:8081",
    )
    client._client = httpx.AsyncClient(transport=transport)

    qwen_llm = QwenChatModel(client=client, enable_thinking=False)
    qwen_emb = QwenEmbeddings(client=client)

    ragas_llm = LangchainLLMWrapper(qwen_llm)
    ragas_emb = LangchainEmbeddingsWrapper(qwen_emb)

    assert ragas_llm is not None
    assert ragas_emb is not None

    # Test embed query through Ragas wrapper
    vec = await ragas_emb.aembed_query("Evaluate answer relevancy")
    assert isinstance(vec, list)
    assert len(vec) == 2

    await client.close()
