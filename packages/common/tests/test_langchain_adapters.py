import json

import httpx
import pytest
from common.inference.client import InferenceClient
from common.inference.langchain_adapters import QwenChatModel, QwenEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.outputs import ChatGeneration
from pydantic import BaseModel


class SampleStructuredOutput(BaseModel):
    mechanism: str
    relevance_score: float


@pytest.mark.asyncio
async def test_qwen_chat_model_agenerate_and_stream() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("stream"):
            chunks = [
                'data: {"choices": [{"delta": {"content": "Austenite "}}]}\n\n',
                'data: {"choices": [{"delta": {"content": "transformation."}}]}\n\n',
                "data: [DONE]\n\n",
            ]
            return httpx.Response(200, content="".join(chunks).encode("utf-8"))
        else:
            return httpx.Response(
                200,
                json={
                    "model": "qwen3-chat",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Precipitation hardening",
                                "reasoning_content": "Internal thought",
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
            )

    transport = httpx.MockTransport(handler)
    client = InferenceClient(llm_base_url="http://mock-llm:8080/v1")
    client._client = httpx.AsyncClient(transport=transport)

    model = QwenChatModel(client=client, enable_thinking=False)

    # Test agenerate
    result = await model.agenerate([[HumanMessage(content="Tell me about hardening")]])
    assert len(result.generations) == 1
    gen = result.generations[0][0]
    assert isinstance(gen, ChatGeneration)
    msg = gen.message
    assert msg.content == "Precipitation hardening"
    assert msg.additional_kwargs["reasoning_content"] == "Internal thought"

    # Test astream
    stream_chunks: list[str] = []
    async for chunk in model.astream([HumanMessage(content="Explain")]):
        stream_chunks.append(str(chunk.content))
    assert "".join(stream_chunks) == "Austenite transformation."
    await client.close()


@pytest.mark.asyncio
async def test_qwen_chat_model_with_structured_output() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["chat_template_kwargs"]["enable_thinking"] is False
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {"mechanism": "Grain refinement", "relevance_score": 0.88}
                            ),
                        }
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            },
        )

    transport = httpx.MockTransport(handler)
    client = InferenceClient(llm_base_url="http://mock-llm:8080/v1")
    client._client = httpx.AsyncClient(transport=transport)

    model = QwenChatModel(client=client, enable_thinking=False)
    structured_runnable = model.with_structured_output(SampleStructuredOutput)

    output = await structured_runnable.ainvoke([HumanMessage(content="Analyze mechanism")])
    assert isinstance(output, SampleStructuredOutput)
    assert output.mechanism == "Grain refinement"
    assert output.relevance_score == 0.88
    await client.close()


@pytest.mark.asyncio
async def test_qwen_embeddings_adapter() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        inputs = body["input"]
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "Qwen3-Embedding",
                "data": [
                    {"index": i, "embedding": [float(i), float(i + 1)], "object": "embedding"}
                    for i, _ in enumerate(inputs)
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    client = InferenceClient(ml_inference_base_url="http://mock-ml:8081")
    client._client = httpx.AsyncClient(transport=transport)

    embeddings = QwenEmbeddings(client=client)

    docs_vecs = await embeddings.aembed_documents(["doc1", "doc2"])
    assert len(docs_vecs) == 2
    assert docs_vecs[0] == [0.0, 1.0]
    assert docs_vecs[1] == [1.0, 2.0]

    query_vec = await embeddings.aembed_query("search query")
    assert query_vec == [0.0, 1.0]
    await client.close()
