import asyncio
import json
import re
import time
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any, Literal, Self, TypeVar

import httpx
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from common.config import settings
from common.inference.schemas import (
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    RerankRequest,
    RerankResponse,
    RerankResult,
)

T = TypeVar("T", bound=BaseModel)

DEFAULT_METALLURGY_INSTRUCTION = "Retrieve semantically similar metallurgical literature and experimental evidence for hypothesis evaluation"


class InferenceClientError(Exception):
    """Base exception for inference client errors."""


class TransientInferenceError(InferenceClientError):
    """Transient error suitable for retry."""


class InferenceTimeoutError(InferenceClientError):
    """Timeout error when calling inference services."""


class InferenceClient:
    """Async unified client for llama.cpp server and FastAPI ml-inference service."""

    def __init__(
        self,
        llm_base_url: str | None = None,
        ml_inference_base_url: str | None = None,
        llm_model_name: str | None = None,
        embedding_model_name: str | None = None,
        reranker_model_name: str | None = None,
        llm_timeout: float | None = None,
        ml_timeout: float | None = None,
        llm_bg_max_concurrency: int | None = None,
    ) -> None:
        self.llm_base_url = (llm_base_url or settings.llm_base_url).rstrip("/")
        self.ml_inference_base_url = (
            ml_inference_base_url or settings.ml_inference_base_url
        ).rstrip("/")
        self.llm_model_name = llm_model_name or settings.llm_model_name
        self.embedding_model_name = embedding_model_name or settings.embedding_model_name
        self.reranker_model_name = reranker_model_name or settings.reranker_model_name
        self.llm_timeout = llm_timeout or settings.llm_request_timeout_seconds
        self.ml_timeout = ml_timeout or settings.ml_inference_timeout_seconds

        concurrency = llm_bg_max_concurrency or settings.llm_bg_max_concurrency
        self.bg_semaphore = asyncio.Semaphore(max(1, concurrency))

        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=self.llm_timeout, write=10.0, pool=30.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def _post_with_retry(
        self,
        url: str,
        json_payload: dict[str, Any],
        timeout: float,
    ) -> httpx.Response:
        retrying = AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1.0, min=1.0, max=5.0),
            retry=retry_if_exception_type(
                (httpx.ConnectError, httpx.NetworkError, TransientInferenceError)
            ),
            reraise=True,
        )
        client = self._get_client()

        async for attempt in retrying:
            with attempt:
                try:
                    response = await client.post(url, json=json_payload, timeout=timeout)
                    if response.status_code == 503:
                        raise TransientInferenceError(f"Service unavailable (503): {response.text}")
                    if response.status_code >= 400:
                        raise InferenceClientError(f"HTTP {response.status_code}: {response.text}")
                    return response
                except httpx.ReadTimeout as exc:
                    raise InferenceTimeoutError(
                        f"Inference request timed out after {timeout}s: {url}"
                    ) from exc
                except httpx.ConnectError as exc:
                    raise TransientInferenceError(f"Connection failed to {url}") from exc
                except httpx.NetworkError as exc:
                    raise TransientInferenceError(f"Network error calling {url}") from exc

        raise InferenceClientError(f"Unexpected retry failure for {url}")

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        enable_thinking: bool = True,
        response_format: dict[str, Any] | None = None,
        priority: Literal["high", "background"] = "background",
    ) -> ChatResponse:
        target_model = model or self.llm_model_name
        payload: dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format

        url = f"{self.llm_base_url}/chat/completions"
        start_time = time.perf_counter()

        if priority == "background":
            async with self.bg_semaphore:
                response = await self._post_with_retry(url, payload, timeout=self.llm_timeout)
        else:
            response = await self._post_with_retry(url, payload, timeout=self.llm_timeout)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        data = response.json()
        choice = data["choices"][0]["message"]
        content = choice.get("content") or ""
        reasoning_content = choice.get("reasoning_content")

        # Fallback parsing for <think> tags if reasoning_content is not separated
        if not reasoning_content and "<think>" in content and "</think>" in content:
            match = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
            if match:
                reasoning_content = match.group(1).strip()
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        return ChatResponse(
            content=content,
            reasoning_content=reasoning_content,
            model=data.get("model", target_model),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        enable_thinking: bool = True,
        priority: Literal["high", "background"] = "high",
    ) -> AsyncIterator[str]:
        target_model = model or self.llm_model_name
        payload: dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        url = f"{self.llm_base_url}/chat/completions"
        client = self._get_client()

        # Stream requests
        async def _stream_generator() -> AsyncIterator[str]:
            try:
                async with client.stream(
                    "POST", url, json=payload, timeout=self.llm_timeout
                ) as resp:
                    if resp.status_code >= 400:
                        raise InferenceClientError(f"HTTP {resp.status_code} in stream")
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        chunk_str = line[len("data: ") :].strip()
                        if chunk_str == "[DONE]":
                            break
                        try:
                            chunk_json = json.loads(chunk_str)
                            delta = chunk_json["choices"][0].get("delta", {})
                            token = delta.get("content") or ""
                            if token:
                                yield token
                        except json.JSONDecodeError:
                            continue
            except httpx.ReadTimeout as exc:
                raise InferenceTimeoutError(f"Stream read timed out: {url}") from exc

        if priority == "background":
            async with self.bg_semaphore:
                async for chunk in _stream_generator():
                    yield chunk
        else:
            async for chunk in _stream_generator():
                yield chunk

    async def chat_structured(
        self,
        messages: list[dict[str, Any]],
        response_model: type[T],
        model: str | None = None,
        temperature: float = 0.0,
        enable_thinking: bool = True,
        priority: Literal["high", "background"] = "background",
    ) -> T:
        schema = response_model.model_json_schema()
        schema_json = json.dumps(schema, ensure_ascii=False)

        if not enable_thinking:
            response_format = {"type": "json_object"}
            system_instruction = (
                f"\nYou MUST output valid JSON complying with the following schema:\n{schema_json}"
            )
            enhanced_messages = list(messages)
            if enhanced_messages and enhanced_messages[0].get("role") == "system":
                enhanced_messages[0] = {
                    "role": "system",
                    "content": enhanced_messages[0]["content"] + system_instruction,
                }
            else:
                enhanced_messages.insert(0, {"role": "system", "content": system_instruction})

            response = await self.chat(
                messages=enhanced_messages,
                model=model,
                temperature=temperature,
                enable_thinking=False,
                response_format=response_format,
                priority=priority,
            )
            return response_model.model_validate_json(response.content)

        # Two-step mode when enable_thinking=True to avoid GBNF syntax conflict with <think> tags
        step1_response = await self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            enable_thinking=True,
            priority=priority,
        )

        extraction_messages = list(messages)
        extraction_messages.append({"role": "assistant", "content": step1_response.content})
        extraction_messages.append(
            {
                "role": "user",
                "content": (
                    "Based on the analysis and thought process above, extract and format the result into strict JSON "
                    f"matching this schema:\n{schema_json}\nReturn ONLY the JSON object."
                ),
            }
        )

        step2_response = await self.chat(
            messages=extraction_messages,
            model=model,
            temperature=0.0,
            enable_thinking=False,
            response_format={"type": "json_object"},
            priority=priority,
        )
        return response_model.model_validate_json(step2_response.content)

    async def embed_documents(
        self,
        texts: list[str],
        client_batch_size: int = 32,
    ) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        url = f"{self.ml_inference_base_url}/v1/embeddings"

        # Client-side chunking to protect against HTTP timeouts
        for i in range(0, len(texts), client_batch_size):
            chunk = texts[i : i + client_batch_size]
            payload = EmbeddingRequest(input=chunk, model=self.embedding_model_name).model_dump()
            resp = await self._post_with_retry(url, payload, timeout=self.ml_timeout)
            parsed = EmbeddingResponse.model_validate(resp.json())
            # Ensure order preserved by index
            sorted_data = sorted(parsed.data, key=lambda d: d.index)
            all_embeddings.extend([d.embedding for d in sorted_data])

        return all_embeddings

    async def embed_queries(
        self,
        queries: list[str],
        instruction: str | None = None,
    ) -> list[list[float]]:
        if not queries:
            return []

        inst = instruction or DEFAULT_METALLURGY_INSTRUCTION
        # Client-side asymmetric formatting
        prefixed_queries = [f"Instruct: {inst}\nQuery: {q}" for q in queries]
        return await self.embed_documents(prefixed_queries)

    async def rerank(
        self,
        query: str,
        documents: list[str],
        instruction: str | None = None,
        return_documents: bool = False,
        top_n: int | None = None,
    ) -> list[RerankResult]:
        if not documents:
            return []

        url = f"{self.ml_inference_base_url}/v1/rerank"
        payload = RerankRequest(
            query=query,
            documents=documents,
            instruction=instruction or DEFAULT_METALLURGY_INSTRUCTION,
            return_documents=return_documents,
            top_n=top_n,
        ).model_dump()

        resp = await self._post_with_retry(url, payload, timeout=self.ml_timeout)
        parsed = RerankResponse.model_validate(resp.json())
        return parsed.results

    async def health_check(self) -> dict[str, bool]:
        client = self._get_client()
        status: dict[str, bool] = {"llm_server": False, "ml_inference": False}

        try:
            r_llm = await client.get(f"{self.llm_base_url.replace('/v1', '')}/health", timeout=5.0)
            status["llm_server"] = r_llm.status_code == 200
        except Exception:
            status["llm_server"] = False

        try:
            r_ml = await client.get(f"{self.ml_inference_base_url}/health", timeout=5.0)
            status["ml_inference"] = r_ml.status_code == 200
        except Exception:
            status["ml_inference"] = False

        return status
