import asyncio
from collections.abc import AsyncIterator
from typing import Any, TypeVar

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field

from common.config import settings
from common.inference.client import InferenceClient

T = TypeVar("T", bound=BaseModel)


def _convert_message_to_dict(message: BaseMessage) -> dict[str, Any]:
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        return {"role": "assistant", "content": message.content}
    elif isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    elif isinstance(message, ChatMessage):
        return {"role": message.role, "content": message.content}
    else:
        return {"role": "user", "content": message.content}


class QwenChatModel(BaseChatModel):
    """LangChain chat model adapter for local Qwen inference."""

    model_name: str = Field(default_factory=lambda: settings.llm_model_name)
    temperature: float = 0.7
    max_tokens: int | None = None
    enable_thinking: bool = False
    client: InferenceClient = Field(default_factory=InferenceClient)

    @property
    def _llm_type(self) -> str:
        return "qwen-chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return asyncio.run(self._agenerate(messages, stop, run_manager=None, **kwargs))

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        dict_messages = [_convert_message_to_dict(m) for m in messages]
        thinking = kwargs.get("enable_thinking", self.enable_thinking)
        temp = kwargs.get("temperature", self.temperature)
        tokens = kwargs.get("max_tokens", self.max_tokens)
        priority = kwargs.get("priority", "background")

        resp = await self.client.chat(
            messages=dict_messages,
            model=self.model_name,
            temperature=temp,
            max_tokens=tokens,
            enable_thinking=thinking,
            priority=priority,
        )

        ai_msg = AIMessage(
            content=resp.content,
            additional_kwargs={
                "reasoning_content": resp.reasoning_content,
                "model": resp.model,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "latency_ms": resp.latency_ms,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        dict_messages = [_convert_message_to_dict(m) for m in messages]
        thinking = kwargs.get("enable_thinking", self.enable_thinking)
        temp = kwargs.get("temperature", self.temperature)
        tokens = kwargs.get("max_tokens", self.max_tokens)
        priority = kwargs.get("priority", "high")

        async for chunk in self.client.chat_stream(
            messages=dict_messages,
            model=self.model_name,
            temperature=temp,
            max_tokens=tokens,
            enable_thinking=thinking,
            priority=priority,
        ):
            yield ChatGenerationChunk(message=AIMessageChunk(content=chunk))

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        """Support for structured output compatible with LangGraph and LangChain pipelines."""
        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            raise TypeError("Schema must be a Pydantic BaseModel subclass.")

        target_schema: type[BaseModel] = schema

        async def _call_structured(input_messages: list[BaseMessage] | dict[str, Any]) -> Any:
            if isinstance(input_messages, dict) and "messages" in input_messages:
                msgs = input_messages["messages"]
            elif isinstance(input_messages, list):
                msgs = input_messages
            else:
                msgs = [HumanMessage(content=str(input_messages))]

            dict_messages = [_convert_message_to_dict(m) for m in msgs]
            return await self.client.chat_structured(
                messages=dict_messages,
                response_model=target_schema,
                model=self.model_name,
                temperature=0.0,
                enable_thinking=self.enable_thinking,
                priority=kwargs.get("priority", "background"),
            )

        return RunnableLambda(_call_structured)


class QwenEmbeddings(Embeddings):
    """LangChain Embeddings adapter for Qwen3-Embedding."""

    def __init__(self, client: InferenceClient | None = None) -> None:
        self.client = client or InferenceClient()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return asyncio.run(self.aembed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        return asyncio.run(self.aembed_query(text))

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.client.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        results = await self.client.embed_queries([text])
        return results[0] if results else []
