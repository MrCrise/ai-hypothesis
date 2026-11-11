from openai import AsyncOpenAI

from common.config import settings


def get_llm_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
