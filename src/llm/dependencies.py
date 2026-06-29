from functools import lru_cache

from src.llm.client import LLMClient


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    return LLMClient()