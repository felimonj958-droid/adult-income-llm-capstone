# src/llm/dependencies.py
from functools import lru_cache

from src.llm.client import LLMClient


@lru_cache
def get_llm_client() -> LLMClient:
    return LLMClient()
