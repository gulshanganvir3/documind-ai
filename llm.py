"""Factory for the chat model so every module uses identical settings."""

from functools import lru_cache

from langchain_openai import ChatOpenAI

import config

@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Return the shared chat model (temperature 0 for factual, repeatable answers)."""
    return ChatOpenAI(
        model = config.CHAT_MODEL, 
        temperature = config.TEMPERATURE,
        api_key = config.OPENAI_API_KEY,
        timeout = 60,
        max_retries = 2,
    )

