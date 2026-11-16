"""The embedding model that turns text into vectors"""

from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

import config

@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    """Return the embedding model used used for both indexing and querying (they MUST match)."""
    return OpenAIEmbeddings(
        model = config.EMBEDDING_MODEL,
        api_key = config.OPENAI_API_KEY,
        chunk_size = 200,
    )
