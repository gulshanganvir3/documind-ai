""" central configuration for DocuMind AI: every tunable value lives here """
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data" / "BookShelf"
INDEX_DIR = BASE_DIR / "vectorstore_index"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # secret key, never hard-code it
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")  # LLM used for QA/summaries/judging
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")  # cheap, strong embeddings
TEMPERATURE = 0.0  # 0 = deterministic, which minimises hallucination


# chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval
TOP_K = 5
FETCH_K = 30

MAX_DISTANCE = 1.5

# Summarization
SUMMARY_BLOCK_CHARS = 6000
MAX_SUMMARY_BLOCKS = 12

MAX_UPLOAD_MB = 50 

def validate_config() -> list[str]:
    """Return a human-readable configuration problems (empty list = all good). """
    problems: list[str] = []
    if not OPENAI_API_KEY:
        problems.append("OPENAI_API_KEY is missing. Copy .env.example to .env and add your key.")
    return problems
