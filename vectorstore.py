"""step4: FAISS vector database - create, persist, load, inspect, and edit the index."""

import shutil
from collections import defaultdict

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import config
from embeddings import get_embeddings

INDEX_NAME = "documind"

def index_exists() -> bool:
    """True when a saved FAISS index is present on disk."""
    return (config.INDEX_DIR / f"{INDEX_NAME}.faiss").exists()


def load_vectorstore() -> FAISS | None:
    """Load the persisted index, or return None when nothing has been indexed yet."""
    if not index_exists():
        return None

    return FAISS.load_local(
        str(config.INDEX_DIR),
        get_embeddings(),
        index_name = INDEX_NAME,
        allow_dangerous_deserialization = True, 
    )


def save_vectorstore(store: FAISS) -> None:
    """persist the index so it survives app restarts."""
    store.save_local(str(config.INDEX_DIR), index_name=INDEX_NAME)


def add_chunks(store: FAISS | None, chunks: list[Document]) -> FAISS:
    """Embed chunks and add them to the index (created it if needed), then save."""
    if store is None:
        store = FAISS.from_documents(chunks, get_embeddings())
    else:
        store.add_documents(chunks)
    save_vectorstore(store)
    return store


def ids_for_source(store: FAISS, source: str) -> list[str]:
    """Internal: docstore ids of every chunks that belongs to one PDF."""
    return [
        doc_id
        for doc_id, doc in store.docstore._dict.items()
        if doc.metadata.get("source") == source
    ]



def remove_document(store: FAISS, source: str) -> FAISS | None:
    """Delete one PDF's chunks from the index; returns None if the index becomes empty."""
    ids = ids_for_source(store, source)
    if ids:
        store.delete(ids)
    if not store.docstore._dict:
        reset_vectorstore()
        return None
    save_vectorstore(store)
    return store


def reset_vectorstore() -> None:
   """Delete the whole saved index."""
   for suffix in (".faiss", ".pkl"):
       path = config.INDEX_DIR / f"{INDEX_NAME}{suffix}"
       if path.exists():
           path.unlink()





def list_documents(store: FAISS | None) -> dict[str, dict]:
    """Summary per indexed PDF: {'file.pdf': {'chunks': n, 'pages': m}}."""
    if store is None:
        return {}

    chunk_count: dict[str, int] = defaultdict(int)
    pages: dict[str, int] = defaultdict(set)
    for doc in store.docstore._dict.values():
        name = doc.metadata.get("source", "unknown")
        chunk_count[name] += 1 
        pages[name].add(doc.metadata.get("page"))
 
    return {
        name: {"chunks": chunk_count[name], "pages": len(pages[name])}
        for name in sorted(chunk_count)
    }




def get_chunks_for(store: FAISS, source: str) -> list[Document]:
    """All chunks of one PDF in reading order (page, then position on the page)."""
    docs = [d for d in store.docstore._dict.values() if d.metadata.get("source") == source]
    return sorted(docs, key=lambda d: (d.metadata.get("page", 0), d.metadata.get("start_index", 0)))
