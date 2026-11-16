"""step2: split page text into overlapping chunks suitable for embedding."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

def split_documents(pages: list[Document]) -> list[Document]:
    """split pages into chunks; page/ source metadata is inherited automatically."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = config.CHUNK_SIZE,
        chunk_overlap = config.CHUNK_OVERLAP,
        separators = ["\n\n", "\n", ". ", " ", ""],
        add_start_index = True,
    )

    chunks  = splitter.split_documents(pages)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
    return chunks

