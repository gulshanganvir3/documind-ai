"""step1 of the pipeline: save oploaded PDFs and extract text page by page."""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

import config

from helpers import sanitize_filename


def save_uploaded_file(uploaded_file) ->  Path:
    """Write Streamlit UploadedFile to disk and return its path."""
    safe_name = sanitize_filename(uploaded_file.name)
    destination = config.UPLOAD_DIR / safe_name
    destination.write_bytes(uploaded_file.getbuffer())
    return destination



def load_pdf(path: Path) -> list[Document]:
    """Extract text one PDF; returns one Document per non-empty page. """
    pages = PyPDFLoader(str(path)).load()  
    cleaned: list[Document] = []
    for page in pages:
        text = page.page_content.strip()
        if not text:
            continue
        cleaned.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "page": int(page.metadata.get("page", 0)) + 1,
                },
            )
        )
    if not cleaned:
        raise ValueError(
            f"no extractable text found in '{path.name}'. It may be scanned PDF (needs OCR)."
        )
    return cleaned



def load_many(paths: list[Path]) -> tuple[list[Document], dict[str, str]]:
    """Load saveral PDFs; Returns (all pages, {filename: error message}) so one bad file doesn't stop the rest."""
    all_pages: list[Document] = []
    errors: dict[str, str] = {}
    for path in paths:
        try:
            all_pages.extend(load_pdf(path))
        except Exception as exe:
            errors[path.name] = str(exe)
    
    return all_pages, errors

