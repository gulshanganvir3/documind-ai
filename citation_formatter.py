"""Turns retrieval chunks into clean, human-readable citations. """
from langchain_core.documents import Document
from helpers import truncate

def build_citations(results: list[tuple[Document, float]], used_ids: list[int] | None=None) -> list[dict]:
    """convert (chunk, distance) pairs into citation dict.
    
    used_ids are 1 based sourse numbers the llm said it relied on. If given, only those
    are returned;  otherwise every retrieved chunk is returned.
    """

    citations: list[dict] = [] #final list shown on the UI
    for number, (doc, distance) in enumerate(results, start=1):
        if used_ids and number not in  used_ids:
            continue
        citations.append( # store everything the ui needs
            {
                "id": number,
                "file": doc.metadata.get("source", "unknown"), 
                "page": doc.metadata.get("page", "?"),
                "text": doc.page_content,
                "distance": round(float(distance), 3),

            }

        )
    return citations


def format_label(citation: dict) -> str:
    """Short label such as 'S1 . report.pdf . p.12',"""
    return f"S{citation['id']} . {citation['file']} . p.{citation['page']}"


def format_markdown(citation: dict) -> str:
    """Markdown contatining label plus a  trimmed quota of the evidance."""
    return f"**{format_label(citation)}**\n\n> {truncate(citation['text'], 500)}"
