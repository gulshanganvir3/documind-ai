"""Feature 6: compare two or more PDFs and report similarities and differences."""

from langchain_community.vectorstores import FAISS  # vector store type
from langchain_core.prompts import ChatPromptTemplate  # prompt builder

from pydantic import BaseModel, Field  # structured output schema
from llm import get_llm  # shared chat model

from retriever import retrieve  # topic-focused retrieval
from summarizer import summarize_document  # whole-document overviews


class DocPoints(BaseModel):
    """points that are specific to single documents."""

    document: str = Field(description="Exact document filename.")
    points: list[str] = Field(description="Distinctive points found only (or mainly) in this document.")


class Comparison(BaseModel):
    """Structured comparison result."""

    overview: str = Field(description = "2-3 sentence high-level comparison.")
    similarities: list[str] = Field(description = "points the documents agree on or share.")
    differences: list[str] = Field(description = "points where documents differ or conflict; name the documents.")
    unique_points: list[DocPoints] = Field(description = "Distictive content per document.")



PROMPT = ChatPromptTemplate(
    [

        (
            "system",
            "You compare documents. Use ONLY the material provided. Be specific, cite document names "
            "in each point, and do not invent facts. If the material is insufficient, say so.",

        ),
        ("human", "Focus: {focus}\n\nMaterial per document: \n\n{material}"),
    ]
)



def compare_documents(store: FAISS, names: list[str], topic: str | None=None, cache: dict | None=None) -> Comparison:
    """Comapare documents overall (summaries) or on a specific topic (retrieved passages)."""
    if len(names)  < 2:
        raise ValueError("Select at least two documents to compare.")
    sections = []
    for name in names:
        if topic:
            hits = retrieve(store, topic, sources=[name], k=6)
            body = "\n".join(f"(p.{d.metadata.get('page')}) {d.page_content}" for d, _ in hits) or "No relevant passages found."
        else:
            if cache is not None and name in cache:
                body = cache[name]
            else:
                body = summarize_document(store, name)
                if cache is not None:
                    cache[name] = body
        sections.append(f"### {name}\n{body}")
    focus = topic or "Overall content, purpose, and conclusions"
    chain = PROMPT | get_llm().with_structured_output(Comparison)
    return chain.invoke({"focus": focus, "material": "\n\n".join(sections)})

