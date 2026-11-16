"""Feature 5: map-reduce summarisation of a whole PDF."""

from langchain_community.vectorstores import FAISS  # vector store type
from langchain_core.output_parsers import StrOutputParser  # LLM message -> plain string
from langchain_core.prompts import ChatPromptTemplate  # prompt builder

import config  # block sizes
from llm import get_llm  # shared chat model
from vectorstore import get_chunks_for  # ordered chunks of one PDF
from helpers import evenly_sample  # cost control for huge documents



MAP_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "Summarize the following excerpt from the document '{name}' in 4-6 bullet points. "
            "Keep key facts, figures and names. Use only the text.\n\nExcerpt:\n{text}",
        ),
    ]
)

REDUCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "Below are partial summaries of the document '{name}', in order. Write a concise, well-structured "
            "summary with: 1) a 2-3 sentence overview, 2) key points as bullets, 3) notable figures/conclusions. "
            "Use only the information given.\n\n{summaries}",
        ),
    ]
)



def summarize_document(store: FAISS, name: str) -> str:
    """Summarize one indexed PDF (map-reduce, bounded by MAX_SUMMARY_BLOCKS)."""
    chunks = get_chunks_for(store, name)
    if not chunks:
        return "Document not found in the index."

    full_text = "\n\n".join(c.page_content for c in chunks)
    size = config.SUMMARY_BLOCK_CHARS
    all_blocks = [full_text[i:i + size] for i in range(0, len(full_text), size)]
    blocks = evenly_sample(all_blocks, config.MAX_SUMMARY_BLOCKS)
    parser = StrOutputParser()
    map_chain = MAP_PROMPT | get_llm() | parser

    partials = map_chain.batch(
        [{"name": name, "text": block} for block in blocks],
        config = {"max_concurrency": 4},

    )

    if len(partials) == 1:
        return partials[0]
    reduce_chain = REDUCE_PROMPT | get_llm() | parser
    return reduce_chain.invoke({"name": name, "summaries": "\n\n".join(partials)})

