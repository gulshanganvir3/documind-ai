"""Feature 8: measure retrieval quality and check that answers are supported by sources."""


from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field 

import config
from llm import get_llm
from retriever import retrieve



class Verdict(BaseModel):
    """LLM as-judge verdict on whether an answer is grounded in the context."""

    score: int = Field(description="1-5. 5 = every claim fully supported by the content, 1 = most unsupported.")
    supported: bool = Field(description="True if all important claims are suppported.")
    unsupported_claim: list[str] = Field(description="Claims in the answer not backed by the  context.")
    reasoning: str = Field(description="One or two sentences explaining the verdict.")




JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",

            "You are a strict fact-checker. Decide wheather the ANSWER is fully supported by the context."
            "Ignore your own knowledge: a claim is supported only if the context states or clearly implies it."
            "A correct refusal ('not found in the documents') when the context lacks the answer counts as supported."
        ),
        ("human", "CONTEXT: \n{context}\nQUESTION: {question}\nANSWER:\n{answer}"),
    ]
)




def judge_faithfulness(question: str, answer: str, context: str) -> Verdict:
    """Check one answer against the evidence it was generated from."""
    chain = JUDGE_PROMPT | get_llm().with_structured_output(Verdict)
    return chain.invoke({"question": question, "answer": answer, "context": context or "(no context retrieved)"})






def evaluate_retrieval(store: FAISS, test_cases: list[dict], k: int = config.TOP_K):
    """Run a labelled test set through retrieval.

    Each test case: {"question": str, "expected_source": str (optional),
                     "expected_page": int (optional), "expected_keyword": str (optional)}.
    A retrieved chunk is a HIT when it matches every expectation that was provided.
    Returns hit-rate@k, mean reciprocal rank (MRR) and per-question rows.
    """

    rows = []
    hits = 0
    reciprocal_ranks = []
    for case in test_cases:
        results = retrieve(store, case["question"], k=k)
        rank = 0
        for position, (doc, _score) in enumerate(results, start=1):
            meta = doc.metadata
            ok = True
            if case.get("expected_source") and meta.get("source") != case["expected_source"]:
                ok = False
            if case.get("expected_page") and meta.get("page") != int(case["expected_page"]):
                ok = False
            if case.get("expected_keyword") and case["expected_keyword"].lower() not in doc.page_content.lower():
                ok = False
            if ok:
                rank = position
                break
        hits += 1 if rank else 0
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        rows.append({"question": case["question"], "hit": bool(rank), "rank": rank or None, "retrieved": len(results)})
        
    total = max(len(test_cases), 1)
    return {
        "hit_rate": hits / total,
        "mrr": sum(reciprocal_ranks) / total,
        "rows": rows,
        }