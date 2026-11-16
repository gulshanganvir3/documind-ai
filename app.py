"""DocuMind AI - Streamlit front end. Run with:  streamlit run app.py"""

import json

import streamlit as st

import config
import evaluator, pdf_loader, text_spliter, vectorstore

from comparator import compare_documents
from qa_chain import answer_question

from summarizer import summarize_document
from citation_formatter import format_label

import os

os.environ["LANGSMITH_PROJECT"] = "DocuMind AI"

st.set_page_config(page_title="DocuMind AI", page_icon="📚", layout="wide")



# config check
problems = config.validate_config()
if problems:
    for problem in problems:
        st.error(problem)
    st.stop()



# session state
if "store" not in st.session_state:
    st.session_state.store = vectorstore.load_vectorstore()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "summaries" not in st.session_state:
    st.session_state.summaries = {}


store = st.session_state.store
docs = vectorstore.list_documents(store)


def refresh(new_store) -> None:
    """Save a new store object in the session state and rerun the page."""
    st.session_state.store = new_store
    st.rerun()



# SideBar
with st.sidebar:
    st.title("📚 DocuMind AI")    
    st.caption("Ask questions about your PDFs, with sources.")


    uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("process documents", type="primary", disabled=not uploads):
        too_big = [u.name for u in uploads if u.size > config.MAX_UPLOAD_MB * 1024 * 1024 ]
        if too_big:
            st.error(f"Too Large (>{config.MAX_UPLOAD_MB} MB): {', '.join(too_big)}")
        else:
            new_files = [u for u in uploads if u.name not in docs]
            if not new_files:
                st.info("All selected files are already indexed.")
            else:
                with st.status("processing...", expanded=True) as status:
                    st.write("Saving files...")
                    paths = [pdf_loader.save_uploaded_file(u) for u in new_files]
                    st.write("Extracting text...")  
                    pages, errors = pdf_loader.load_many(paths)
                    for fname, msg in errors.items():
                        st.warning(f"{fname}: {msg}")
                    if pages:
                        st.write("Spitting into chunks...")
                        chunks = text_spliter.split_documents(pages)
                        st.write(f"Embedding {len(chunks)} chunks (this calls OpenAI)...")

                        try:
                            new_store = vectorstore.add_chunks(st.session_state.store, chunks)
                        except Exception as exc:
                            status.update(label="Failed", state="error")
                            st.error(f"Indexing failed: {exc}")
                        else:
                            status.update(label="Done", state="complete")
                            refresh(new_store)
                    else:
                        status.update(label="No readable documents", state="error")


st.divider()
st.subheader("Indexed documents")
if not docs:
    st.caption("Nothing indexed yet.")
for fname, info in docs.items():
    col_a, col_b = st.columns([4, 1])
    col_a.markdown(f"**{fname}** \n{info['pages']} pages . {info['chunks']} chunks")
    if col_b.button("🗑", key=f"del_{fname}", help=f"Remove {fname}"):
        refresh(vectorstore.remove_document(st.session_state.store, fname))

st.divider()
top_k = st.slider("Passege per answer (k)", 2, 10, config.TOP_K)
if docs and st.button("Reset Everything"):
    vectorstore.reset_vectorstore()
    st.session_state.messages = []
    st.session_state.summaries = {}
    refresh(None)



# Main Area
st.header("DocuMind AI - Document Intelligent & Research Asistant")
if not docs:
    st.info("👈 Upload one or more PDFs and click **Process documents** to begin.")
    st.stop()

tab_ask, tab_sum, tab_cmp, tab_eval = st.tabs(["💬 Ask", "📝 Summarize", "⚖️ Compare", "🧪 Evaluate"])



# Ask TAB
with tab_ask:
    scope = st.multiselect("Search in (leave empty for all documents)",  list(docs))
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for c in msg.get("citations", []):
                with  st.expander(format_label(c)):
                    st.write(c["text"])

    question = st.chat_input("Ask a question about your documents...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                try:
                    result = answer_question(st.session_state.store, question, sources=scope or None, k=top_k)
                except Exception as exc:
                    result = {"answer":  f"⚠️ Error:{exc}", "answerable": False, "citations": [], "context": ""}
            st.markdown(result["answer"])
            if not result["answerable"]:
                st.caption("Not enough supporting evidance in the selected documents.")
            for c in result["citations"]:
                with st.expander(format_label(c)):
                    st.write(c["text"])

        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"], "citations": result["citations"],
             "question": question, "context": result["context"]}
        )


    if st.session_state.messages and st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun() 



# summarize tab
with tab_sum:
    target = st.selectbox("Documet", list(docs), key="sum_target")
    if st.button("Generate summary"):
        with st.spinner("Summarizing..."):
            try:
                st.session_state.summaries[target] = summarize_document(st.session_state.store, target)
            except Exception as exc:
                st.error(f"Summary failed: {exc}")
    if target in st.session_state.summaries:
        st.markdown(st.session_state.summaries[target])




with tab_cmp:
    picked = st.multiselect("Document to compare (2+)", list(docs), key="cmp_docs")
    topic = st.text_input("Opetional focus topic (e.g. 'pricing', 'methodology')")
    if st.button("Compare", disabled=len(picked) < 2):
        with st.spinner("Comparing..."):
            try:
                cmp = compare_documents(st.session_state.store, picked, topic or None, cache=st.session_state.summaries)
                st.subheader("Overview")
                st.write(cmp.overview)
                st.subheader("Similarities")
                for s in cmp.similarities:
                    st.markdown(f" - {s}")
                st.subheader("Differences")
                for d in cmp.differences:
                    st.markdown(f" - {d}")
                st.subheader("Unique to each document")
                for item in cmp.unique_points:
                    st.markdown(f"**{item.document}**")
                    for p in item.points:
                        st.markdown(f" - {p}")
            except Exception as exc:
                st.error(f"Comparison failed: {exc}")



# Evaluation TAB
with tab_eval:
    st.subheader("1) Is the last answer supported by its sources?")
    assistant_turns = [m for m in st.session_state.messages if m["role"] == "assistant" and "question" in m]
    if not assistant_turns:
        st.caption("Ask a question in the Ask tab first.")
    elif st.button("Check last answer"):
        last = assistant_turns[-1]
        with st.spinner("Fact-checking..."):
            try:
                verdict = evaluator.judge_faithfulness(last["question"], last["content"], last["context"])
                st.metric("Faithfulness (1-5)", verdict.score)
                st.write("✅ Supported" if verdict.supported else "⚠️ Contains unsupported claims")
                st.write(verdict.reasoning)
                for claim in verdict.unsupported_claim:
                    st.markdown(f"- ❌ {claim}")
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
    

    st.divider()
    st.subheader("2) Retrieval quality on a test set")
    example = json.dumps(
        [{"question": "What is the main topic?", "expected_source": next(iter(docs)), "expected_keyword": ""}], indent=2

    )

    raw = st.text_area("Text cases (JSON list: question, excepted_source, excepted_source, excepted_page, excepted_keyword)", example, height=180)
    if st.button("Run retrieval test"):
        try:
            cases = json.loads(raw)
            report = evaluator.evaluate_retrieval(st.session_state.store, cases, k=top_k)
            col1, col2 = st.columns(2)
            col1.metric(f"Hit rate @ {top_k}", f"{report['hit_rate']:.0%}")
            col2.metric("MMR", f"{report['mrr']:.02f}")   
            st.dataframe(report["rows"], use_container_width=True)
        except Exception as exc:
            st.error(f"Could not run test: {exc}")



