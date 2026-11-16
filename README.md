# documind-ai
Streamlit app to chat with your PDFs, get cited answers, summaries, document comparisons, and answer quality evaluation.


## Features
- Upload multiple PDFs (with a size limit) and index them
- Ask questions and get answers with cited source passages
- Search in all documents or only selected ones
- Summarize any document
- Compare two or more documents (similarities, differences, unique points)
- Evaluate answers: faithfulness check and retrieval test (hit rate, MRR)
- Adjust passages per answer (k), delete documents, or reset everything
- LangSmith tracing

## Tech Stack
Python, Streamlit, LangChain, OpenAI, FAISS, LangSmith

## Setup
1. Clone the repo
2. Install packages: `pip install -r requirements.txt`
3. Create a `.env` file in the project folder and add your key:
   `OPENAI_API_KEY=your_key_here`
4. Run: `streamlit run app.py`
