# T-RAG — A Retrieval-Augmented Generation System with Evaluation

A retrieval-augmented generation (RAG) pipeline that answers questions over a
collection of PDF documents, paired with an LLM-as-judge evaluation harness that
measures answer quality and localizes failures to either retrieval or generation.

Built as a learning project to understand RAG end to end — not just wiring the
components together, but measuring how well the system actually performs.

## What it does

Ask a natural-language question about the loaded documents and get an answer
grounded in the source material, with the retrieved sources cited. A separate
evaluation script scores the system across a fixed question set so pipeline
changes (chunk size, retrieval depth, prompt wording) can be compared with real
numbers instead of guesswork.

## Architecture

The pipeline follows the standard RAG flow: **load → split → embed → store →
retrieve → generate.**

| Stage | Choice | Why |
|-------|--------|-----|
| Load | `PyPDFDirectoryLoader` | Ingests a folder of PDFs, one document per page with source/page metadata |
| Split | `RecursiveCharacterTextSplitter` (chunk_size=500, overlap=200) | Splits on natural boundaries; chosen over structure-based splitting because raw PDF text carries no reliable header markup |
| Embed | `sentence-transformers/all-MiniLM-L6-v2` (local) | Runs locally — no API cost or rate limits on the embedding step |
| Store | Chroma (persisted to disk) | Embed once, reuse across runs instead of re-embedding on every launch |
| Retrieve | Top-k similarity search (k=3) | Fetches the most relevant chunks per question |
| Generate | Google Gemini | Produces an answer grounded strictly in the retrieved context |

The code is split into three files so the app and the evaluation share one
source of truth for the pipeline:

- **`rag_core.py`** — builds the pipeline once on import and exposes `get_answer()`
- **`app.py`** — a Gradio web UI wrapping `get_answer()`
- **`eval.py`** — the evaluation harness

## Evaluation

The evaluation uses the **LLM-as-judge** approach (the method behind RAGAS,
DeepEval, and TruLens) to score three component-level metrics:

- **Context relevance** — did the retriever fetch the right chunks?
- **Faithfulness** — is the answer supported by the retrieved chunks, with no fabrication?
- **Answer relevance** — does the answer actually address the question asked?

Reading the three together localizes failures: low context relevance points to a
retrieval problem (chunking, k, embeddings), while high context relevance with a
poor answer points to a generation problem (prompt, model). This separation is
the reason to measure components rather than just eyeballing final answers.

### Results

_(To be filled in.)_

- Answer relevance: **X.X / 5** across N questions
- Main failure mode observed: _e.g. retrieval missing specific numeric facts_
- Key takeaway: _e.g. chunk_size 500 balanced precision and context for these documents_

## Design decisions & tradeoffs

- **Local embeddings over an API.** Moving embeddings to a local
  sentence-transformer eliminated rate-limit failures during batch indexing and
  removed per-embedding cost, at the price of running the model on-device.
- **Recursive chunking over structure-based.** The documents are structured, but
  as PDFs their headers don't survive as markup, so a structure-aware splitter
  had nothing reliable to split on. Recursive splitting on paragraph/sentence
  boundaries was the pragmatic choice.
- **Persisted vector store.** Persisting Chroma to disk means the corpus is
  embedded once and loaded on subsequent runs, avoiding repeated embedding work.
- **Separated core / app / eval.** A shared `rag_core` module guarantees the
  evaluation tests the same pipeline the app ships, avoiding copy-paste drift.

## Known limitations

- LLM-as-judge scores carry known biases (e.g. verbosity) and are best used for
  *comparing* configurations, not as absolute quality certificates.
- Free-tier LLM quotas cap how many questions can be run per day, which bounds
  both the app and the size of an evaluation run.
- Retrieval is pure vector similarity; no reranking or hybrid (keyword + vector)
  search yet.

## Possible next steps

- Add a reranking step or hybrid search to improve retrieval precision.
- Implement claim-level faithfulness (extract claims, verify each against context).
- Deploy as a hosted "Ask me" assistant on a personal site.

## Tech stack

Python · LangChain · Chroma · sentence-transformers · Google Gemini · Gradio

## Setup for users

```bash
python3.12 -m venv t-rag
source t-rag/bin/activate
pip install -r requirements.txt
```

Add a `.env` file with your API key:

```
GOOGLE_API_KEY=your-key-here
```

Place your PDFs in `t-rag/pdfs/`, then:

```bash
python app.py     # launch the web UI
python eval.py    # run the evaluation harness
```
