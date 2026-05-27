# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install langchain langchain-google-genai langchain-community chromadb pypdf python-dotenv sentence-transformers
```

Create a `.env` file with:
```
GOOGLE_API_KEY=...
```

## Running the code

```bash
python embedding_explainer.py   # Embedding concepts demo (no index needed)
python vectordb_internals.py    # ChromaDB internals demo (no index needed)
python rag_pipeline.py          # Full RAG pipeline with sample questions
python chat.py                  # Interactive CLI chat
```

The vector index is auto-created in `./chroma_db/` on first run and reused on subsequent runs. Delete that directory to force re-indexing.

## Architecture

Two-phase RAG system over a single PDF resume:

**Indexing (one-time):** `PyPDFLoader` → `RecursiveCharacterTextSplitter` (600 chars, 100 overlap) → HuggingFace `all-MiniLM-L6-v2` (384 dims, local) → ChromaDB (HNSW, cosine similarity, persisted to `./chroma_db/`)

**Querying (per question):** embed query → ChromaDB MMR search (`k=4`, `fetch_k=10`, `lambda_mult=0.7`) → `gemini-1.5-flash` (`temperature=0`, `max_output_tokens=512`) with a grounding prompt

**Key files:**
- `rag_pipeline.py` — Core pipeline with all indexing/retrieval/generation logic; `chat.py` imports from it
- `chat.py` — Thin interactive wrapper around `rag_pipeline.py`
- `embedding_explainer.py` / `vectordb_internals.py` — Standalone educational demos, no shared code with the pipeline

**MMR retrieval** (`lambda_mult=0.7`) balances 70% relevance with 30% diversity to avoid returning near-duplicate chunks from the same paragraph.

The LLM prompt instructs the model to answer using **only** the retrieved chunks and cite the source — this grounds answers and prevents hallucination.
