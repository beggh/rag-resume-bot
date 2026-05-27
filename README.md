# Resume RAG Q&A Bot

Ask natural language questions about a candidate's resume using Retrieval-Augmented Generation.

## Architecture

```
INDEXING (one-time)
  PDF → PyPDFLoader → pages
  pages → RecursiveCharacterTextSplitter → chunks (600 chars, 100 overlap)
  chunks → OpenAI text-embedding-3-small → 1536-dim vectors
  vectors + text → ChromaDB (HNSW index, cosine similarity)

QUERYING (per question)
  query → embed → 1536-dim query vector
  query vector → ChromaDB MMR search → top 4 relevant chunks
  [system prompt + chunks + query] → GPT-4o-mini → grounded answer
```

## Key concepts

### Why chunk? Why not embed the whole doc?
- Embedding the full document averages out all meaning → blurry retrieval
- Chunks let the model find the *exact* paragraph, not the *approximate* document
- Overlap (100 chars) prevents answers from being cut across chunk boundaries

### Cosine similarity vs Euclidean distance
| Metric | Formula | Sensitive to |
|--------|---------|-------------|
| Cosine | `dot(A,B) / (|A|*|B|)` | Direction (meaning) |
| Euclidean | `sqrt(sum((A-B)^2))` | Magnitude + direction |

Cosine is preferred for text — "cat" and "cats cats cats cats" should be similar, but Euclidean distance would penalize the longer text.

### MMR (Maximal Marginal Relevance)
Standard similarity returns the 4 most similar chunks, which may all be near-duplicates (e.g., all from the same paragraph). MMR re-ranks to balance relevance with diversity:

```
MMR(d) = λ * sim(query, d) - (1-λ) * max(sim(d, selected))
```

`lambda_mult=0.7` means 70% relevance, 30% diversity.

### What's stored in ChromaDB per chunk
```
{
  id:        "chunk_3",
  embedding: [0.21, -0.73, 0.09, ... ×1536],
  document:  "Implemented idempotent SQS consumer scaling to 10K events/min...",
  metadata:  {"source": "PrakharSinghal_Resume.pdf", "page": 0}
}
```

## Setup

```bash
pip install langchain langchain-openai langchain-community chromadb pypdf python-dotenv
```

Create a `.env` file:
```
OPENAI_API_KEY=sk-...
```

## Run

```bash
# Understand embeddings (no index needed)
python embedding_explainer.py

# See ChromaDB internals
python vectordb_internals.py

# Full pipeline + sample questions
python rag_pipeline.py

# Interactive chat
python chat.py
```

## Files

| File | Purpose |
|------|---------|
| `rag_pipeline.py` | Full RAG pipeline with detailed comments |
| `embedding_explainer.py` | What embeddings are + cosine similarity math |
| `vectordb_internals.py` | What ChromaDB actually stores + raw query API |
| `chat.py` | Interactive CLI |
| `chroma_db/` | Persisted vector index (auto-created) |

## Extension ideas

- Swap ChromaDB for Pinecone/Qdrant for production scale
- Add `parent_document_retriever` to store small chunks but retrieve large ones
- Add a reranker (Cohere Rerank) after retrieval for higher accuracy
- Add evaluation with RAGAS (faithfulness, answer relevance, context recall)
- Build a Streamlit UI around `chat.py`
