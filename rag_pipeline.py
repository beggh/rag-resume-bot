"""
RAG Pipeline — Resume Q&A Bot
==============================
Project: Ask natural language questions about a candidate's resume.
Stack  : LangChain + HuggingFace Embeddings + ChromaDB + Gemini (Google)

Flow:
  1. Load PDF → extract raw text
  2. Chunk text → overlapping windows of ~500 tokens
  3. Embed each chunk → 384-dim float vector (local, no API needed)
  4. Store vectors + text in ChromaDB (local disk)
  5. At query time: embed query, cosine-search DB, return top-K chunks
  6. Feed [chunks + query] to LLM → grounded answer
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# LangChain document loaders & splitters
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# HuggingFace local embedding model (384 dims, no API key needed)
from langchain_community.embeddings import HuggingFaceEmbeddings

# Google Gemini LLM (free tier)
from langchain_google_genai import ChatGoogleGenerativeAI

# ChromaDB vector store integration
from langchain_community.vectorstores import Chroma

# RAG chain primitives
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()  # reads GOOGLE_API_KEY from .env


# ──────────────────────────────────────────────
# STEP 1: Load the PDF
# ──────────────────────────────────────────────
def load_pdf(path: str) -> list:
    """
    PyPDFLoader reads each page of the PDF as a separate Document object.
    Each Document has:
      .page_content  →  raw text of the page
      .metadata      →  {"source": "file.pdf", "page": 0}
    """
    loader = PyPDFLoader(path)
    pages = loader.load()
    print(f"[LOAD] Loaded {len(pages)} page(s) from {path}")
    return pages


# ──────────────────────────────────────────────
# STEP 2: Chunk the text
# ──────────────────────────────────────────────
def chunk_documents(pages: list) -> list:
    """
    WHY CHUNK?
    LLMs have a fixed context window. Embedding a 10-page doc as one vector
    loses granularity — the signal gets averaged out. Smaller chunks let the
    retriever find the exact paragraph that answers the question.

    RecursiveCharacterTextSplitter tries to split on \\n\\n → \\n → space
    in order, so it respects natural paragraph breaks.

    chunk_size    = max characters per chunk (~100-150 tokens at 4 chars/token)
    chunk_overlap = how many chars carry over between adjacent chunks
                    (prevents answers from being cut across boundaries)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    print(f"[CHUNK] Split into {len(chunks)} chunk(s)")
    for i, c in enumerate(chunks[:3]):  # preview first 3
        print(f"  chunk[{i}] ({len(c.page_content)} chars): {c.page_content[:80].strip()!r}...")
    return chunks


# ──────────────────────────────────────────────
# STEP 3 + 4: Embed & store in ChromaDB
# ──────────────────────────────────────────────
def build_vector_store(chunks: list, persist_dir: str = "./chroma_db") -> Chroma:
    """
    HOW EMBEDDINGS WORK:
    HuggingFaceEmbeddings runs the all-MiniLM-L6-v2 model locally.
    The model returns a list of 384 float numbers per chunk.
    These floats encode semantic meaning — similar sentences cluster nearby.
    No API key is needed; the model is downloaded once and cached locally.

    HOW CHROMADB STORES THEM:
    ChromaDB builds an HNSW (Hierarchical Navigable Small World) index.
    HNSW is a graph where each node connects to its nearest neighbours.
    Similarity search navigates this graph in O(log N) instead of O(N).

    What's stored per chunk:
      - The float vector (384 dims)
      - The original text (for retrieval)
      - Metadata (source file, page number)
      - An auto-generated ID
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",  # 384 dims, fast, runs locally
    )

    # Chroma.from_documents:
    #   1. Calls embed_documents(chunks) → list of vectors
    #   2. Inserts (vector, text, metadata) into the HNSW index
    #   3. Persists to disk at persist_dir
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
    )
    print(f"[EMBED] Stored {len(chunks)} vectors in ChromaDB at '{persist_dir}'")
    return vector_store


def load_vector_store(persist_dir: str = "./chroma_db") -> Chroma:
    """Load an already-built index from disk (skip re-embedding on restart)."""
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=persist_dir, embedding_function=embedding_model)


# ──────────────────────────────────────────────
# STEP 5: Retriever — cosine similarity search
# ──────────────────────────────────────────────
def build_retriever(vector_store: Chroma, k: int = 4):
    """
    The retriever wraps the vector store.
    When you call retriever.get_relevant_documents(query):
      1. query text is embedded → 1536-dim query vector
      2. ChromaDB computes cosine similarity between query vector and all stored vectors
      3. Returns the top-K chunks with highest similarity scores

    cosine_similarity(A, B) = (A·B) / (|A| * |B|)
    Range: -1 (opposite) to 1 (identical). Typical good match: > 0.75

    search_type options:
      "similarity"        → pure cosine similarity (default)
      "mmr"               → Maximal Marginal Relevance — balances
                            relevance with diversity to avoid repetitive chunks
    """
    retriever = vector_store.as_retriever(
        search_type="mmr",          # diversity-aware retrieval
        search_kwargs={
            "k": k,                 # return top 4 chunks
            "fetch_k": 10,          # MMR candidate pool size
            "lambda_mult": 0.7,     # 0=max diversity, 1=pure similarity
        },
    )
    print(f"[RETRIEVER] MMR retriever ready (k={k})")
    return retriever



# ──────────────────────────────────────────────
# STEP 6: RAG chain — retrieval + generation
# ──────────────────────────────────────────────
def build_rag_chain(retriever):
    """
    The RAG prompt template injects retrieved chunks into the LLM's context.
    {context} = the top-K chunk texts joined together
    {question} = the user's original query

    This is the core insight of RAG:
    The LLM never sees the whole document — only the most relevant excerpts.
    This keeps costs low and answers grounded (reduces hallucination).
    """
    prompt_template = """You are an expert HR analyst and technical recruiter.
Use ONLY the following resume excerpts to answer the question.
If the answer is not in the excerpts, say "I don't have that information in the resume."
Always cite which part of the resume supports your answer.

Resume excerpts:
{context}

Question: {question}

Answer (be specific and concise):"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",   # free tier: 15 req/min, 1M tokens/day
        temperature=0,              # 0 = deterministic, no creativity needed
        max_output_tokens=512,
    )

    # RetrievalQA wires retriever → prompt → LLM automatically
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",          # "stuff" = concatenate all chunks into one prompt
        retriever=retriever,
        return_source_documents=True, # include which chunks were used
        chain_type_kwargs={"prompt": PROMPT},
    )
    return chain


# ──────────────────────────────────────────────
# Main: index once, query many times
# ──────────────────────────────────────────────
def index_resume(pdf_path: str, db_dir: str = "./chroma_db"):
    """Run the full indexing pipeline (call this once)."""
    pages = load_pdf(pdf_path)
    chunks = chunk_documents(pages)
    vector_store = build_vector_store(chunks, db_dir)
    return vector_store


def answer_question(chain, question: str) -> dict:
    """Run a RAG query and print results with source attribution."""
    print(f"\n{'='*60}")
    print(f"Q: {question}")
    print(f"{'='*60}")

    result = chain.invoke({"query": question})
    answer = result["result"]
    sources = result["source_documents"]

    print(f"\nA: {answer}")
    print(f"\n[Sources used — {len(sources)} chunk(s)]")
    for i, doc in enumerate(sources, 1):
        page = doc.metadata.get("page", "?")
        preview = doc.page_content[:120].replace("\n", " ").strip()
        print(f"  [{i}] Page {page}: {preview!r}...")

    return result


if __name__ == "__main__":
    PDF_PATH = "PrakharSinghal_Resume.pdf"
    DB_DIR   = "./chroma_db"

    # Index (skip if DB already exists)
    if not Path(DB_DIR).exists():
        print("Building index for the first time...")
        vector_store = index_resume(PDF_PATH, DB_DIR)
    else:
        print("Loading existing index from disk...")
        vector_store = load_vector_store(DB_DIR)

    retriever = build_retriever(vector_store, k=4)
    chain     = build_rag_chain(retriever)

    # Sample questions
    questions = [
        "What are Prakhar's top technical skills?",
        "Tell me about the corporate reporting pipeline project.",
        "How many years of experience does Prakhar have?",
        "What databases has Prakhar worked with?",
        "Describe Prakhar's LLM or AI-related experience.",
        "What is Prakhar's educational background?",
    ]

    for q in questions:
        answer_question(chain, q)
