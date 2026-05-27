"""
Vector DB Internals
====================
Shows exactly what ChromaDB stores and how to query it at a low level,
independent of LangChain abstractions.
"""

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed(text: str) -> list[float]:
    """Local HuggingFace embedding — no API key needed."""
    return _model.encode(text).tolist()


def demo_chroma():
    """
    ChromaDB storage model:
    ┌─────────────────────────────────────────────────────────┐
    │  Collection: "resume_chunks"                            │
    │  ┌─────────────────────────────────────────────────┐   │
    │  │  id: "chunk_0"                                  │   │
    │  │  embedding: [0.21, -0.73, 0.09, ...]   (384 d)  │   │
    │  │  document: "Prakhar Singhal | BITS Pilani..."    │   │
    │  │  metadata: {"source": "resume.pdf", "page": 0}  │   │
    │  └─────────────────────────────────────────────────┘   │
    │  ... (one row per chunk)                                │
    └─────────────────────────────────────────────────────────┘

    The HNSW index sits alongside the raw data for fast ANN search.
    """

    # In-memory client for this demo (no persistence)
    client = chromadb.Client()

    collection = client.create_collection(
        name="resume_chunks",
        metadata={"hnsw:space": "cosine"},  # distance metric
    )

    # Resume-like documents
    docs = [
        "Implemented idempotent SQS consumer scaling to 10K events/min for data pipeline.",
        "Engineered data ingestion pipeline slashing report generation from 15 days to 4 hours.",
        "Implemented and integrated unified interface for LLM APIs with prompt preprocessing.",
        "Architected AWS IDP pipeline using OCR and ML anomaly detection for insurance claims.",
        "Google Calendar 2-way event sync using CDC, Kafka achieving 3K RPM scalability.",
        "Bachelors of Engineering in Mechanical Engineering + MSc in Economics from BITS Pilani.",
        "Skills: Java, Python, GoLang, AWS, Docker, Kubernetes, PostgreSQL, Redis, Kafka.",
        "Reduced latency of product discovery, improving revenue by 5% across platforms.",
    ]

    print("Embedding and inserting documents into ChromaDB...")
    embeddings = [embed(d) for d in docs]

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(docs))],
        embeddings=embeddings,
        documents=docs,
        metadatas=[{"source": "resume.pdf", "page": 0, "chunk": i} for i in range(len(docs))],
    )

    print(f"Stored {collection.count()} vectors\n")

    # Query the collection
    queries = [
        "What is Prakhar's experience with machine learning or AI?",
        "What are the infrastructure and cloud skills?",
        "Tell me about the data pipeline performance improvements.",
    ]

    for q in queries:
        q_vec = embed(q)
        results = collection.query(
            query_embeddings=[q_vec],
            n_results=2,
            include=["documents", "distances", "metadatas"],
        )

        print(f"Query: {q!r}")
        for doc, dist in zip(results["documents"][0], results["distances"][0]):
            similarity = 1 - dist  # chroma returns cosine distance; convert to similarity
            bar = "█" * int(similarity * 25)
            print(f"  sim={similarity:.3f} {bar}")
            print(f"  → {doc[:90]}...")
        print()


if __name__ == "__main__":
    demo_chroma()
