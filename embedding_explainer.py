"""
Embedding Explainer
====================
Demonstrates what embeddings are, what they look like numerically,
and how cosine similarity works — with real examples from the resume.
Run this independently to understand the math before running the RAG pipeline.
"""

import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    cosine_similarity = dot(A, B) / (norm(A) * norm(B))

    Why cosine instead of Euclidean distance?
    - Euclidean is sensitive to vector magnitude (length of text matters)
    - Cosine only cares about DIRECTION → pure meaning, not length
    - Range: -1 (opposite) to +1 (identical)
    """
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def demo_embeddings():
    model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    sentences = {
        # resume-relevant
        "candidate_exp":   "4.5 years of experience in backend software engineering",
        "sde_job":         "Looking for a senior software engineer with 4+ years experience",
        "llm_work":        "Implemented LLM engine with prompt preprocessing for doctors",
        "llm_jd":          "Experience with LLMs, fine-tuning, and agentic workflows required",
        # unrelated
        "cooking":         "How to make pasta with tomato sauce",
        "finance":         "Stock market closed down 2% on inflation fears",
    }

    print("Generating embeddings...")
    texts = list(sentences.values())
    keys  = list(sentences.keys())
    vecs  = model.embed_documents(texts)

    print(f"\nEmbedding dimensionality: {len(vecs[0])} floats per sentence  (all-MiniLM-L6-v2, 384 dims)")
    print(f"First 8 values of 'candidate_exp' vector:")
    print(f"  {[round(x, 4) for x in vecs[0][:8]]} ...")

    print("\n--- Cosine Similarity Matrix ---")
    print(f"{'':25}", end="")
    for k in keys:
        print(f"{k[:12]:>14}", end="")
    print()

    for i, ki in enumerate(keys):
        print(f"{ki[:25]:25}", end="")
        for j, _ in enumerate(keys):
            sim = cosine_similarity(vecs[i], vecs[j])
            marker = " ◄" if i != j and sim > 0.70 else ""
            print(f"{sim:>13.3f}{marker[:1]:1}", end="")
        print()

    print("\n--- Key observations ---")
    pairs = [
        ("candidate_exp", "sde_job",   "Resume exp ↔ Job requirement"),
        ("llm_work",      "llm_jd",    "LLM work ↔ AI JD requirement"),
        ("candidate_exp", "cooking",   "Resume ↔ Cooking (unrelated)"),
        ("llm_work",      "finance",   "LLM work ↔ Finance (unrelated)"),
    ]
    for a, b, label in pairs:
        ai, bi = keys.index(a), keys.index(b)
        sim = cosine_similarity(vecs[ai], vecs[bi])
        bar = "█" * int(sim * 20)
        print(f"  {label:<40} {sim:.3f}  {bar}")


if __name__ == "__main__":
    demo_embeddings()
