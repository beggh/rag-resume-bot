"""
Interactive CLI — Resume Q&A Bot
==================================
Run this after indexing is done. Type questions and get answers.
"""

from pathlib import Path
from rag_pipeline import (
    index_resume,
    load_vector_store,
    build_retriever,
    build_rag_chain,
    answer_question,
)

PDF_PATH = "PrakharSinghal_Resume.pdf"
DB_DIR   = "./chroma_db"

def main():
    if not Path(DB_DIR).exists():
        print("First run — indexing resume...")
        vs = index_resume(PDF_PATH, DB_DIR)
    else:
        print("Loading existing index...")
        vs = load_vector_store(DB_DIR)

    retriever = build_retriever(vs, k=4)
    chain     = build_rag_chain(retriever)

    print("\n" + "="*60)
    print("  Resume Q&A Bot — Prakhar Singhal's Resume")
    print("  Type 'quit' to exit")
    print("="*60 + "\n")

    while True:
        question = input("Ask a question: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        answer_question(chain, question)

if __name__ == "__main__":
    main()
