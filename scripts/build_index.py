# scripts/build_index.py

import json
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def main():
    chunks_path = Path("data/processed/chunks.json")
    index_dir = Path("index")
    index_dir.mkdir(exist_ok=True)

    # Load chunks
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]
    print(f"[INFO] Loaded {len(texts)} chunks")

    # Hugging Face embeddings (local, no API)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("[INFO] Computing embeddings and building FAISS index...")
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
    )

    # Save FAISS index in LangChain format
    vectorstore.save_local(str(index_dir))

    # Save metadata (chunk_id, doc_id, etc.)
    with open(index_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print("[OK] FAISS index built with Hugging Face embeddings")


if __name__ == "__main__":
    main()

