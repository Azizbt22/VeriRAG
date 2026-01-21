# scripts/build_index.py

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def main():
    chunks_path = Path("data/processed/chunks.json")
    index_dir = Path("index")
    index_dir.mkdir(exist_ok=True)

    if not chunks_path.exists():
        raise FileNotFoundError("chunks.json not found. Run preprocess_data.py first.")

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    documents = [
        Document(
            page_content=c["text"],
            metadata={
                "doc_id": c["doc_id"],
                "chunk_id": c["chunk_id"],
            },
        )
        for c in chunks
    ]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(documents, embeddings)

    vectorstore.save_local(index_dir)

    print(f"[OK] Built LangChain FAISS index with {len(documents)} chunks")


if __name__ == "__main__":
    main()


