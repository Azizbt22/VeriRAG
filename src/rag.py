# src/rag.py

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def get_retriever(index_dir: str = "index", k: int = 4):
    index_dir = Path(index_dir)

    if not index_dir.exists():
        raise FileNotFoundError("Index directory not found. Run build_index.py first.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vectorstore.as_retriever(search_kwargs={"k": k})
