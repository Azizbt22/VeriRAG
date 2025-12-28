# src/rag.py

import json
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.retrievers import BaseRetriever


INDEX_DIR = Path("index")


def load_retriever(
    index_dir: Path = INDEX_DIR,
    k: int = 4,
) -> BaseRetriever:
    """
    Load a precomputed FAISS index and expose it as a LangChain retriever.

    Assumes the following files exist in index_dir:
    - faiss.index
    - metadata.json

    Index construction, chunking, and embedding computation
    are assumed to have been done offline beforehand.
    """

    # MUST match the embedding model used in build_index.py
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        folder_path=str(index_dir),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    return vectorstore.as_retriever(search_kwargs={"k": k})


def load_metadata(
    index_dir: Path = INDEX_DIR,
) -> Optional[dict]:
    """
    Load metadata associated with indexed chunks (optional).
    Useful for analysis or debugging.
    """

    metadata_path = index_dir / "metadata.json"
    if not metadata_path.exists():
        return None

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)
