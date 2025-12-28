import json
from pathlib import Path

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def main():
    chunks_path = Path("data/processed/chunks.json")
    index_dir = Path("index")
    index_dir.mkdir(exist_ok=True)

    # Load chunks
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]
    print(f"[INFO] Loaded {len(texts)} chunks")

    # TF-IDF vectorization
    print("[INFO] Computing TF-IDF embeddings...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X = vectorizer.fit_transform(texts)
    X = X.toarray().astype("float32")
    X = normalize(X, axis=1)

    # Build FAISS index (cosine similarity)
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    # Save index
    faiss.write_index(index, str(index_dir / "faiss.index"))

    # Save metadata
    with open(index_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"[OK] FAISS index built with {index.ntotal} vectors")


if __name__ == "__main__":
    main()
