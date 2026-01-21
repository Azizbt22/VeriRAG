import json
import yaml
import re
from pathlib import Path


def load_config():
    with open("configs/data.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_text(text: str) -> str:
    """
    Minimal, deterministic text cleaning suitable for TF-IDF.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9.,;:!?()\-\s]", "", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int, overlap: int):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start = end - overlap

    return chunks


def main():
    cfg = load_config()

    raw_dir = Path(cfg["dataset"]["raw_dir"])
    processed_dir = Path(cfg["dataset"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    chunk_id = 0

    for file in raw_dir.glob("*.txt"):
        text = file.read_text(encoding="utf-8", errors="ignore")
        text = clean_text(text)

        chunks = chunk_text(
            text=text,
            chunk_size=cfg["chunking"]["chunk_size"],
            overlap=cfg["chunking"]["chunk_overlap"],
        )

        for c in chunks:
            all_chunks.append({
                "chunk_id": chunk_id,
                "doc_id": file.name,
                "text": c
            })
            chunk_id += 1

    out_path = processed_dir / "chunks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"[OK] Saved {len(all_chunks)} cleaned chunks")


if __name__ == "__main__":
    main()
