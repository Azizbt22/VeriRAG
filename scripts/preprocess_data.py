import json
import yaml
from pathlib import Path
from transformers import AutoTokenizer


TOKENIZER_NAME = "meta-llama/Llama-2-7b-hf"  
# replace by "mistralai/Mistral-7B-v0.1"  when using mistral


def load_config():
    with open("configs/data.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def chunk_text_with_tokenizer(text, tokenizer, chunk_size, overlap):
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    chunks = []
    start = 0

    while start < len(token_ids):
        end = start + chunk_size
        chunk_token_ids = token_ids[start:end]

        chunk_text = tokenizer.decode(
            chunk_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        chunks.append(chunk_text)
        start = end - overlap

    return chunks


def main():
    cfg = load_config()

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        use_fast=True
    )

    raw_dir = Path(cfg["dataset"]["raw_dir"])
    processed_dir = Path(cfg["dataset"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    chunk_id = 0

    for file in raw_dir.glob("*.txt"):
        text = file.read_text(encoding="utf-8", errors="ignore")

        chunks = chunk_text_with_tokenizer(
            text=text,
            tokenizer=tokenizer,
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

    print(f"[OK] Saved {len(all_chunks)} chunks")


if __name__ == "__main__":
    main()
