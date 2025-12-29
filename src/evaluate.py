# src/evaluate.py

import json
from pathlib import Path
from typing import Dict, Any, List

from src.models import get_llm
from src.rag import get_retriever
from src.agent import build_agent


# ---------- CONFIG ----------

QUESTIONS_PATH = Path("data/eval/questions.json")
OUTPUT_PATH = Path("runs/eval_results.json")
OUTPUT_PATH.parent.mkdir(exist_ok=True)


# ---------- EVALUATION ----------

def evaluate_question(agent, q: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs the agent on a single question and extracts
    evaluation-relevant signals.
    """
    result = agent(q["question"])

    verdict = result.get("verdict", "")
    verdict_label = "PASS" if "PASS" in verdict else "FAIL"

    retrieval_trace = result.get("retrieval_trace", [])
    used_docs = list({d["doc_id"] for d in retrieval_trace})

    return {
        "id": q["id"],
        "question": q["question"],
        "verdict": verdict_label,
        "num_retrieved_chunks": len(retrieval_trace),
        "used_documents": used_docs,
        "answer": result.get("answer", ""),
        "verdict_raw": verdict,
        "retrieval_trace": retrieval_trace,
    }


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r["verdict"] == "PASS")

    avg_chunks = (
        sum(r["num_retrieved_chunks"] for r in results) / total
        if total > 0 else 0
    )

    doc_usage = {}
    for r in results:
        for doc in r["used_documents"]:
            doc_usage[doc] = doc_usage.get(doc, 0) + 1

    return {
        "total_questions": total,
        "pass_rate": passed / total if total > 0 else 0,
        "average_chunks_retrieved": round(avg_chunks, 2),
        "document_usage": doc_usage,
    }


# ---------- MAIN ----------

def main():
    print("[INFO] Loading evaluation questions...")
    questions = json.loads(QUESTIONS_PATH.read_text())

    print("[INFO] Initializing agent...")
    llm = get_llm()
    retriever = get_retriever()
    agent = build_agent(llm, retriever)

    print("[INFO] Running evaluation...")
    results = []
    for q in questions:
        print(f"→ Evaluating: {q['id']}")
        res = evaluate_question(agent, q)
        results.append(res)

    metrics = compute_metrics(results)

    output = {
        "metrics": metrics,
        "results": results,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n=== EVALUATION SUMMARY ===")
    print(f"Pass rate: {metrics['pass_rate'] * 100:.1f}%")
    print(f"Avg retrieved chunks: {metrics['average_chunks_retrieved']}")
    print("Document usage:", metrics["document_usage"])
    print(f"\n[OK] Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
