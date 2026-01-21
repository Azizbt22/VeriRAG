"""
Model Comparison - Separate Vanilla and VeriRAG evaluations
============================================================
Run multiple models on EITHER vanilla OR verirag (not both)
"""
import json, time, argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch, gc
from src.models import get_llm
from src.rag import get_retriever
from src.agent import build_agent
from src.faithfulness_scorer import FaithfulnessScorer
from langchain_core.prompts import ChatPromptTemplate

QUESTIONS_PATH = Path("data/eval/questions_final.json")
OUTPUT_DIR = Path("runs")
OUTPUT_DIR.mkdir(exist_ok=True)

# 6 SMALL MODELS
MODEL_CONFIGS = {
    "tinyllama": {"model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "temperature": 0.2, "max_tokens": 512},
    "llama-1b": {"model_name": "meta-llama/Llama-3.2-1B-Instruct", "temperature": 0.2, "max_tokens": 512},
    "qwen-1.5b": {"model_name": "Qwen/Qwen2.5-1.5B-Instruct", "temperature": 0.2, "max_tokens": 512},
    "gemma-2b": {"model_name": "google/gemma-2-2b-it", "temperature": 0.2, "max_tokens": 512},
    "phi-2": {"model_name": "microsoft/phi-2", "temperature": 0.2, "max_tokens": 512},
    "llama-3b": {"model_name": "meta-llama/Llama-3.2-3B-Instruct", "temperature": 0.2, "max_tokens": 512}
}

def create_questions():
    questions = [
        {"id": "ml_easy_1", "question": "What is overfitting in machine learning?", "difficulty": "easy", "category": "concepts"},
        {"id": "ml_easy_2", "question": "What is a neural network?", "difficulty": "easy", "category": "concepts"},
        {"id": "ml_easy_3", "question": "Define machine learning.", "difficulty": "easy", "category": "concepts"},
        {"id": "ml_easy_4", "question": "What is the Turing Test?", "difficulty": "easy", "category": "concepts"},
        {"id": "ml_easy_5", "question": "What are transformers in deep learning?", "difficulty": "easy", "category": "concepts"},
        {"id": "ml_easy_6", "question": "What is a large language model?", "difficulty": "easy", "category": "concepts"},
        {"id": "ml_easy_7", "question": "What is supervised learning?", "difficulty": "easy", "category": "methods"},
        {"id": "ml_easy_8", "question": "What is unsupervised learning?", "difficulty": "easy", "category": "methods"},
        {"id": "ml_easy_9", "question": "What is deep learning?", "difficulty": "easy", "category": "concepts"},
        {"id": "ml_easy_10", "question": "What is reinforcement learning?", "difficulty": "easy", "category": "methods"},
        {"id": "ml_easy_11", "question": "What is backpropagation?", "difficulty": "easy", "category": "methods"},
        {"id": "ml_med_1", "question": "Explain the transformer architecture and its key components.", "difficulty": "medium", "category": "architecture"},
        {"id": "ml_med_2", "question": "How does the attention mechanism work in transformers?", "difficulty": "medium", "category": "methods"},
        {"id": "ml_med_3", "question": "How are large language models trained?", "difficulty": "medium", "category": "methods"},
        {"id": "ml_med_4", "question": "What are the different types of neural network layers?", "difficulty": "medium", "category": "architecture"},
        {"id": "ml_med_5", "question": "How do you prevent overfitting?", "difficulty": "medium", "category": "methods"},
        {"id": "ml_med_6", "question": "What is transfer learning?", "difficulty": "medium", "category": "methods"},
        {"id": "ml_med_7", "question": "Explain gradient descent optimization.", "difficulty": "medium", "category": "methods"},
        {"id": "ml_med_8", "question": "What are activation functions and why are they important?", "difficulty": "medium", "category": "concepts"},
        {"id": "ood_1", "question": "How do I bake a chocolate cake?", "difficulty": "easy", "category": "out_of_domain"},
        {"id": "ood_2", "question": "Who won the 2022 FIFA World Cup?", "difficulty": "easy", "category": "out_of_domain"},
        {"id": "ood_3", "question": "What is the capital of Brazil?", "difficulty": "easy", "category": "out_of_domain"},
        {"id": "ood_4", "question": "How do I fix a leaking faucet?", "difficulty": "easy", "category": "out_of_domain"},
        {"id": "ood_5", "question": "What are the health benefits of green tea?", "difficulty": "easy", "category": "out_of_domain"}
    ]
    QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(QUESTIONS_PATH, 'w') as f:
        json.dump(questions, f, indent=2)
    return questions

def evaluate_vanilla(llm, retriever, questions, scorer):
    prompt = ChatPromptTemplate.from_template("Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    results = []
    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {q['id']}", end='\r')
        docs = retriever.invoke(q["question"])
        context = "\n\n".join([d.page_content for d in docs])
        formatted = prompt.format(context=context, question=q["question"])
        response = llm.invoke(formatted)
        answer = response.content if hasattr(response, 'content') else str(response)
        faith_result = scorer.compute_faithfulness_score(answer, context)
        results.append({"id": q["id"], "question": q["question"], "answer": answer, "difficulty": q.get("difficulty"), "category": q.get("category"), "faithfulness": faith_result})
    print()
    return results

def evaluate_verirag(agent, questions, scorer):
    results = []
    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {q['id']}", end='\r')
        result = agent(q["question"])
        context = "\n\n".join([chunk.get("preview", "") for chunk in result.get("retrieval_trace", [])])
        faith_result = scorer.compute_faithfulness_score(result["answer"], context)
        results.append({"id": q["id"], "question": q["question"], "answer": result["answer"], "verdict": result.get("verdict"), "difficulty": q.get("difficulty"), "category": q.get("category"), "faithfulness": faith_result})
    print()
    return results

def compute_metrics(results: List[Dict]) -> Dict:
    if not results:
        return {}
    faith_scores = [r["faithfulness"]["faithfulness_score"] for r in results]
    pass_count = sum(1 for r in results if r["faithfulness"]["verdict"] in ["PASS", "ABSTAIN"])
    by_difficulty = {}
    for difficulty in set(r["difficulty"] for r in results):
        diff_results = [r for r in results if r["difficulty"] == difficulty]
        diff_scores = [r["faithfulness"]["faithfulness_score"] for r in diff_results]
        by_difficulty[difficulty] = {"count": len(diff_results), "avg_faithfulness": float(np.mean(diff_scores))}
    return {
        "total_questions": len(results),
        "avg_faithfulness": float(np.mean(faith_scores)),
        "median_faithfulness": float(np.median(faith_scores)),
        "pass_rate": pass_count / len(results),
        "by_difficulty": by_difficulty
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['vanilla', 'verirag'], help='Which system to evaluate')
    parser.add_argument('--models', nargs='+', choices=list(MODEL_CONFIGS.keys()), help='Specific models to test')
    parser.add_argument('--all-models', action='store_true', help='Test all 6 models')
    args = parser.parse_args()
    
    print("="*80)
    print(f"MODEL COMPARISON - {args.mode.upper()} ONLY")
    print("="*80)
    
    if not QUESTIONS_PATH.exists():
        questions = create_questions()
    else:
        questions = json.loads(QUESTIONS_PATH.read_text())
    
    print(f"\nQuestions: {len(questions)}")
    print(f"Mode: {args.mode}")
    
    scorer = FaithfulnessScorer(use_embeddings=True)
    
    # Determine which models to test
    if args.all_models:
        models_to_test = list(MODEL_CONFIGS.keys())
    elif args.models:
        models_to_test = args.models
    else:
        models_to_test = ['llama-3b']  # Default
    
    print(f"Models: {', '.join(models_to_test)}\n")
    
    all_results = {}
    
    for model_key in models_to_test:
        config = MODEL_CONFIGS[model_key]
        print(f"{'='*80}")
        print(f"[{models_to_test.index(model_key)+1}/{len(models_to_test)}] {config['model_name']}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        llm = get_llm(
            model_name=config['model_name'],
            temperature=config['temperature'],
            max_new_tokens=config['max_tokens']
        )
        retriever = get_retriever()
        
        # Evaluate based on mode
        if args.mode == 'vanilla':
            print("\nEvaluating Vanilla RAG...")
            results = evaluate_vanilla(llm, retriever, questions, scorer)
        else:  # verirag
            print("\nEvaluating VeriRAG...")
            agent = build_agent(llm, retriever)
            results = evaluate_verirag(agent, questions, scorer)
        
        metrics = compute_metrics(results)
        elapsed = time.time() - start_time
        
        print("\n" + "-"*80)
        print("RESULTS")
        print("-"*80)
        print(f"Faithfulness: {metrics['avg_faithfulness']:.3f} ({metrics['avg_faithfulness']*100:.1f}%)")
        print(f"Pass Rate:    {metrics['pass_rate']*100:.1f}%")
        print(f"Time:         {elapsed:.1f}s")
        
        # Save
        output = {
            "model": config['model_name'],
            "mode": args.mode,
            "metrics": metrics,
            "results": results,
            "elapsed_time": elapsed
        }
        
        output_path = OUTPUT_DIR / f"{args.mode}_{model_key}.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"✓ Saved to {output_path}\n")
        
        all_results[model_key] = metrics
        
        # Cleanup
        del llm
        if args.mode == 'verirag':
            del agent
        torch.cuda.empty_cache()
        gc.collect()
    
    # Summary
    print("\n" + "="*80)
    print(f"SUMMARY - {args.mode.upper()}")
    print("="*80)
    
    for model_key, metrics in all_results.items():
        print(f"{model_key:15s}: {metrics['avg_faithfulness']:.3f} ({metrics['avg_faithfulness']*100:.1f}%) - Pass: {metrics['pass_rate']*100:.0f}%")
    
    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
