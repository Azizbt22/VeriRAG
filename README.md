# VeriRAG: Self-Verification for RAG

## Overview

**VeriRAG** is a lightweight approach to improving the *reliability* of Retrieval-Augmented Generation (RAG) systems without fine-tuning or multiple models.

Instead of introducing an external verifier, VeriRAG relies on a single instruction-tuned language model prompted to:
1. retrieve relevant context,
2. generate an answer,
3. explicitly verify whether the answer is supported by the retrieved context, or abstain otherwise.



---

## Motivation

Standard RAG systems often:
- hallucinate when context is weak or missing,
- answer confidently even when information is out-of-domain.

VeriRAG explicitly addresses this by forcing the model to **justify its answer against retrieved evidence** and abstain when grounding is insufficient.

---
**24 questions across 3 categories:**

| Category | Count | Examples | Purpose |
|----------|-------|----------|---------|
| **Easy ML Concepts** | 11 | Overfitting, neural networks, supervised learning | Test basic retrieval |
| **Medium Technical** | 8 | Transformer architecture, attention mechanism, training | Test deeper understanding |
| **Out-of-Domain** | 5 | Baking, sports, geography, plumbing | **Test abstention** (critical!) |

Out-of-domain questions are crucial for measuring whether the model can reliably say "I don't know" instead of hallucinating.

### Faithfulness Scoring

**Multi-metric approach** combining 5 complementary metrics:

```python
faithfulness_score = (
    0.35 * semantic_similarity +  # Highest weight - sentence embeddings
    0.25 * rouge_l +              # Longest common subsequence
    0.20 * rouge_2 +              # Bigram overlap
    0.10 * bleu +                 # N-gram precision (1-4 grams)
    0.10 * word_overlap           # Stemmed token matching
)
```

**Why multiple metrics?**
Semantic similarity captures paraphrasing
ROUGE / BLEU capture explicit grounding
Word overlap provides robustness when embeddings are noisy

This hybrid approach aims to avoid over-reliance on either lexical matching or embeddings alone.

## Results

Tested on 24 questions (ML concepts + out-of-domain) with  different models:

| Model | Size | Vanilla RAG | VeriRAG | Improvement |
|-------|------|-------------|---------|-------------|
| **Qwen-2.5-1.5B** | 1.5B | 75.9% | **92.1%** | **+21.4%** |
| **Llama-3.2-3B** | 3.0B | 81.3% | **92.8%** | **+14.1%** |
| Llama-3.2-1B | 1.0B | 81.5% | 82.6% | +1.4% |
| Phi-2 | 2.7B | 81.9% | 69.7% | -15.0% |


**Key findings:**
Key Findings

✅ VeriRAG significantly improves faithfulness for instruction-tuned models (LLaMA, Qwen)

✅ 100% correct abstention on out-of-domain questions

⚠️ VeriRAG degrades performance for extractive / code-focused models (Phi-2)



## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Prepare data
python scripts/preprocess_data.py
python scripts/build_index.py

# 3. Run evaluation
python evaluate_final.py --model llama-3b
```

## How to Use

### Run Evaluation

**Compare vanilla vs VeriRAG:**
For example :
```bash
python evaluate_final.py --model qwen-1.5b
```

### Use in Your Code

```python
from src.models import get_llm
from src.rag import get_retriever
from src.agent import build_agent

# Setup
llm = get_llm(model_name="meta-llama/Llama-3.2-3B-Instruct")
retriever = get_retriever(k=4)
agent = build_agent(llm, retriever)

# Ask questions
result = agent("What is overfitting?")
print(result['answer'])
print(result['verdict'])  # PASS, FAIL, or ABSTAIN
```

### Demo Application

```bash
streamlit run app_enhanced.py
```

Features:

-Live RAG vs VeriRAG comparison
-Retrieval trace visualization
-Faithfulness scoring
-Explicit abstention behavior

---

## Customize

### Use Your Own Data

**1. Add your documents to `data/raw/`:**
```bash
data/raw/
├── your_doc1.txt
├── your_doc2.txt
└── your_doc3.txt
```

**2. Preprocess and build index:**
```bash
python scripts/preprocess_data.py
python scripts/build_index.py
```

### Use Your Own Questions

**Create `data/eval/custom_questions.json`:**
```json
[
  {
    "id": "q1",
    "question": "Your question here?",
    "difficulty": "easy",
    "category": "your_domain"
  }
]
```
Adjust Retrieval
Edit src/rag.py:
``` python 
# Change number of retrieved chunks
retriever = vectorstore.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 6}  # Default: 4, try 6-8 for more context
)
```


Trade-offs:

More chunks (k=6-8): More comprehensive context, but slower and more noise
Fewer chunks (k=2-3): Faster, but might miss relevant info

## Limitations & Future Work

-Verification effectiveness is model-dependent

-Small models may require simplified verification prompts and external lightweight verifiers

**Future work** could explore:

-confidence-aware scoring

-adaptive verification thresholds








