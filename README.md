# VeriRAG: Self-Verification for RAG

## What is VeriRAG?

VeriRAG is a simple approach to make RAG (Retrieval-Augmented Generation) systems more reliable. Instead of using multiple models or fine-tuning, it uses **a single carefully-designed prompt** that makes the model verify its own answers.

**Key idea:** The same model retrieves context AND verifies whether it can actually answer the question based on that context.

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
- ROUGE catches exact/near-exact matches
- BLEU measures precision across different n-gram sizes
- Semantic similarity captures paraphrasing (e.g., "ML model" ≈ "machine learning system")
- Word overlap provides fallback when embeddings unavailable

## Results

Tested on 24 questions (ML concepts + out-of-domain) with  different models:

| Model | Size | Vanilla RAG | VeriRAG | Improvement |
|-------|------|-------------|---------|-------------|
| **Qwen-2.5-1.5B** | 1.5B | 75.9% | **92.1%** | **+21.4%** |
| **Llama-3.2-3B** | 3.0B | 81.3% | **92.8%** | **+14.1%** |
| Llama-3.2-1B | 1.0B | 81.5% | 82.6% | +1.4% |
| Phi-2 | 2.7B | 81.9% | 69.7% | -15.0% |


**Key findings:**
-  Works best with instruction-tuned models (Llama, Qwen)
-  Perfect abstention on out-of-domain questions (100% accuracy)
-  Doesn't work with code-focused models (Phi-2)



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

### Run the Demo App

**Basic interface:**
```bash
streamlit run app.py
```

**Enhanced interface with metrics:**
```bash
streamlit run app_enhanced.py
```

Features:
- Live question answering
- Faithfulness scoring
- Side-by-side comparison (Vanilla vs VeriRAG)
- Retrieval trace visualization

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
) ```


Trade-offs:

More chunks (k=6-8): More comprehensive context, but slower and more noise
Fewer chunks (k=2-3): Faster, but might miss relevant info


### Add New Models

**Edit MODEL_CONFIGS in `evaluate_final.py`:**







