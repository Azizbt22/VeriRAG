# src/run.py

from src.models import get_llm
from src.rag import load_retriever


def main():
    # 1. Load components
    llm = get_llm()
    retriever = load_retriever(k=4)

    # 2. User query
    query = "Explain what a transformer is in simple terms."

    # 3. Retrieve documents
    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)

    # 4. Build prompt
    prompt = f"""
You are an expert AI assistant.
Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

    # 5. Call LLM
    response = llm.invoke(prompt)

    print("\n=== ANSWER ===\n")
    print(response)


if __name__ == "__main__":
    main()
