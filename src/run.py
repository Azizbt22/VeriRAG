# src/run.py

from src.models import get_llm
from src.rag import get_retriever
from src.agent import build_agent


def main():
    llm = get_llm()
    retriever = get_retriever()

    agent = build_agent(llm, retriever)

    question = "Explain the transformer architecture and its key components."

    result = agent(question)

    print("\n=== PLAN ===\n")
    print(result["plan"])

    print("\n=== ANSWER ===\n")
    print(result["answer"])

    print("\n=== VERDICT ===\n")
    print(result["verdict"])


if __name__ == "__main__":
    main()



