# src/run.py

from src.models import get_llm
from src.rag import get_retriever
from src.agent import (
    PLANNER_PROMPT,
    GENERATOR_PROMPT,
    VERIFIER_PROMPT,
)
from src.controller import AgentController


def main():
    # 1️⃣ Load core components
    llm = get_llm()
    retriever = get_retriever()

    # 2️⃣ Build runnable steps
    planner = PLANNER_PROMPT | llm
    generator = GENERATOR_PROMPT | llm
    verifier = VERIFIER_PROMPT | llm

    # 3️⃣ Wrap retriever so it returns proper state
    def retrieve(inputs):
        docs = retriever.invoke(inputs["question"])

        context = "\n\n".join(d.page_content for d in docs)

        return {
            "question": inputs["question"],
            "context": context,
            "retrieval_trace": [
                {
                    "chunk_id": d.metadata.get("chunk_id"),
                    "doc_id": d.metadata.get("doc_id"),
                    "preview": d.page_content[:200],
                }
                for d in docs
            ],
        }

    # 4️⃣ Create controller
    controller = AgentController(
        retriever=retrieve,
        generator=generator.invoke,
        verifier=verifier.invoke,
        max_attempts=2,
    )

    # 5️⃣ Run
    question = "Explain the transformer architecture and its key components."

    plan = planner.invoke({"question": question})
    result = controller.run(question)

    # 6️⃣ Display
    print("\n=== PLAN ===\n")
    print(plan)

    print("\n=== ANSWER ===\n")
    print(result["answer"])

    print("\n=== VERDICT ===\n")
    print(result["verdict"])

    print("\n=== TRACE ===\n")
    for step in result["history"]:
        print(step)


if __name__ == "__main__":
    main()
