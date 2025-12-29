# src/agent.py

from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


# ---------- PROMPTS ----------

PLANNER_PROMPT = ChatPromptTemplate.from_template(
    """
You are an expert AI researcher.

Given the user question below, produce a short plan describing:
- what information is required
- what concepts must be explained
- what level of technical depth is expected

User question:
{question}

Return a concise plan (3–5 bullet points).
"""
)

GENERATOR_PROMPT = ChatPromptTemplate.from_template(
    """
You are a graduate-level AI assistant.
Answer the question using ONLY the provided context.
Be technical, concise, and precise.

Context:
{context}

Question:
{question}

Draft answer:
"""
)

VERIFIER_PROMPT = ChatPromptTemplate.from_template(
    """
You are a strict verifier.

Given:
- the context
- the generated answer

Check whether:
1. The answer is grounded in the context
2. The answer is technically rigorous
3. The answer avoids oversimplified or childish explanations

Return one of:
- PASS
- FAIL: <short reason>

Context:
{context}

Answer:
{answer}
"""
)

REFINER_PROMPT = ChatPromptTemplate.from_template(
    """
You are an expert editor.

Improve the answer below by:
- increasing technical precision
- removing vague or pedagogical language
- ensuring it is grounded in the context

Context:
{context}

Original answer:
{answer}

Refined answer:
"""
)


# ---------- AGENT PIPELINE ----------

def build_agent(llm, retriever):
    """
    Build an explicit agentic RAG pipeline.
    """

    # 1️⃣ Planner
    planner = PLANNER_PROMPT | llm

    # 2️⃣ Retriever
    def retrieve(inputs: Dict[str, Any]) -> Dict[str, Any]:
        docs = retriever.invoke(inputs["question"])
        context = "\n\n".join(d.page_content for d in docs)
        return {**inputs, "context": context}

    retrieve_step = RunnableLambda(retrieve)

    # 3️⃣ Generator
    generator = GENERATOR_PROMPT | llm

    # 4️⃣ Verifier
    verifier = VERIFIER_PROMPT | llm

    # 5️⃣ Refiner
    refiner = REFINER_PROMPT | llm

    # ---------- ORCHESTRATION ----------

    def agent_run(question: str) -> Dict[str, Any]:
        # Plan
        plan = planner.invoke({"question": question})

        # Retrieve
        state = retrieve_step.invoke({"question": question})

        # Generate
        draft = generator.invoke(
            {
                "question": question,
                "context": state["context"],
            }
        )

        # Verify
        verdict = verifier.invoke(
            {
                "context": state["context"],
                "answer": draft,
            }
        )

        # Refine if needed
        if "FAIL" in verdict:
            final_answer = refiner.invoke(
                {
                    "context": state["context"],
                    "answer": draft,
                }
            )
        else:
            final_answer = draft

        return {
            "plan": plan,
            "answer": final_answer,
            "verdict": verdict,
        }

    return agent_run
