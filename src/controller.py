# src/controller.py

from typing import Dict, Any


class AgentController:
    """
    Controls an agent loop:
    Retrieve → Generate → Verify → Retry if needed

    This class is intentionally simple, explicit, and debuggable.
    """

    def __init__(
        self,
        retriever,
        generator,
        verifier,
        max_attempts: int = 2,
    ):
        self.retriever = retriever
        self.generator = generator
        self.verifier = verifier
        self.max_attempts = max_attempts

    def run(self, question: str) -> Dict[str, Any]:
        history = []
        current_question = question

        for attempt in range(self.max_attempts):
            # 1️⃣ Retrieve
            state = self.retriever({"question": current_question})

            # Safety check
            if "context" not in state:
                raise KeyError(
                    "Retriever must return a 'context' field. "
                    "Make sure you aggregate documents into text."
                )

            # 2️⃣ Generate
            answer = self.generator(
                {
                    "question": current_question,
                    "context": state["context"],
                }
            )

            # 3️⃣ Verify
            verdict = self.verifier(
                {
                    "context": state["context"],
                    "answer": answer,
                }
            )

            history.append(
                {
                    "attempt": attempt,
                    "question": current_question,
                    "verdict": verdict,
                    "retrieval_trace": state.get("retrieval_trace", []),
                }
            )

            # 4️⃣ Stop if grounded
            if "PASS" in verdict:
                return {
                    "answer": answer,
                    "verdict": verdict,
                    "history": history,
                    "status": "SUCCESS",
                }

            # 5️⃣ Feedback loop (very important)
            current_question = (
                current_question
                + "\n\nIMPORTANT: Refine the answer. "
                "Be strictly grounded in the retrieved documents. "
                "Avoid generic or pedagogical explanations."
            )

        # 6️⃣ Abstain if unreliable
        return {
            "answer": "I cannot answer reliably based on the provided documents.",
            "verdict": "FAIL",
            "history": history,
            "status": "FAILED",
        }
