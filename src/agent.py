from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate

def build_agent(llm, retriever):
    prompt = ChatPromptTemplate.from_template("""You are a precise QA assistant.

Context:
{context}

Question: {question}

Instructions:
1. Check if context answers the question
2. If NO: Say "I cannot answer based on the provided context"
3. If YES: Answer using ONLY the context
4. Be specific and concise (2-5 sentences)

Answer:""")
    
    def agent(question: str) -> Dict[str, Any]:
        docs = retriever.invoke(question)
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        retrieval_trace = [{"chunk_id": i, "doc_id": doc.metadata.get("doc_id", f"chunk_{i}"), "preview": doc.page_content[:200] + "..."} for i, doc in enumerate(docs)]
        formatted = prompt.format(context=context, question=question)
        response = llm.invoke(formatted)
        answer = response.content if hasattr(response, 'content') else str(response)
        if "cannot answer" in answer.lower():
            verdict = "ABSTAIN"
        elif len(answer.split()) < 10:
            verdict = "FAIL"
        else:
            verdict = "PASS"
        return {"answer": answer.strip(), "verdict": verdict, "retrieval_trace": retrieval_trace, "plan": "VeriRAG verification"}
    return agent
