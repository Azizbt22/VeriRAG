import streamlit as st
from typing import List

from src.models import get_llm
from src.rag import get_retriever
from src.agent import build_agent
from src.faithfulness_scorer import FaithfulnessScorer
from langchain_core.prompts import ChatPromptTemplate


# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VeriRAG Demo",
    layout="wide",
)

st.title("🔍 VeriRAG: Self-Verification for RAG")
st.caption("Compare Vanilla RAG vs VeriRAG on the same question")


# -----------------------------------------------------------------------------
# Sidebar configuration
# -----------------------------------------------------------------------------
st.sidebar.header("Configuration")

MODEL_NAME = st.sidebar.selectbox(
    "Model",
    [
        "meta-llama/Llama-3.2-3B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "microsoft/phi-2",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ],
    index=0,
)

TOP_K = st.sidebar.slider("Retrieved chunks (k)", 2, 8, 4)
USE_EMBEDDINGS = st.sidebar.checkbox("Use embedding-based faithfulness", value=True)


# -----------------------------------------------------------------------------
# Load components (cached)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_components(model_name: str, k: int):
    llm = get_llm(
        model_name=model_name,
        temperature=0.2,
        max_new_tokens=512,
    )
    retriever = get_retriever(k=k)
    agent = build_agent(llm, retriever)
    return llm, retriever, agent


@st.cache_resource
def load_scorer(use_embeddings: bool):
    return FaithfulnessScorer(use_embeddings=use_embeddings)


llm, retriever, agent = load_components(MODEL_NAME, TOP_K)
scorer = load_scorer(USE_EMBEDDINGS)


# -----------------------------------------------------------------------------
# Vanilla RAG helper
# -----------------------------------------------------------------------------
def run_vanilla_rag(question: str):
    prompt = ChatPromptTemplate.from_template(
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    formatted = prompt.format(context=context, question=question)
    response = llm.invoke(formatted)
    answer = response.content if hasattr(response, "content") else str(response)
    return answer, context, docs


# -----------------------------------------------------------------------------
# Main UI
# -----------------------------------------------------------------------------
question = st.text_input(
    "Ask a question",
    placeholder="e.g. What is overfitting in machine learning?",
)

run = st.button("Run")

if run and question.strip():
    col1, col2 = st.columns(2)

    # -------------------------
    # Vanilla RAG
    # -------------------------
    with col1:
        st.subheader("📄 Vanilla RAG")

        vanilla_answer, vanilla_context, vanilla_docs = run_vanilla_rag(question)
        vanilla_faith = scorer.compute_faithfulness_score(
            vanilla_answer, vanilla_context
        )

        st.markdown("**Answer**")
        st.write(vanilla_answer)

        st.markdown("**Faithfulness**")
        st.metric(
            "Score",
            f"{vanilla_faith['faithfulness_score']:.3f}",
        )

        with st.expander("Retrieved context"):
            for i, d in enumerate(vanilla_docs):
                st.markdown(f"**Chunk {i+1}**")
                st.write(d.page_content[:800] + "...")

    # -------------------------
    # VeriRAG
    # -------------------------
    with col2:
        st.subheader("✅ VeriRAG (with verification)")

        result = agent(question)
        context = "\n\n".join(
            chunk.get("preview", "") for chunk in result.get("retrieval_trace", [])
        )

        faith = scorer.compute_faithfulness_score(result["answer"], context)

        st.markdown("**Answer**")
        st.write(result["answer"])

        st.markdown("**Verdict**")
        st.write(result.get("verdict", "N/A"))

        st.markdown("**Faithfulness**")
        st.metric(
            "Score",
            f"{faith['faithfulness_score']:.3f}",
        )

        with st.expander("Retrieved context"):
            for i, chunk in enumerate(result.get("retrieval_trace", [])):
                st.markdown(f"**Chunk {i+1}**")
                st.write(chunk.get("preview", "")[:800] + "...")

    # -------------------------
    # Summary
    # -------------------------
    st.markdown("---")
    delta = faith["faithfulness_score"] - vanilla_faith["faithfulness_score"]

    st.subheader("📊 Comparison")
    st.write(
        f"""
        **Vanilla RAG:** {vanilla_faith['faithfulness_score']:.3f}  
        **VeriRAG:** {faith['faithfulness_score']:.3f}  
        **Δ Faithfulness:** {delta:+.3f}
        """
    )
