# app.py
import streamlit as st

from src.models import get_llm
from src.rag import get_retriever
from src.agent import build_agent


# --------------------
# App config
# --------------------
st.set_page_config(
    page_title="VeriRAG Demo",
    layout="wide",
)

st.title("🔍 VeriRAG — Verified Retrieval-Augmented Generation")
st.caption(
    "Agentic RAG system with grounding verification and retrieval trace."
)

# --------------------
# Lazy load components
# --------------------
@st.cache_resource
def load_agent():
    llm = get_llm()
    retriever = get_retriever()
    agent = build_agent(llm, retriever)
    return agent

agent = load_agent()

# --------------------
# Input
# --------------------
question = st.text_area(
    "Ask a technical question:",
    value="Explain the transformer architecture and its key components.",
    height=100,
)

run_btn = st.button("Run VeriRAG")

# --------------------
# Run agent
# --------------------
if run_btn:
    with st.spinner("Running agent pipeline..."):
        result = agent(question)

    # --------------------
    # Output
    # --------------------
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("✅ Final Answer")
        st.write(result["answer"])

    with col2:
        st.subheader("🧪 Verifier Verdict")
        verdict = result["verdict"]
        if "PASS" in verdict:
            st.success("PASS")
        else:
            st.error(verdict)

    # --------------------
    # Retrieval trace (if available)
    # --------------------
    st.subheader("📚 Retrieval Trace")

    if "retrieval_trace" in result:
        for i, doc in enumerate(result["retrieval_trace"], 1):
            with st.expander(f"Chunk {i} — {doc.get('doc_id')}"):
                st.code(doc.get("preview", ""), language="text")
    else:
        st.info("No retrieval trace available.")

    # --------------------
    # Plan (optional but nice)
    # --------------------
    st.subheader("🧠 Planner Output")
    st.write(result["plan"])