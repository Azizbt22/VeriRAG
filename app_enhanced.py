# app_enhanced.py
"""
Enhanced Streamlit UI for VeriRAG with:
- Side-by-side model comparison
- Detailed verification breakdown
- Interactive retrieval exploration
- Evaluation metrics visualization
"""

import streamlit as st
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

from src.models import get_llm
from src.rag import get_retriever
from src.agent import build_agent
from src.evaluate_enhanced import VeriRAGEvaluator


# --------------------
# Configuration
# --------------------
st.set_page_config(
    page_title="VeriRAG - Verified RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .pass-verdict {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 5px;
        padding: 10px;
        font-weight: bold;
        color: #155724;
    }
    .fail-verdict {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 5px;
        padding: 10px;
        font-weight: bold;
        color: #721c24;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# --------------------
# Sidebar Configuration
# --------------------
st.sidebar.title("⚙️ Configuration")

# Model selection
model_options = {
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B-Instruct",
    "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.2",
    # Add more models as available
}

selected_model = st.sidebar.selectbox(
    "Select Model",
    options=list(model_options.keys()),
    index=0
)

# RAG parameters
k_docs = st.sidebar.slider("Number of documents to retrieve", 2, 8, 4)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
max_tokens = st.sidebar.slider("Max tokens", 256, 1024, 512, 64)

# Verification mode
verification_mode = st.sidebar.selectbox(
    "Verification Mode",
    ["Strict (PASS/FAIL)", "Detailed (with scores)", "Off (vanilla RAG)"]
)

show_advanced = st.sidebar.checkbox("Show advanced metrics", value=False)


# --------------------
# Load Components
# --------------------
@st.cache_resource
def load_agent(model_name: str, k: int, temp: float, max_tok: int):
    """Load agent with specified configuration."""
    llm = get_llm(
        model_name=model_options[model_name],
        temperature=temp,
        max_new_tokens=max_tok
    )
    retriever = get_retriever(k=k)
    agent = build_agent(llm, retriever)
    evaluator = VeriRAGEvaluator(llm_as_judge=llm)
    return agent, llm, evaluator


# --------------------
# Main Interface
# --------------------
st.title("🔍 VeriRAG: Verification-Aware Retrieval-Augmented Generation")
st.markdown("""
An **agentic RAG system** with explicit verification to ensure answer faithfulness.  
Ask technical questions about ML/AI and see how the system retrieves, generates, and verifies answers.
""")

# Tabs for different features
tab1, tab2, tab3 = st.tabs([
    "💬 Interactive Query",
    "📊 Batch Evaluation",
    "📈 System Analysis"
])


# --------------------
# TAB 1: Interactive Query
# --------------------
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Ask a Question")
        question = st.text_area(
            "Enter your question:",
            value="Explain the transformer architecture and its key components.",
            height=100,
            help="Ask technical questions about machine learning, neural networks, etc."
        )
        
        # Example questions
        with st.expander("📝 Example Questions"):
            examples = [
                "What is overfitting in machine learning?",
                "Explain the attention mechanism in transformers.",
                "How does backpropagation work?",
                "What is the Turing Test?",
                "Compare CNNs and transformers for image processing."
            ]
            for ex in examples:
                if st.button(ex, key=f"ex_{ex}"):
                    question = ex
    
    with col2:
        st.subheader("🎯 Quick Actions")
        run_btn = st.button("🚀 Run VeriRAG", type="primary", use_container_width=True)
        compare_btn = st.button("⚖️ Compare with Vanilla RAG", use_container_width=True)
        clear_btn = st.button("🗑️ Clear Results", use_container_width=True)
    
    # Run query
    if run_btn:
        with st.spinner("🔄 Processing query through VeriRAG pipeline..."):
            agent, llm, evaluator = load_agent(
                selected_model, k_docs, temperature, max_tokens
            )
            result = agent(question)
        
        st.success("✅ Processing complete!")
        
        # Display results in organized sections
        st.markdown("---")
        
        # Answer section
        st.subheader("📝 Generated Answer")
        st.markdown(f"**Answer:** {result['answer']}")
        
        # Verification section
        col_v1, col_v2 = st.columns([1, 2])
        
        with col_v1:
            st.subheader("🔐 Verification")
            verdict = result.get("verdict", "")
            if "PASS" in verdict:
                st.markdown('<div class="pass-verdict">✅ VERIFIED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="fail-verdict">❌ NOT VERIFIED</div>', unsafe_allow_html=True)
        
        with col_v2:
            st.subheader("📋 Verdict Details")
            st.text(verdict)
        
        # Advanced metrics
        if show_advanced:
            st.subheader("📊 Detailed Metrics")
            
            # Get context from retrieval
            context = "\n".join([
                doc.get("preview", "")
                for doc in result.get("retrieval_trace", [])
            ])
            
            # Compute faithfulness
            faith_metrics = evaluator.evaluate_faithfulness(
                result["answer"],
                context,
                llm
            )
            
            # Display metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            with col_m1:
                st.metric(
                    "Faithfulness Score",
                    f"{faith_metrics['faithfulness_score']:.2%}"
                )
            
            with col_m2:
                st.metric(
                    "Total Claims",
                    faith_metrics['total_claims']
                )
            
            with col_m3:
                st.metric(
                    "Supported Claims",
                    faith_metrics['supported_claims']
                )
            
            with col_m4:
                st.metric(
                    "Unsupported Claims",
                    len(faith_metrics['unsupported_claims'])
                )
            
            # Show unsupported claims if any
            if faith_metrics['unsupported_claims']:
                with st.expander("⚠️ View Unsupported Claims"):
                    for i, claim in enumerate(faith_metrics['unsupported_claims'], 1):
                        st.markdown(f"{i}. {claim}")
        
        # Retrieval trace
        st.subheader("📚 Retrieved Documents")
        
        if "retrieval_trace" in result:
            for i, doc in enumerate(result["retrieval_trace"], 1):
                with st.expander(
                    f"📄 Document {i}: {doc.get('doc_id', 'Unknown')} (Chunk {doc.get('chunk_id', '?')})"
                ):
                    st.code(doc.get("preview", "No preview available"), language="text")
        else:
            st.info("No retrieval trace available.")
        
        # Planning output
        if "plan" in result:
            with st.expander("🧠 Planner Output"):
                st.markdown(result["plan"])


# --------------------
# TAB 2: Batch Evaluation
# --------------------
with tab2:
    st.subheader("📊 Batch Evaluation on Test Set")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        Run the system on a predefined set of questions to evaluate performance.
        This will compute aggregate metrics including pass rate and faithfulness.
        """)
    
    with col2:
        eval_btn = st.button("🎯 Run Evaluation", type="primary")
    
    if eval_btn:
        with st.spinner("Running evaluation on test set..."):
            # Load test questions
            questions_path = Path("data/eval/questions_extended.json")
            if questions_path.exists():
                questions = json.loads(questions_path.read_text())
            else:
                questions = json.loads(Path("data/eval/questions.json").read_text())
            
            agent, llm, evaluator = load_agent(
                selected_model, k_docs, temperature, max_tokens
            )
            
            # Run evaluation
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, q in enumerate(questions):
                status_text.text(f"Evaluating: {q['id']}")
                result = agent(q["question"])
                
                # Get verdict
                verdict = result.get("verdict", "")
                verdict_label = "PASS" if "PASS" in verdict else "FAIL"
                
                results.append({
                    "id": q["id"],
                    "question": q["question"],
                    "verdict": verdict_label,
                    "answer_length": len(result["answer"].split())
                })
                
                progress_bar.progress((i + 1) / len(questions))
            
            status_text.text("Evaluation complete!")
            progress_bar.empty()
        
        # Display results
        st.success(f"✅ Evaluated {len(results)} questions")
        
        # Aggregate metrics
        pass_count = sum(1 for r in results if r["verdict"] == "PASS")
        pass_rate = pass_count / len(results)
        
        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            st.metric("Pass Rate", f"{pass_rate:.1%}")
        
        with col_r2:
            st.metric("Total Questions", len(results))
        
        with col_r3:
            avg_length = sum(r["answer_length"] for r in results) / len(results)
            st.metric("Avg Answer Length", f"{avg_length:.0f} words")
        
        # Results table
        st.subheader("Detailed Results")
        st.dataframe(results, use_container_width=True)
        
        # Visualization
        st.subheader("Results Visualization")
        
        # Pass/Fail distribution
        fig = px.pie(
            values=[pass_count, len(results) - pass_count],
            names=["PASS", "FAIL"],
            title="Verification Verdict Distribution",
            color_discrete_map={"PASS": "#28a745", "FAIL": "#dc3545"}
        )
        st.plotly_chart(fig, use_container_width=True)


# --------------------
# TAB 3: System Analysis
# --------------------
with tab3:
    st.subheader("📈 System Performance Analysis")
    
    st.markdown("""
    Load and visualize evaluation results from previous runs.
    Compare different configurations and analyze system behavior.
    """)
    
    # Load existing results
    results_files = list(Path("runs").glob("*.json"))
    
    if results_files:
        selected_file = st.selectbox(
            "Select results file",
            options=[f.name for f in results_files]
        )
        
        if st.button("📂 Load and Analyze"):
            results_path = Path("runs") / selected_file
            data = json.loads(results_path.read_text())
            
            st.json(data, expanded=False)
            
            # If it's model comparison results
            if "metrics" in data and isinstance(data.get("results"), list):
                st.subheader("Performance Metrics")
                
                metrics = data["metrics"]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Pass Rate", f"{metrics.get('pass_rate', 0):.1%}")
                
                with col2:
                    st.metric(
                        "Avg Chunks Retrieved",
                        f"{metrics.get('average_chunks_retrieved', 0):.1f}"
                    )
                
                with col3:
                    st.metric("Total Questions", metrics.get('total_questions', 0))
    else:
        st.info("No evaluation results found. Run an evaluation first!")


# --------------------
# Footer
# --------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>VeriRAG: A Verification-Aware RAG System | Built with Streamlit</p>
    <p>Configure settings in the sidebar to customize the system behavior.</p>
</div>
""", unsafe_allow_html=True)
