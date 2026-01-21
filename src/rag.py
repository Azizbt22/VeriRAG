from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def get_retriever(k=4):
    index_path = Path("index")
    if not index_path.exists():
        raise FileNotFoundError("Index not found. Run scripts/build_index.py")
    print("Loading FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    print(f"✓ Retriever ready (k={k})")
    return retriever
