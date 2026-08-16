import streamlit as st
import tempfile
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline, ChatHuggingFace
from langchain_community.vectorstores import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Advanced UI Configuration
st.set_page_config(page_title="NexusFin Intelligence Suite", layout="wide", page_icon="🌐")

# 2. Modern Glassmorphism & Tight Spacing CSS
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #030712 0%, #0f172a 50%, #1e1b4b 100%);
        color: #ffffff;
    }
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label, [data-testid="stSidebar"] div {
        color: #f1f5f9 !important;
    }
    /* Completely Darken and Fix File Uploader Box & Buttons */
    [data-testid="stFileUploader"] {
        background-color: #0f172a !important;
        border: 1px dashed rgba(56, 189, 248, 0.4) !important;
        border-radius: 12px;
        padding: 12px;
    }
    [data-testid="stFileUploader"] section {
        background-color: #090d16 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] small, [data-testid="stFileUploader"] div, [data-testid="stFileUploader"] label {
        color: #ffffff !important;
    }
    /* Force upload/browse button to have solid dark background with bright text */
    [data-testid="stFileUploader"] button {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid #38bdf8 !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background-color: #334155 !important;
        color: #38bdf8 !important;
    }
    .hero-banner {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        padding: 20px 30px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.5);
        margin-bottom: 15px;
        margin-top: 5px;
    }
    .metric-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(12px);
        padding: 16px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        margin-bottom: 15px;
    }
    .cyber-badge {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: #ffffff;
        padding: 5px 12px;
        border-radius: 30px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .stTextInput input {
        background-color: rgba(15, 23, 42, 0.8) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        padding: 12px !important;
    }
    p, span, label, div {
        color: #f1f5f9;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Session State Initialization
if "uploaded_files_memory" not in st.session_state:
    st.session_state.uploaded_files_memory = []
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# 4. Cache Models
@st.cache_resource
def load_models():
    import torch
    from transformers import BitsAndBytesConfig
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = HuggingFacePipeline.from_model_id(
        model_id="microsoft/Phi-3-mini-4k-instruct",
        task="text-generation",
        device=0,
        model_kwargs={"quantization_config": quantization_config},
        pipeline_kwargs={
            "max_new_tokens": 250,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False 
        },
    )
    chat_model = ChatHuggingFace(llm=llm)
    return embeddings, chat_model

with st.spinner("🚀 Initializing NexusFin neural vector pipeline..."):
    embeddings, chat_model = load_models()

# 5. Sidebar Memory & Analytics Vault
with st.sidebar:
    st.markdown("### 🌐 NexusFin Vault")
    st.caption("Active Secure Session Memory")
    st.markdown("---")
    st.markdown("##### **Ingested Documents:**")
    
    if st.session_state.uploaded_files_memory:
        for fname in st.session_state.uploaded_files_memory:
            st.markdown(f"📄 `{fname}`")
    else:
        st.caption("No corporate filings indexed yet.")
        
    st.markdown("---")
    st.subheader("📊 System Telemetry")
    st.metric("Total Executed Queries", len(st.session_state.query_history))
    st.metric("Neural Engine Core", "Phi-3 4-bit", "GPU Optimized")

# 6. Transformed Hero Header
st.markdown("""
    <div class="hero-banner">
        <h1 style="color: #ffffff; margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em;">🌐 NexusFin Intelligence Suite</h1>
        <p style="color: #94a3b8; margin-top: 6px; font-size: 0.95rem;">Transformative AI-Powered Semantic Analytics & Multi-Document Financial Extraction Engine</p>
    </div>
""", unsafe_allow_html=True)

# 7. Corporate Filing Ingestion Hub (Reduced Spacing)
st.markdown("### 📥 Corporate Filing Ingestion Hub")
st.markdown("Upload SEC Form 10-Ks, annual reports, or quarterly earnings statements to populate the secure vector database.")

uploaded_files = st.file_uploader("Drop target financial documents here", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if st.button("Process & Compile Database", type="primary", use_container_width=True):
        with st.spinner("🔄 Parsing structures, chunking data, and embedding vectors..."):
            all_splits = []
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.uploaded_files_memory:
                    st.session_state.uploaded_files_memory.append(uploaded_file.name)
                    
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                splits = text_splitter.split_documents(docs)
                all_splits.extend(splits)
            
            st.session_state.vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
            st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
            
        st.success(f"✨ Successfully vectorized {len(uploaded_files)} file(s)! You can now execute queries below.")

# Reduced divider spacing
st.markdown("<div style='margin: 10px 0;'></div>", unsafe_allow_html=True)

# 8. Integrated Semantic Query Terminal
st.markdown("### 💬 Semantic Query Terminal")

if st.session_state.retriever is None:
    st.info("💡 **Awaiting Ingestion:** Please upload and compile financial PDFs above to unlock the query terminal.")
else:
    col_q1, col_q2 = st.columns([4, 1])
    with col_q1:
        query = st.text_input("Enter precise financial query or metric request:", placeholder="e.g., What was total revenue growth YoY?")
    with col_q2:
        st.write("")
        st.write("")
        submit_btn = st.button("Execute Query", use_container_width=True, type="primary")

    if submit_btn and query:
        if query not in st.session_state.query_history:
            st.session_state.query_history.append(query)

    if st.session_state.query_history:
        st.markdown("<div style='margin: 10px 0;'></div>", unsafe_allow_html=True)
        for idx, q in enumerate(st.session_state.query_history[::-1]):
            query_id = len(st.session_state.query_history) - idx
            st.markdown(f"""
                <div class="metric-card">
                    <span class="cyber-badge">Query Node #{query_id}</span>
                    <h3 style="color: #ffffff; margin-top: 10px; margin-bottom: 12px; font-size: 1.15rem; font-weight: 600;">⚡ {q}</h3>
            """, unsafe_allow_html=True)
            
            with st.spinner("🔍 Executing multi-vector similarity scan & neural extraction..."):
                system_prompt = (
                    "You are an elite financial data extractor. Use ONLY the provided context. "
                    "Analyze tables accurately. If the information is not in the context, output: 'I do not have content like this in the document.'\n\n"
                    "Context: {context}"
                )
                prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
                
                qa_chain = create_stuff_documents_chain(chat_model, prompt)
                rag_chain = create_retrieval_chain(st.session_state.retriever, qa_chain)
                
                response = rag_chain.invoke({"input": q})
                clean_answer = re.sub(r"<\|.*?\|>", "", response["answer"]).strip()
                
                is_invalid = "I do not have content like this" in clean_answer or "I cannot find" in clean_answer
                
                if is_invalid:
                    st.error("⚠️ I do not have content like this in the document.")
                else:
                    st.success(clean_answer)
                    with st.expander("🔍 View Verified Source Context & Page Citations"):
                        for i, doc in enumerate(response["context"]):
                            st.markdown(f"**Source {i+1} (Page {doc.metadata.get('page', 'Unknown')}):**")
                            st.caption(doc.page_content)
                            
            st.markdown("</div>", unsafe_allow_html=True)
