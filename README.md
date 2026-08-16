# 🌐 NexusFin Intelligence Suite

An enterprise-grade, local AI semantic engine designed for deep financial analysis, SEC Form 10-K extraction, and multi-document vector retrieval. Built with **Streamlit**, **LangChain**, **ChromaDB**, and **Phi-3-mini (4-bit quantized)** running locally on GPU.

## 🚀 Key Features
* **Multi-Document Ingestion Hub:** Securely ingest, chunk, and embed multiple corporate SEC filings or annual reports simultaneously.
* **Granular Semantic Query Terminal:** Perform precise vector searches across complex financial tables and qualitative risk disclosures.
* **Strict Hallucination Guardrails:** Automatically triggers strict fallback prompts and conceals citations if exact data is absent from context.
* **GPU-Optimized Memory:** Features 4-bit quantization (`BitsAndBytes`) to fit state-of-the-art open-source LLMs locally within free-tier GPU constraints.

## 🛠️ Tech Stack
* **UI Framework:** Streamlit (Glassmorphism Dark Mode)
* **Orchestration:** LangChain
* **Vector Database:** ChromaDB
* **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
* **LLM Engine:** Microsoft Phi-3-mini-4k-instruct (4-bit NF4 quantized)

## ⚙️ Quick Start Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/vishwa-patel-1709/NexusFin-Intelligence-Suite.git](https://github.com/vishwa-patel-1709/NexusFin-Intelligence-Suite.git)
   cd NexusFin-Intelligence-Suite
