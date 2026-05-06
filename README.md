# Offline Agentic RAG System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)
![GraphRAG](https://img.shields.io/badge/GraphRAG-NetworkX-purple)
![Local_AI](https://img.shields.io/badge/AI-100%25_Offline-success)

A 100% local, privacy-first **Agentic Retrieval-Augmented Generation (CRAG)** system designed to process and query complex academic documents (Physics, Mathematics, and Computer Science). 

This project orchestrates local Large Language Models (LLMs) and embedding models through an intuitive **Streamlit Web Interface**. Evolving beyond standard RAG, it acts as an autonomous research agent capable of routing queries, executing Python code, searching the web, and reflecting on its own answers to prevent hallucinations.

## Key Features

1. **Adaptive Semantic Routing:** Powered by a local LLM, the system dynamically classifies user intent and routes the query to the appropriate tool (Local RAG, Python Execution, or Web Search).
2. **Self-Reflective CRAG (Corrective RAG):** Implements an internal grading loop where a "judge" LLM evaluates the generated response against the retrieved context. If hallucinations are detected, the system automatically deletes the output and regenerates a strictly grounded answer.
3. **Agentic Math & Code Execution:** Equipped with a secure Python REPL tool. When asked to solve equations or plot functions, the LLM writes, executes, and renders Python code (e.g., `matplotlib`, `scipy`) locally in real-time.
4. **GraphRAG & Hybrid Search:** Combines **Vector Semantic Search** (ChromaDB), **Lexical Search** (BM25), and a **Local Knowledge Graph** (NetworkX) to map complex theoretical relationships. Candidate documents are strictly re-scored using a local Cross-Encoder.
5. **Context-Aware Memory (Standalone Questions):** Automatically reformulates follow-up questions using the chat history, ensuring the retriever always searches with full semantic context during multi-turn conversations.
6. **Incremental Document Ingestion:** Optimized chunking pipeline featuring a state-cache (JSON). Adding new documents only processes the new files, saving massive amounts of CPU/GPU time.
7. **Web Search Fallback:** Integrates the Tavily API to fetch real-time data automatically if the local academic documents do not contain the answer.
8. **100% Offline & Private:** Core operations run entirely locally on consumer hardware (AMD/NVIDIA GPUs via LM Studio).

## Architecture & Tech Stack

* **Frontend:** `Streamlit` for UI, state management, and real-time output streaming.
* **Agentic Nodes:** Custom routing, rewriting, and hallucination-grading nodes.
* **Retrieval Pipeline:** Custom Hybrid Retriever (`BM25` + `ChromaDB`), Knowledge Graphs (`NetworkX`), and Re-Ranking (`Sentence-Transformers`).
* **LLM Inference:** `LM Studio` local server API (OpenAI drop-in replacement).
* **Orchestration:** `LangChain` ecosystem.

---

## Prerequisites

* Python 3.10 or higher.
* [LM Studio](https://lmstudio.ai/) installed and running in the background.
* The `lms` CLI tool enabled in LM Studio.
* *Note for GPU Acceleration:* Depending on your hardware (NVIDIA vs AMD), you may need specific PyTorch installations (CUDA or ROCm/DirectML) for optimal embedding generation.

## Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/santoh15/RAG-system-local.git](https://github.com/santoh15/RAG-system-local.git)
cd RAG-system-local
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up LM Studio**
* Open the **LM Studio** desktop application.
* Download your preferred text generation models (e.g., `Qwen-2.5-Coder`, `Llama-3`) and a vision/embedding model (e.g., `text-embedding-bge-m3`).
* The Python script will automatically manage the server startup and model loading via CLI (`lms load`).

## Usage

To start the graphical interface, run the following command in your terminal:

```bash
streamlit run app.py
```

**Workflow inside the App:**
1. **Configure Paths:** Enter the absolute paths for your Input directory (PDFs) and Output directory (Text and Vector Store).
2. **Process Documents:** Select a vision model and chunking method. The system will incrementally vectorize the documents and update the database.
3. **Build Knowledge Graph (Optional):** Extract academic entities and relationships to enable GraphRAG.
4. **Start Chatting:** Load the models and ask complex questions. Watch the agent decide whether to search your notes, execute Python scripts, or query the web!

## Project Structure

```text
├── app.py                         # Streamlit Main UI & Agent Orchestration
├── src/
│   ├── advanced_nodes.py          # Adaptive routing, standalone questions & self-reflection
│   ├── advanced_retriever.py      # Hybrid Search, GraphRAG & Cross-Encoder Re-Ranking
│   ├── agent_tools.py             # Python REPL execution & Tavily Web Search
│   ├── knowledge_graph.py         # NetworkX triplet extraction and graph builder
│   ├── chunking.py                # Semantic/Recursive Chunking with incremental caching
│   ├── chat_with_RAG.py           # Core LLM consultation and query translation
│   ├── chat_bot.py                # LM Studio server/CLI commands wrapper
│   ├── embedding.py               # Chroma vector store creation and loading
│   ├── pdf_image_txt_converter.py # Extract and convert PDFs and images to plain text
│   └── features.py                # Helper functions for image and PDF processing
├── requirements.txt               # Pinned Python dependencies
├── .gitignore
└── README.md
```

## Roadmap & Future Work

* **Interactive Graph Visualization:** Add an interactive UI component (e.g., `pyvis`) to let users visually explore the generated Knowledge Graph directly inside Streamlit.
* **Automated Pipeline Evaluation:** Integrate frameworks like RAGAS or TruLens to systematically measure context precision, recall, and answer faithfulness.
* **Multi-Agent Collaboration:** Split tasks into specialized sub-agents (e.g., a dedicated "Math Expert" and a "Theory Expert") that debate and compile a final answer.

## Author
**Santiago Huck** - B.Sc. in Physics
* [LinkedIn](https://www.linkedin.com/in/santiago-huck-621a02236)
* [GitHub](https://github.com/santoh15)

---
*Feel free to fork this project or submit pull requests!*