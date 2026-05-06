# Offline Academic RAG System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)
![Local_AI](https://img.shields.io/badge/AI-100%25_Offline-success)

A 100% local, privacy-first Advanced Retrieval-Augmented Generation (RAG) system designed to process and query complex academic documents (Physics, Mathematics, and Computer Science). 

This project orchestrates local Large Language Models (LLMs) and embedding models through an intuitive **Streamlit Web Interface**, creating a conversational AI assistant that grounds its answers strictly on a custom offline knowledge base, preventing hallucinations while keeping all data private.

## Key Features

* **Interactive Web UI:** Fully integrated with **Streamlit**, providing a clean interface to manage file paths, select models, and chat with your documents dynamically.
* **100% Offline & Private:** No API keys, no cloud dependencies, no data leaks. Everything runs locally on consumer hardware.
* **Advanced Hybrid Search Pipeline:** Combines **Vector Semantic Search** (ChromaDB) with **Lexical Search** (BM25) to ensure exact keyword matching for mathematical notation, author names, and specific jargon alongside deep contextual understanding.
* **Cross-Encoder Re-Ranking:** Implements a native multi-stage retrieval pipeline. The system fetches dozens of document candidates and strictly re-scores them using a local `HuggingFaceCrossEncoder` model, ensuring the LLM only receives the most mathematically and conceptually accurate context.
* **Automated Query Translation:** Seamlessly bridges the language gap. Ask questions in your native language (e.g., Spanish); the system automatically translates the query to English in the background to maximize retrieval precision over foreign academic literature, before delivering the final answer back in your language.
* **Real-time LLM Streaming:** The chat interface features word-by-word streaming responses (`yield`), providing a fast and fluid user experience.
* **Semantic & Recursive Chunking:** Choose between character-based or semantic splitters directly from the UI, preserving the coherence of mathematical theorems, proofs, and complex academic context.
* **Conversational Memory:** Implements a sliding window memory system to allow contextual follow-up questions without overflowing the LLM's context window.

## Architecture & Tech Stack

1. **Frontend:** `Streamlit` for UI and state management (`st.session_state`).
2. **Retrieval Pipeline:** Custom-built Hybrid Retriever (`BM25` + `ChromaDB`) and `Sentence-Transformers` (Cross-Encoders).
3. **Vector Database:** `ChromaDB` for persistent, on-disk semantic search.
4. **LLM Inference:** `LM Studio` local server API (OpenAI drop-in replacement).
5. **Orchestration:** `LangChain` ecosystem.

---

## Prerequisites

* Python 3.10 or higher.
* [LM Studio](https://lmstudio.ai/) installed and running in the background.
* The `lms` CLI tool enabled in LM Studio.
* At least 16GB of RAM (32GB recommended for running Cross-Encoders + Embeddings + LLM simultaneously).

## Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/santoh15/RAG-system-local.git](https://github.com/santoh15/RAG-system-local.git)
cd RAG-system-local
```

**2. Install dependencies**
Ensure your environment has the latest packages installed:
```bash
pip install -r requirements.txt
pip install -U langchain langchain-community langchain-core rank_bm25 sentence-transformers torchvision
```

**3. Set up LM Studio**
* Open the **LM Studio** desktop application.
* Ensure you have downloaded your preferred text generation models (e.g., `Qwen-2.5-Coder`, `Llama-3`) and a vision/embedding model (e.g., `text-embedding-bge-m3`).
* The Python script will automatically manage the server startup and model loading via CLI (`lms load`).

## 🏃‍♂️ Usage

To start the graphical interface, run the following command in your terminal:

```bash
streamlit run app.py
```

**Workflow inside the App:**
1. **Configure Paths:** In the sidebar, enter the absolute paths for your Input directory (where your PDFs live) and Output directory (where text and ChromaDB will be saved).
2. **Process Documents:** If new files are detected, select a vision model and a chunking method, then click **Process**. The system will vectorize the documents and update the database.
3. **Start Chatting:** Select your conversational RAG model and click **Load Vector Store and Model**. The system will load ChromaDB, initialize the BM25 index, and spin up the Cross-Encoder.
4. Ask complex academic questions in the main chat window and watch the streaming response based on your documents!

## Project Structure

```text
├── app.py                         # Streamlit Main UI & Application Logic
├── src/
│   ├── advanced_retriever.py      # Custom Hybrid Search & Cross-Encoder Re-Ranking
│   ├── chunking.py                # Semantic or recursive Chunking logic
│   ├── chat_with_RAG.py           # Streaming LLM consult, memory and query translation
│   ├── chat_bot.py                # LM Studio server/CLI commands wrapper
│   ├── embedding.py               # Chroma vector store creation and loading
│   ├── pdf_image_txt_converter.py # Extract and convert PDFs and images to plain text
│   └── features.py                # Helper functions to process images and PDFs
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md
```

## Roadmap & Future Work

* **Source Citations in UI:** Enhance the chat interface to visually display clickable references and snippets of the exact document chunks the LLM used to generate its answer.
* **Automated Pipeline Evaluation:** Integrate frameworks like RAGAS or TruLens to systematically measure context precision, recall, and answer faithfulness (MLOps integration).
* **Self-Reflective RAG (CRAG):** Implement a local evaluation loop where the LLM grades the retrieved context and automatically triggers query refinement if the information is insufficient.
* **Agentic Math Execution:** Equip the LLM with a local Python REPL tool to dynamically compute equations, plot functions, and verify mathematical claims found in the documents.
* **GraphRAG Integration:** Extract entities and relationships from academic papers to build a local Knowledge Graph, complementing the vector search for complex multi-hop reasoning.
* **Academic Export Tools:** Add functionality to export chat sessions directly to LaTeX/Markdown formats and automatically generate BibTeX citations for the referenced chunks.

## Author
**Santiago Huck** - B.Sc. in Physics
* [LinkedIn](https://www.linkedin.com/in/santiago-huck-621a02236)
* [GitHub](https://github.com/santoh15)

---
*Feel free to fork this project or submit pull requests!*