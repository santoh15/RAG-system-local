# Offline Academic RAG System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)
![Local_AI](https://img.shields.io/badge/AI-100%25_Offline-success)

A 100% local, privacy-first Retrieval-Augmented Generation (RAG) system designed to process and query complex academic documents (Physics, Mathematics, and Computer Science). 

This project orchestrates local Large Language Models (LLMs) and embedding models through an intuitive **Streamlit Web Interface**, creating a conversational AI assistant that grounds its answers strictly on a custom offline knowledge base, preventing hallucinations while keeping all data private.

## Key Features

* **Interactive Web UI:** Fully integrated with **Streamlit**, providing a clean interface to manage file paths, select models, and chat with your documents dynamically.
* **100% Offline & Private:** No API keys, no cloud dependencies, no data leaks. Everything runs locally on consumer hardware.
* **Real-time LLM Streaming:** The chat interface features word-by-word streaming responses (`yield`), providing a fast and fluid user experience similar to ChatGPT or Claude.
* **Semantic & Recursive Chunking:** Choose between character-based or semantic splitters (via `sentence-transformers`) directly from the UI, preserving the coherence of mathematical theorems, proofs, and complex academic context.
* **Optimized Memory Management:** The app manages VRAM intelligently via `LM Studio CLI` commands, allowing the concurrent coexistence of the embedding model (for fast context retrieval) and the conversational LLM.
* **Conversational Memory:** Implements a sliding window memory system to allow contextual follow-up questions without overflowing the LLM's context window.

## Architecture & Tech Stack

1. **Frontend:** `Streamlit` for UI and state management (`st.session_state`).
2. **Document Ingestion:** Reads local `.txt`, images or `.pdf` files.
3. **Embedding & Chunking:** `HuggingFaceEmbeddings` (BAAI/bge-m3) running natively.
4. **Vector Database:** `ChromaDB` for persistent, on-disk semantic search.
5. **LLM Inference:** `LM Studio` local server API (OpenAI drop-in replacement).
6. **Orchestration:** `LangChain` & `LangChain-HuggingFace`.

---

## Prerequisites

* Python 3.10 or higher.
* [LM Studio](https://lmstudio.ai/) installed and running in the background.
* The `lms` CLI tool enabled in LM Studio.
* At least 16GB of RAM (32GB recommended for running embeddings + LLM simultaneously).

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
* Ensure you have downloaded your preferred text generation models (e.g., `Qwen-30B-Coder`, `Llama-3`) and a vision/embedding model (e.g., `text-embedding-bge-m3`).
* The Python script will automatically manage the server startup and model loading via CLI (`lms load`).

## 🏃‍♂️ Usage

To start the graphical interface, run the following command in your terminal:

```bash
streamlit run app.py
```

**Workflow inside the App:**
1. **Configure Paths:** In the sidebar, enter the absolute paths for your Input directory (where your PDFs live) and Output directory (where text and ChromaDB will be saved).
2. **Process Documents:** If new files are detected, select a vision model and a chunking method, then click **Process**. The system will vectorize the documents and update the database.
3. **Start Chatting:** Select your conversational RAG model and click **Load Vector Store and Model**. 
4. Ask complex academic questions in the main chat window and watch the streaming response based on your documents!

## Project Structure

```text
├── app.py                         # Streamlit Main UI & Application Logic
├── src/
│   ├── chunking.py                # Semantic or recursive Chunking logic
│   ├── chat_with_RAG.py           # Streaming LLM consult and memory management
│   ├── chat_bot.py                # LM Studio server/CLI commands wrapper
│   ├── embedding.py               # Chroma vector store creation and loading
│   ├── pdf_image_txt_converter.py # Extract and convert PDFs and images to plain text
│   └── features.py                # Helper functions to process images and PDFs
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md
```
*(Note: Input and Output directories are now dynamically defined by the user through the Streamlit interface).*

## Roadmap & Future Work

* **Source Citations in UI:** Enhance the chat interface to visually display clickable references and snippets of the exact document chunks the LLM used to generate its answer.
* **Advanced Document Parsing:** Improve the extraction of complex mathematical formulas and tables from PDFs before embedding.
* **Multi-session Support:** Save and load previous chat histories natively within the Streamlit app.

## Author
**Santiago Huck** - B.Sc. in Physics
* [LinkedIn](https://www.linkedin.com/in/santiago-huck-621a02236)
* [GitHub](https://github.com/santoh15)

---
*Feel free to fork this project or submit pull requests!*