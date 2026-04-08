# Offline Academic RAG System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)
![Local_AI](https://img.shields.io/badge/AI-100%25_Offline-success)

A 100% local, privacy-first Retrieval-Augmented Generation (RAG) system designed to process and query complex academic documents (Physics, Mathematics, and Computer Science). 

This project orchestrates local Large Language Models (LLMs) and embedding models to create a conversational AI assistant that grounds its answers strictly on a custom offline knowledge base, preventing hallucinations while keeping all data private.

## Key Features

* **100% Offline & Private:** No API keys, no cloud dependencies, no data leaks. Everything runs locally on consumer hardware.
* **Semantic Chunking:** Unlike traditional character-based splitters, this pipeline uses `sentence-transformers` to split texts semantically, preserving the coherence of mathematical theorems, proofs, and complex academic context.
* **Optimized Hardware Usage:** Leverages CPU/RAM for the embedding generation (BGE-M3) to free up VRAM, allowing heavier LLMs (like Qwen or Llama) to run concurrently on the GPU via LM Studio.
* **Conversational Memory:** Implements a sliding window memory system to allow contextual follow-up questions without overflowing the LLM's context window.
* **Hybrid Prompting:** Configurable system prompts to act as a strict document retriever or a hybrid academic tutor.

## Architecture & Tech Stack

1. **Document Ingestion:** Reads local `.txt`, images or `.pdf` files.
2. **Embedding & Chunking:** `HuggingFaceEmbeddings` (BAAI/bge-m3) running natively via `sentence-transformers`.
3. **Vector Database:** `ChromaDB` for persistent, on-disk semantic search.
4. **LLM Inference:** `LM Studio` local server API (OpenAI drop-in replacement).
5. **Orchestration:** `LangChain` & `LangChain-HuggingFace`.

---

## Prerequisites

* Python 3.10 or higher.
* [LM Studio](https://lmstudio.ai/) installed.
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
* Open LM Studio and navigate to the **Local Server** tab.
* Load your preferred LLM (e.g., `Qwen-30B-Coder` or `Llama-3`).
* Ensure the server is running on `http://localhost:1234/v1`.
* *(Note: The embedding model `BGE-M3` is handled natively by the Python script using your CPU/RAM, so you only need to load the text generation model in LM Studio).*

## 🏃‍♂️ Usage

**Step 1: Ingest and Vectorize Documents**
Place your academic notes (e.g., `.pdf` files) in the input directory. Then, run the data pipeline to semantically chunk the text and build the ChromaDB vector store:
```bash
python RAG.py
```
*This file ask you if want convert the files in the directory_in  in to text, next take this text files and create a persistent embeddings directory.*

**Step 2: Start the Chat Assistant**
Once the database is ready, launch the conversational interface in Visual Studio Code


## Project Structure

```text
├── src/
│   ├── chunking.py                # Semantic or recursive Chunking logic
│   ├── chat_with_RAG.py           # Conversational loop and memory management
│   ├── chat_bot.py                # Load the models and start or close the LM Studio server
│   ├── embedding.py               # Load the embedding model and create a vectorial database
│   ├── pdf_image_txt_converter.py # Extract and convert PDFs and images to plain text
│   └── features.py                # Helper functions to process images and PDFs
├── dir_in/                        # Directory for your raw image/PDF files (Not tracked by Git)
│   └── dir_out/                   # Directory for extracted .txt files
│       ├── embeddings/            # Directory to save the vector database (Created automatically)
│       └── chunks_for_embedding/  # Directory to save text chunks in JSON (Created automatically)
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md
```

## Roadmap & Future Work

While the core RAG architecture and local LLM orchestration are fully functional, the following features are planned for future releases to improve accessibility and user experience:

* **Graphical User Interface (GUI):** Transition from the current Command-Line Interface (CLI) to a responsive web-based chat interface (e.g., using **Streamlit** or **Gradio**).
* **Dynamic File Management:** Implement a drag-and-drop document upload system within the UI. This will allow users to seamlessly ingest new PDFs or text files on the fly and update the ChromaDB vector store without needing to manually edit directory paths in the source code.
* **Source Citations in UI:** Enhance the chat interface to visually display clickable references and snippets of the exact document chunks the LLM used to generate its answer.

## Author
**Santiago Huck** - B.Sc. in Physics
* [LinkedIn](https://www.linkedin.com/in/santiago-huck-621a02236)
* [GitHub](https://github.com/santoh15)

---
*Feel free to fork this project or submit pull requests!*