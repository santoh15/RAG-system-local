import streamlit as st
import os
import subprocess
from src.chunking import directory_chunk_to_json
from src.pdf_image_txt_converter import extract_text_from_pdf
from src.embedding import read_chunks_for_embedding, create_chroma_vector_store, load_base_vectorial
from src.chat_bot import start_server_lmstudio, close_server_lmstudio
from src.chat_whit_RAG import consult_llm_whith_memory

st.set_page_config(page_title="RAG Chatbot Local", layout="wide")


API_URL = "http://localhost:1234/v1/chat/completions"

# --- ESTADO DE LA SESIÓN ---
if "messages" not in st.session_state:
    prompt_sistema = """Sos un asistente. 
    Tu tarea es responder a la pregunta del usuario. 
    Primero, utilizá la información provista en el 'Contexto' de los apuntes como base principal. 
    Luego, podés expandir, explicar o dar ejemplos utilizando tu propio conocimiento general. 
    Si usás fórmulas o código, formatealos correctamente."""
    st.session_state.messages = [{"role": "system", "content": prompt_sistema}]

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


with st.sidebar:
    st.header("📂 Path configuration")
    
    
    path_in = st.text_input(
        "Input directory (PDFs):", 
        placeholder="Ex: C:\\documents\\pdfs",
        value=st.session_state.get('path_in', "")
    )
    
    path_out = st.text_input(
        "Output directory (.txt files/Vector Store):", 
        placeholder="Ex: C:\\projects\\rag_output",
        value=st.session_state.get('path_out', "")
    )

    
    if path_in and path_out:
        if os.path.exists(path_in):
            st.session_state.path_in = path_in
            st.success("✅ Valid dir_in path")
        else:
            st.error("❌ The input directory does not exist.")
            
        if os.path.exists(path_out):
            st.session_state.path_out = path_out
            st.success("✅ Valid output directory")
        else:
            # Si no existe, podrías ofrecer crearla
            if st.button("Create output directory"):
                os.makedirs(path_out, exist_ok=True)
                st.rerun()
    else:
        st.warning("⚠️ Please enter both paths to continue.")
    st.header("⚙️ System Configuration")
    
    st.subheader("1. Document Processing")
    vision_model = st.selectbox(
        "Vision Model:", 
        ['minicpm-v-4_5', 'pixtral-12b', 'meta-llama-3-8b-instruct', 'qwen2.5-coder-7b-instruct']
    )
    chunking_method = st.radio("Chunking Method:", ["1. Recursive", "2. Semantic"])
    chunk_choice = '1' if "Recursive" in chunking_method else '2'

    new_files = False
    
    if 'path_in' in st.session_state and 'path_out' in st.session_state:
        if os.path.exists(st.session_state.path_in):
            for name in os.listdir(st.session_state.path_in):
                if not os.path.isdir(os.path.join(st.session_state.path_in, name)):
                    base, _ = os.path.splitext(name)
                    # Revisa usando la ruta de salida de la UI
                    if not os.path.exists(os.path.join(st.session_state.path_out, f"{base}.txt")):
                        new_files = True
                        break
        
        if new_files:
            st.warning("⚠️ New files detected in the input directory that are not in the Database.")
            if st.button("Process and Add to Database"):
                with st.spinner("Extracting text from PDFs/Images... This may take a while."):
                    extract_text_from_pdf(st.session_state.path_in, st.session_state.path_out, API_URL, vision_model)
                with st.spinner("Chunking the text..."):
                    directory_chunk_to_json(st.session_state.path_out, chunk_choice)
                with st.spinner("Creating Embeddings and Database..."):
                    docs = read_chunks_for_embedding(st.session_state.path_out)
                    create_chroma_vector_store(docs, st.session_state.path_out, API_URL)
                st.success("✅ Database updated successfully!")
                st.rerun()
        else:
            st.success("✅ The database is up to date with the current files.")
    else:
        st.info("Waiting for valid paths to search for documents...")

    st.divider()
    
    st.subheader("2. Chat Configuration")
    chat_model = st.selectbox(
        "Model for chatting (RAG):", 
        ['meta-llama-3-8b-instruct', 'qwen2.5-coder-7b-instruct', 'minicpm-v-4_5']
    )
    
    if st.button("Load Vector Store and Model for RAG"):
        with st.spinner("Starting server and loading models..."):
            start_server_lmstudio(chat_model)
            subprocess.run(['lms', 'load', 'text-embedding-bge-m3'], shell=True) 
            st.session_state.vector_store = load_base_vectorial(API_URL, st.session_state.path_out)
            
        st.success("✅ RAG System ready for chatting!")

    if st.button("Shut Down LM Studio Server"):
        close_server_lmstudio()
        st.info("Server shut down.")


st.title("📚 RAG Assistant Local")


for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


if prompt := st.chat_input("Type your question here..."):
    if st.session_state.vector_store is None:
        st.error("⚠️ Please load the vector store and model from the sidebar first.")
    else:
        with st.chat_message("user"):
            st.markdown(prompt)
    
        with st.spinner("Searching in the documents..."):
            results = st.session_state.vector_store.similarity_search(prompt, k=3)
            context_made = "\n\n".join([doc.page_content for doc in results])
            
            user_message = f"""
            [Context]:{context_made}
            [Actual question]:{prompt}"""

        
        historial_temporal = st.session_state.messages.copy()
        historial_temporal.append({"role": "user", "content": user_message})

        
        with st.chat_message("assistant"):
            stream = consult_llm_whith_memory(historial_temporal, API_URL)
            final_answer = st.write_stream(stream)
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": final_answer})
        
        if len(st.session_state.messages) > 11:
            st.session_state.messages.pop(1) 
            st.session_state.messages.pop(1)