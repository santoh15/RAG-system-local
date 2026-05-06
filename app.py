import streamlit as st
import os
import subprocess
import re

from src.advanced_nodes import route_query, rewrite_query, grade_hallucination, get_standalone_question
from src.knowledge_graph import build_knowledge_graph
from src.chunking import directory_chunk_to_json
from src.pdf_image_txt_converter import extract_text_from_pdf
from src.chat_bot import start_server_lmstudio, close_server_lmstudio, get_local_models
from src.embedding import read_chunks_for_embedding, create_chroma_vector_store, load_base_vectorial
from src.chat_whit_RAG import consult_llm_whith_memory, translate_query_to_english, grade_retrieved_context
from src.advanced_retriever import build_advanced_retriever
from src.agent_tools import execute_python_code, web_search

st.set_page_config(page_title="RAG Chatbot Local", layout="wide")

API_URL = "http://localhost:1234/v1/chat/completions"


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
            if st.button("Create output directory"):
                os.makedirs(path_out, exist_ok=True)
                st.rerun()
    else:
        st.warning("⚠️ Please enter both paths to continue.")
        
    st.header("⚙️ System Configuration")
    available_models = get_local_models()

    st.subheader("1. Document Processing")
    vision_model = st.selectbox(
    "Vision Model (Select a multimodal model):", 
    available_models)

    chunking_method = st.radio("Chunking Method:", ["1. Recursive", "2. Semantic"])
    chunk_choice = '1' if "Recursive" in chunking_method else '2'

    new_files = False
    
    if 'path_in' in st.session_state and 'path_out' in st.session_state:
        if os.path.exists(st.session_state.path_in):
            for name in os.listdir(st.session_state.path_in):
                if not os.path.isdir(os.path.join(st.session_state.path_in, name)):
                    base, _ = os.path.splitext(name)
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

    st.markdown("---")
    st.markdown("**Advanced Processing (Optional)**")
    if st.button("Build Knowledge Graph (GraphRAG)"):
        if 'path_out' in st.session_state and os.path.exists(st.session_state.path_out):
            with st.spinner("Extracting entities and building Graph... This may take several minutes."):
                start_server_lmstudio('meta-llama-3-8b-instruct') # O el modelo que prefieras usar
                build_knowledge_graph(st.session_state.path_out, API_URL)
                close_server_lmstudio()
            st.success("✅ Knowledge Graph created successfully!")
        else:
            st.error("⚠️ Please define a valid output directory first.")

    st.divider()
    
    st.subheader("2. Chat Configuration")
    chat_model = st.selectbox(
        "Model for chatting (RAG):", 
        available_models
    )
    if st.button("Load Vector Store and Model for RAG"):
        with st.spinner("Starting server and loading models..."):
            start_server_lmstudio(chat_model)
            subprocess.run(['lms', 'load', 'text-embedding-bge-m3'], shell=True) 
            
            st.session_state.vector_store = load_base_vectorial(API_URL, st.session_state.path_out)
            flat_documents = read_chunks_for_embedding(st.session_state.path_out)
            graph_file = os.path.join(st.session_state.path_out, "knowledge_graph.graphml")
            
            if os.path.exists(graph_file):
                st.session_state.retriever = build_advanced_retriever(
                    st.session_state.vector_store, 
                    flat_documents, 
                    graph_path=graph_file
                )
                st.success("✅ System ready! (Vector + GraphRAG enabled)")
            else:
                st.session_state.retriever = build_advanced_retriever(
                    st.session_state.vector_store, 
                    flat_documents
                )
                st.success("✅ System ready! (Vector RAG only)")

    if st.button("Shut Down LM Studio Server"):
        close_server_lmstudio()
        st.info("Server shut down.")
    
    st.divider()

st.title("📚 RAG Assistant Local")

for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if prompt := st.chat_input("Type your question here..."):
    if not hasattr(st.session_state, 'retriever'):
        st.error("⚠️ Please load the vector store and model from the sidebar first.")
    else:
        with st.chat_message("user"):
            st.markdown(prompt)
    
        source_type = None
        local_results = []
        generated_code = ""
        execution_result = ""
        context_made = ""
        
        with st.spinner("Analyzing intent and searching context..."):
            standalone_prompt = get_standalone_question(st.session_state.messages, prompt, API_URL)
            route = route_query(standalone_prompt, API_URL)
            if route == 'math':
                source_type = "calculation"
                st.caption("Route: Math. Generating and executing local Python script...")
                
                code_prompt = f"""Write a standalone Python script to solve: {prompt}.
                
                RULES:
                1. Use EXACTLY one 'import' per line.
                2. For example start with:
                   import numpy as np
                   import matplotlib.pyplot as plt
                   from scipy.integrate import quad
                4. ALWAYS end with plt.savefig('temp_plot.png') and plt.close().
                5. Output ONLY the code inside a ```python block."""
                
                temp_code_history = [{"role": "user", "content": code_prompt}]
                
                stream_code = consult_llm_whith_memory(temp_code_history, API_URL)
                raw_response = "".join(list(stream_code))
                match = re.search(r'```(?:python)?\s*(.*?)\s*```', raw_response, re.DOTALL | re.IGNORECASE)
                
                if match:
                    generated_code = match.group(1).strip()
                else:
                    generated_code = raw_response.strip()

                generated_code = generated_code.replace("plt.show()", "plt.savefig('temp_plot.png')\nplt.close()")

                replacements = {
                    "import numpy as npimport": "import numpy as np\nimport",
                    "as pltfrom": "as plt\nfrom",
                    "plt.show()": "plt.savefig('temp_plot.png')\nplt.close()"
                }
                for old, new in replacements.items():
                    generated_code = generated_code.replace(old, new)

                execution_result = execute_python_code(generated_code)
                
                user_message = f"""
                [SYSTEM NOTE]: Calculation executed. Explain result briefly.
                [Execution Result]: {execution_result}
                [Actual question]: {standalone_prompt}"""
                
            elif route == 'web':
                source_type = "web"
                st.warning("Route: Web. Fetching real-time internet data...")
                web_context = web_search(standalone_prompt)
                
                user_message = f"""
                [SYSTEM NOTE]: Answer using the following real-time information extracted from the internet:
                [Web Context]:{web_context}
                [Actual question]:{standalone_prompt}"""
                
            else:
                source_type = "local"
                st.caption("Route: RAG. Querying internal database...")
                
                rewritten_prompt = rewrite_query(standalone_prompt, API_URL)
                english_query = translate_query_to_english(rewritten_prompt, API_URL)
                
                local_results = st.session_state.retriever.invoke(english_query)
                context_made = "\n\n".join([doc.page_content for doc in local_results])
                
                relevance_grade = grade_retrieved_context(english_query, context_made, API_URL)
                
                if relevance_grade == 'yes':
                    st.caption("Relevant local context found. Generating answer...")
                    user_message = f"""
                    [Context]:{context_made}
                    [Actual question]:{standalone_prompt}"""
                else:
                    source_type = "web"
                    st.warning("Local documents do not contain the answer. Invoking Web Search...")
                    web_context = web_search(standalone_prompt)
                    
                    user_message = f"""
                    [SYSTEM NOTE]: The context in the local documents was IRRELEVANT. 
                    Answer using the following real-time information extracted from the internet:
                    [Web Context]:{web_context}
                    [Actual question]:{standalone_prompt}"""


        temporal_history = st.session_state.messages.copy()
        temporal_history.append({"role": "user", "content": user_message})

        with st.chat_message("assistant"): 
            message_placeholder = st.empty()
            
            with message_placeholder.container():
                stream = consult_llm_whith_memory(temporal_history, API_URL)
                final_answer = st.write_stream(stream)
            
            if source_type == "local" and context_made:
                with st.spinner("Verifying answer accuracy..."):
                    is_grounded = grade_hallucination(context_made, final_answer, API_URL)
                    
                if is_grounded == 'no':
                    st.warning("⚠️ **Self-RAG:** Unverified information detected. Auto-correcting...")
                    
                    message_placeholder.empty() 
                    
                    correction_message = f"""
                    [SYSTEM NOTE]: Tu respuesta anterior contenía alucinaciones o información externa no solicitada.
                    
                    Reescribe tu respuesta basándote ESTRICTAMENTE en el siguiente contexto. 
                    Si el contexto NO contiene la respuesta exacta, simplemente di: "Los documentos locales no contienen información suficiente para responder a esto."
                    NO inventes datos.
                    
                    [Contexto]: {context_made}
                    [Pregunta]: {standalone_prompt}
                    """
                    
                    correction_history = st.session_state.messages.copy()
                    correction_history.append({"role": "user", "content": correction_message})
                    
                    with message_placeholder.container():
                        st.caption("🔄 *Rewriting response based strictly on local context...*")
                        stream_corregido = consult_llm_whith_memory(correction_history, API_URL)
                        final_answer = st.write_stream(stream_corregido)
            if source_type == "local" and local_results:
                with st.expander("Local sources consulted (Vector Store & Graph)"):
                    for doc in local_results:
                        st.markdown(f"**File:** {doc.metadata.get('fuente', 'Unknown')}")
                        st.caption(f"... {doc.page_content[:200]} ...")
            
            elif source_type == "web":
                with st.expander("Web sources consulted (Tavily)"):
                    st.info("This information was retrieved in real-time from the web.")
            
            elif source_type == "calculation":
                with st.expander("Python code generated for calculus"):
                    st.code(generated_code, language="python")
                    st.caption(f"**Terminal result:** {execution_result}")
            
            if os.path.exists("temp_plot.png"):
                st.image("temp_plot.png")
                os.remove("temp_plot.png")

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": final_answer})
        
        if len(st.session_state.messages) > 11: 
            st.session_state.messages.pop(1)  
            st.session_state.messages.pop(1)
                
with st.sidebar:
    st.subheader("3. Session Export")
    
    def generate_export_text(history):
        text = "# RAG Conversation Log\n\n"
        text += "> Automatically generated from the Local RAG Assistant.\n\n---\n\n"
        
        for msg in history:
            if msg["role"] == "system":
                continue
                
            if msg["role"] == "user":
                text += f"### User:\n{msg['content']}\n\n"
            elif msg["role"] == "assistant":
                clean_content = msg['content'].replace("[SYSTEM NOTE]:", "")
                text += f"### Assistant (RAG):\n{clean_content}\n\n---\n\n"
                
        return text

    markdown_text = generate_export_text(st.session_state.messages)
    
    st.download_button(
        label="Download History as Markdown (.md)",
        data=markdown_text,
        file_name="rag_research_history.md",
        mime="text/markdown",
        help="Download the entire conversation to easily import it into your word processor."
    )