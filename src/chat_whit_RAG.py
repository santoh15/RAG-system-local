import requests
import os
import subprocess
from src.embedding import load_base_vectorial
from src.chat_bot import start_server_lmstudio, close_server_lmstudio


def consult_llm_whith_memory(historial_mensajes, API_URL):
    '''
    Consult to the LLM with the historial of messages, including the context and the actual question.
    The LLM will use the context to answer the question, and it can also use the previous messages to keep the conversation coherent.
    Args:
        historial_mensajes (list): List of messages in the conversation, including the context and the actual question.
        API_URL (str): URL of the LLM API.
    Returns:
        str: The answer from the LLM.

    '''
    payload = {
        "model": "local-model",
        "messages": historial_mensajes,
        "temperature": 0.3,
        "max_tokens": 4096,
        "stream": False
    }
    try:
        answer = requests.post(API_URL, json=payload, timeout=600)
        answer.raise_for_status()
        return answer.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error to consult the LLM: {e}"



def start_chat(API_URL, dir_out):
    '''
    Starts the chat session with the RAG-enabled LLM.
    Args:
        API_URL (str): URL of the LLM API.
        dir_out (str): Directory where the translation of PDF files will be saved.
    Returns:
        None
    '''
    start_server_lmstudio("text-embedding-bge-m3")
    models=['meta-llama-3-8b-instruct','minicpm-v-4_5','qwen2.5-coder-7b-instruct']
    
    print('Choose model to chat using RAG:')
    for i, model in enumerate(models):
        print(f"{i+1}. {model}")
    choice = int(input("Enter your choice: ")) - 1
    
    model_name = models[choice]
    subprocess.run(['lms', 'load', model_name], shell=True)
    
    vector_store = load_base_vectorial(API_URL, dir_out)
    print("\n LLM ready. Write 'exit' to terminate.")
    print("-" * 60)
    
    prompt_sistema = """Sos un asistente. 
    Tu tarea es responder a la pregunta del usuario. 
    Primero, utilizá la información provista en el 'Contexto' de los apuntes como base principal. 
    Luego, podés expandir, explicar o dar ejemplos utilizando tu propio conocimiento general. 
    Si usás fórmulas o código, formatealos correctamente."""

    
    historial_chat = [{"role": "system", "content": prompt_sistema}]
    
    while True:
        question = input("\n I: ")
        
        if question.lower() in ['salir', 'exit', 'quit']:
            break
            
        if not question.strip():
            continue
            
        print("Finding relevant information in the vector store...")
        results = vector_store.similarity_search(question, k=3)
        context_made = "\n\n".join([doc.page_content for doc in results])
        
        user_message = f"""
    [Context]:{context_made}
    [Actual question]:{question}"""

        historial_chat.append({"role": "user", "content": user_message})
        
        final_answer = consult_llm_whith_memory(historial_chat, API_URL)
        
        print("\n--- IA ---")
        print(final_answer)
        print("----------")
        historial_chat.append({"role": "assistant", "content": final_answer})
        
        if len(historial_chat) > 11:
            historial_chat.pop(1)
            historial_chat.pop(1)
    close_server_lmstudio()
    



def decide_add_files_db(dir_in, dir_out):
    '''
    Checks if exists PDF files but not the corresponding .txt files in the output directory, which means that there are new files to add to the vector store database.
    If there are new files, it asks the user if they want to add them to the database. If the user wants to add them, convert the new PDF files to .txt files and add them to the vector store database.
    If the user doesn't want to add them, skip the process and continue with the chat session. If there are no new files, skip the process and continue with the chat session.
    
    Args:
        dir_in (str): Directory where the PDF files are located.
        dir_out (str): Directory where the translation of PDF files will be saved.
    Returns:
        str: 'yes' if the user wants to add the new files to the database, 'no' if the user doesn't want to add the new files to the database or no new files are detected.
    '''

    archives = os.listdir(dir_in)
    
    for idx, name in enumerate(archives, 1):
        full_path = os.path.join(dir_in, name)
        
        if os.path.isdir(full_path):
            continue

        base_name, ext = os.path.splitext(name)
        ext = ext.lower()
        path_txt_out = os.path.join(dir_out, f"{base_name}.txt")
        
        if not os.path.exists(path_txt_out):
            print(f"New files detected ( .txt does not exists)")
            print(f"Do you want to add the new files to the database? (yes/no)")
            choice = input().lower()
            if choice == 'yes' or choice == 'y':
                return 'yes'
            else:
                return 'no'
        else:
            return 'no'
