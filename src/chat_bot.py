''''    

The functions in this file are a simple chatbot application that interacts with a large language model (LLM) using the OpenAI API.
The chatbot allows users to input questions and receive responses from the LLM, maintaining a conversation history.
The user can exit the chat by typing 'exit' or 'quit'.

Names of models available in LM Studio:
- meta-llama-3-8b-instruct
- qwen2.5-coder-7b-instruct
- minicpm-v-4_5


'''
import time
import subprocess
from openai import OpenAI

def start_chatbot_lmstudio(model_name):
    '''
    Starts a chatbot that interacts with a local LLM using the OpenAI API. The chatbot maintains a conversation history and allows the user to exit by typing 'exit' or 'quit'.
    Args:
        model_name (str): The name of the LLM model to use for the chatbot.

    '''
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    print("--- Starting chat with local LLM, {}.  (write 'exit' or 'quit' to exit)---".format(model_name))
    historial = []

    while True:
        question = input("\nYou: ")
        
        if question.lower() in ['salir', 'exit', 'quit']:
            print("Exiting chat.")
            break
            
        historial.append({'role': 'user', 'content': question})
        
        try:
            answer = client.chat.completions.create(
                model=model_name, 
                messages=historial,
                temperature=0.7,
                stream=True
            )
            print("\nLLM: ", end="", flush=True)
            
            message_complete = ""
            for part in answer:
                content = part.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
                    message_complete += content
            
            print()
            
            historial.append({'role': 'assistant', 'content': message_complete})
            
        except Exception as e:
            print(f"\nError: {e}")


def start_server_lmstudio(model_name):
    '''
    Starts the LM Studio server and loads the specified model. It first unloads any previously loaded models to free up VRAM, then starts the server and loads the new model.
    Args:
        model_name (str): The name of the LLM model to load into VRAM.
    Returns:
        process: The subprocess running the LM Studio server, which can be used to manage the server lifecycle.
    
    '''
    subprocess.run(['lms', 'unload', '--all'], shell=True, stdout=subprocess.DEVNULL)
    print("Starting LM Studio server...")
    process = subprocess.Popen(['lms', 'server', 'start'], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    time.sleep(3)
    
    print("Loading {} into VRAM...".format(model_name))
    subprocess.run(['lms', 'load', model_name], shell=True)
    
    return process

def close_server_lmstudio():
    '''
    Closes the LM Studio server and unloads all loaded models. This function is useful for freeing up system resources after the chatbot session is complete.
    Args:
        None
    Returns:
        None
    
    '''
    subprocess.run(['lms', 'unload', '--all'], shell=True, stdout=subprocess.DEVNULL)
    subprocess.run(['lms', 'server', 'stop'], shell=True)
    print("Server turned off.")

import subprocess
import re

import subprocess
import re

import subprocess
import re

def get_local_models():
    """
    Obtiene los nombres de los modelos (primera columna) de la salida de 'lms ls'.
    """
    try:
        # Ejecutamos el comando lms ls
        result = subprocess.run(['lms', 'ls'], capture_output=True, text=True, shell=True)
        if result.returncode != 0:
            return []
        
        lines = result.stdout.strip().split('\n')
        models = []
        
        # Palabras que queremos ignorar porque son encabezados o texto informativo
        ignore_list = ["LLM", "EMBEDDING", "PARAMS", "ARCH", "SIZE", "DEVICE", "You have", "---"]

        for line in lines:
            line = line.strip()
            # Dividimos la línea por 2 o más espacios
            parts = re.split(r'\s{2,}', line)
            
            if len(parts) > 0:
                # El nombre del modelo es SIEMPRE la primera parte (índice 0)
                model_name = parts[0].strip()
                
                # Validamos que no sea un encabezado ni esté vacío
                if model_name and not any(word in model_name for word in ignore_list):
                    models.append(model_name)
        
        return sorted(list(set(models)))
    except Exception as e:
        print(f"Error detectando modelos: {e}")
        return []