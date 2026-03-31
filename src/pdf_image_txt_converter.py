import os
from src.features import process_full_dir
from src.chat_bot import start_server_lmstudio, close_server_lmstudio

def extract_text_from_pdf(dir_in, dir_out, API_URL):
    """
    This function takes a directory path as input, reads all PDF files in that directory, and extracts the text from each PDF file.
    
    Args:
        - dir_in: The directory path where the PDF files are located.
        - dir_out: The directory path where the extracted text will be saved.
        - API_URL: The URL of the LLM API endpoint.
    Returns:
        - A dictionary where the keys are the PDF file names and the values are the extracted text from each PDF file.
    """
    models=['meta-llama-3-8b-instruct','qwen2.5-coder-7b-instruct','minicpm-v-4_5', 'pixtral-12b']

    os.makedirs(dir_out, exist_ok=True)

    print('Choose the model you want to use to convert PDFs to text:')
    for i, model in enumerate(models):
        print(f"{i+1}. {model}")
    choice = int(input("Enter your choice: ")) - 1
    model_name = models[choice]

    start_server_lmstudio(model_name)
        
    if not os.path.exists(dir_in):
        print(f"Error: In directory '{dir_in}' not exist.")
    else:
        process_full_dir(dir_in, API_URL, dir_out)
        print("\n--- Transcribe done ---")

    close_server_lmstudio()