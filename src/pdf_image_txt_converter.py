import os
from src.features import process_full_dir
from src.chat_bot import start_server_lmstudio, close_server_lmstudio

def extract_text_from_pdf(dir_in, dir_out, API_URL, model_name):
    """
    This function takes a directory path as input, reads all PDF files in that directory, and extracts the text from each PDF file.
    
    Args:
        - dir_in: The directory path where the PDF files are located.
        - dir_out: The directory path where the extracted text will be saved.
        - API_URL: The URL of the LLM API endpoint.
    Returns:
        - A dictionary where the keys are the PDF file names and the values are the extracted text from each PDF file.
    """
    
    os.makedirs(dir_out, exist_ok=True)
    start_server_lmstudio(model_name)
    if not os.path.exists(dir_in):
        print(f"Error: In directory '{dir_in}' not exist.")
    else:
        from src.features import process_full_dir # Asegúrate de que la importación funcione
        process_full_dir(dir_in, API_URL, dir_out)
    close_server_lmstudio()
