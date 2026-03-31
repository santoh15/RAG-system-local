from matplotlib.pylab import choice

from src.chunking import directory_chunk_to_json
from src.pdf_image_txt_converter import extract_text_from_pdf
from src.embedding import read_chunks_for_embedding, create_chroma_vector_store
from src.chat_whit_RAG import start_chat, decide_add_files_db

'''
This is the main file that will execute the entire process of extracting text from PDFs, chunking the text, creating a vector store for embeddings,
and starting a chat interface with Retrieval-Augmented Generation (RAG) capabilities.
Is necessary write the correct paths for the input directory (where the PDFs are located) and the output directory (where the transcriptions and chunks will be stored).
Make sure to have the API running at the specified URL before executing this script, as it will be used for both embedding creation and chat interactions.

'''

dir_in = r"your\input\directory\path"
dir_out = r"your\output\directory\path"
API_URL = "http://localhost:1234/v1/chat/completions"

choice = decide_add_files_db(dir_in, dir_out)

if choice == 'yes':
    extract_text_from_pdf(dir_in, dir_out, API_URL)
    directory_chunk_to_json(dir_out)
    documents = read_chunks_for_embedding(dir_out)
    create_chroma_vector_store(documents, dir_out, API_URL)
    print(f"New files added to the database.")

else:
    print(f"Skipping the process of adding new files to the database.")
    documents = read_chunks_for_embedding(dir_out)
    create_chroma_vector_store(documents, dir_out, API_URL)



start_chat(API_URL, dir_out)