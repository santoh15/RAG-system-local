import time
from src.chat_bot import start_server_lmstudio, close_server_lmstudio
import os
import json
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def read_chunks_for_embedding(dir_out):
    '''
    Reads the chunks of text from a JSON file for embedding. Each chunk is expected to have a "content" field for the text and a "metadatos" field for metadata.
    Returns a list of Document objects, each containing the content and metadata for a chunk.
    The function handles exceptions that may occur during file reading and JSON parsing, printing an error message if any issues arise.
    Args:
            dir_out (str): The directory path where the JSON file containing the chunks is located.
    
    Returns:
            List[Document]: A list of Document objects created from the chunks in the JSON file. Each Document contains the page content and associated metadata.
                            If an error occurs, an empty list is returned.
    
    '''
    documents = []
    try:
        json_path = os.path.join(dir_out, "chunks_for_embedding\\prepared_chunks.json")
        with open(json_path, "r", encoding="utf-8") as json_file:
            chunks_data = json.load(json_file)
            documents = [Document(page_content=chunk["content"], metadata=chunk["metadatos"]) for chunk in chunks_data]
        
    except Exception as e:
        print(f"Error reading chunks for embedding: {e}")
    return documents


def create_chroma_vector_store(documents, dir_out, API_URL):
    '''
    Creates a Chroma vector store from a list of Document objects, using OpenAI embeddings.
    The function starts a local LM Studio server to access the embedding model, creates the vector store, and then closes the server.
    It handles exceptions that may occur during the process, printing an error message if any issues arise.
    Args:
        documents (List[Document]): A list of Document objects to be embedded and stored in the Chroma vector store.
        dir_out (str): The directory path where the transcription filesa are located, saving the embeddings in a subdirectory called "embeddings".
        API_URL (str): The base URL for the LM Studio API, used to access the embedding model.
    Returns:
        None: The function does not return a value, but it creates and persists a Chroma vector store in the specified directory, creating a directory if it doesn't exist to save the embeddings
              If an error occurs, it prints an error message and ensures that the LM Studio server is closed properly.
    '''
    try:
        start_server_lmstudio('text-embedding-bge-m3')
        correct_url = API_URL.split("/chat")[0]
        time.sleep(5)

        embeddings = OpenAIEmbeddings(
        base_url=correct_url,
        api_key="lm-studio",
        model="text-embedding-bge-m3",
        chunk_size=8,       
        check_embedding_ctx_length=False)
        path_db = os.path.join(dir_out, "embeddings")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=path_db)
        print(f"Chroma vector store created and persisted in '{path_db}'.")
    except Exception as e:
        print(f"Error creating Chroma vector store: {e}")
    finally:
        close_server_lmstudio()


def load_base_vectorial(API_URL, dir_out):
    '''
    Loads the base vector store from the Chroma database.
    Args:
        API_URL (str): The base URL for the LM Studio API, used to access the embedding model.
        dir_out (str): The directory path where the transcription files are located, finding the embeddings in a subdirectory called "embeddings".
    Returns:
        Chroma: The loaded Chroma vector store.
    
    '''
    print("Conecting with the Chroma database...")
    correct_url = API_URL.split("/chat")[0]
    embeddings = OpenAIEmbeddings(
        base_url=correct_url,
        api_key="lm-studio",
        model="text-embedding-bge-m3",       
        check_embedding_ctx_length=False)
    vector_store = Chroma(
        persist_directory=os.path.join(dir_out, "embeddings"),
        embedding_function=embeddings
    )
    return vector_store
