import json
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings



def chunking_text_recursive(text, chunk_size=800, chunk_overlap=350):
    """
    This function takes a long text and splits it into smaller chunks using the RecursiveCharacterTextSplitter from Langchain.
    
    Args:
        - text: The long text to be split.
        - chunk_size: The maximum size of each chunk in characters (default is 800).
        - chunk_overlap: The number of characters that overlap between chunks to maintain context (default is 200).
    
    Returns:
        - A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    return text_splitter.split_text(text)


def chunking_text_semantic(text):
    """
    This function takes a long text and splits it into smaller chunks using the SemanticChunker from Langchain.
    
    Args:
        - text: The long text to be split.
    
    Returns:
        - A list of text chunks.
    
    """
    embeddings_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    chunker_semantico = SemanticChunker(
        embeddings_model, 
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90 
    )
    documentos = chunker_semantico.create_documents([text])
    chunks_text = [doc.page_content for doc in documentos]
    return chunks_text





def directory_chunk_to_json(dir_in):
    '''
    This function processes all the .txt files in a given directory, chunks the text using the chunking_text function,
    and saves the chunks in a JSON file whit the metadata of the original file.
    
    Args:
        - dir_in: The directory containing the .txt files to be processed.
    Returns:
        - None (the function saves the output as JSON files in the same directory).

    '''
    
    try:
        print(f"The directory to load .txt files for chunking is: {dir_in}")
        
        archives = os.listdir(dir_in)
        total_archives = len(archives)
        chunks_for_save = []
        print('Choose the chunking method:')
        print('1. Recursive')
        print('2. Semantic')
        choice = input('Enter your choice (1 or 2): ')

        for idx, name in enumerate(archives, 1):
            full_path = os.path.join(dir_in, name)
            
            if os.path.isdir(full_path):
                continue
            print(f"\n[{idx}/{total_archives}] Processing: {name}...")
            with open(full_path, 'r', encoding='utf-8') as f:
                
                text = f.read()
                if choice == '1':
                    chunks = chunking_text_recursive(text)
                elif choice == '2':
                    chunks = chunking_text_semantic(text)
    
                for chunk in chunks:
                    doc = Document(
                        page_content=chunk,
                        metadata={"fuente": name}
                    )
                    chunks_for_save.append({
                        "content": doc.page_content,
                        "metadatos": doc.metadata
                        })
        json_archive_old = os.path.join(dir_in, "chunks_for_embedding\\prepared_chunks.json")
        with open(json_archive_old, "w", encoding="utf-8") as archivo_json:
            json.dump(chunks_for_save, archivo_json, ensure_ascii=False, indent=4)
        

        print(f"[✓] {len(chunks_for_save)} chunks saved in '{json_archive_old}'.")
    except Exception as e:
        print(f"  Error: {e}")