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
        model_kwargs={},
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





def directory_chunk_to_json(dir_out, choice):
    '''
    This function processes all new .txt files in dir_out, chunks the text,
    and appends them to the JSON file without repeating old ones.
    Args:
        - dir_out: The directory where the .txt files are located and where the JSON will be saved.
        - choice: The chunking method to use ('1' for recursive, '2' for semantic).
    '''
    try:
        print(f"The directory to load .txt files for chunking is: {dir_out}")
        
        archives = os.listdir(dir_out)
        json_archive = os.path.join(dir_out, "chunks_for_embedding", "prepared_chunks.json")
        
        chunks_for_save = []
        processed_files = set()
        
        if os.path.exists(json_archive):
            with open(json_archive, "r", encoding="utf-8") as f:
                chunks_for_save = json.load(f)
                for chunk in chunks_for_save:
                    processed_files.add(chunk["metadatos"]["fuente"])
            print(f"[*] Found {len(processed_files)} files already processed in the JSON.")

        new_chunks_count = 0
        
        for idx, name in enumerate(archives, 1):
            full_path = os.path.join(dir_out, name)
            
            if os.path.isdir(full_path) or not name.endswith('.txt'):
                continue
                
            if name in processed_files:
                continue

            print(f"\n[+] Processing NEW file: {name}...")
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
                    new_chunks_count += 1
        
        if new_chunks_count > 0:
            os.makedirs(os.path.dirname(json_archive), exist_ok=True)
            with open(json_archive, "w", encoding="utf-8") as archivo_json:
                json.dump(chunks_for_save, archivo_json, ensure_ascii=False, indent=4)
            print(f"[✓] Added {new_chunks_count} new chunks. JSON updated in '{json_archive}'.")
        else:
            print("[✓] No new .txt files detected for processing.")
            
    except Exception as e:
        print(f"  Error: {e}")