import os
import json
import networkx as nx
from src.chat_bot import start_server_lmstudio, close_server_lmstudio
import requests

def extract_triplets_from_text(text, API_URL):
    """
    Extract triplets from the given text.
    Args:        
        text (str): The input text to extract triplets from.
        API_URL (str): The URL of the local LLM API.
    """
    system_prompt = """You are a precise data extractor for chemistry and physics texts.
Extract entities and relationships from the text in the format: Subject | Relationship | Object.
Do not output anything else. If no clear relationships exist, output 'None'.
Example: Phenylacetylene | is a | dienophile"""

    payload = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract triplets from:\n{text}"}
        ],
        "temperature": 0.1,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=180)
        content = response.json()['choices'][0]['message']['content'].strip()
        
        triplets = []
        for line in content.split('\n'):
            parts = [p.strip() for p in line.split('|')]
            if len(parts) == 3:
                triplets.append(tuple(parts))
        return triplets
    except Exception as e:
        print(f"Error extracting triplets: {e}")
        return []

def build_knowledge_graph(dir_out, API_URL):
    """
    Read the prepared chunks and build the knowledge graph.
    Args:
        dir_out (str): The output directory where the prepared chunks are stored.
        API_URL (str): The URL of the local LLM API.
    """
    json_path = os.path.join(dir_out, "chunks_for_embedding", "prepared_chunks.json")
    if not os.path.exists(json_path):
        print("No chunks found to build the graph.")
        return
        
    with open(json_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    G = nx.DiGraph()
    
    print(f"Building Knowledge Graph from {len(chunks_data)} chunks...")
    for idx, chunk in enumerate(chunks_data):
        print(f"Processing chunk {idx+1}/{len(chunks_data)}...")
        triplets = extract_triplets_from_text(chunk['content'], API_URL)
        
        for subject, relation, obj in triplets:
            G.add_edge(subject, obj, relation=relation, source=chunk['metadatos']['fuente'])
            

    graph_path = os.path.join(dir_out, "knowledge_graph.graphml")
    nx.write_graphml(G, graph_path)
    print(f"Graph saved to {graph_path} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")