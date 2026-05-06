from langchain_community.retrievers.bm25 import BM25Retriever
from sentence_transformers import CrossEncoder
import networkx as nx
import os

class CustomAdvancedRetriever:
    def __init__(self, vector_store, documents, graph_path=None):
        self.vector_store = vector_store
        
        self.bm25 = BM25Retriever.from_documents(documents)
        self.bm25.k = 10

        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        self.graph = None
        if graph_path and os.path.exists(graph_path):
            self.graph = nx.read_graphml(graph_path)

    def extract_entity_from_query(self, query):
        ''' Extracts potential entities from the query by matching query words with graph nodes. 
            Args: 
                -query (str): The user's query.
            Returns: 
                -List of graph nodes that are relevant to the query.
            '''
        
        if not self.graph: return []
        
        query_words = query.lower().split()
        relevant_nodes = []
        for node in self.graph.nodes():
            if any(word in str(node).lower() for word in query_words):
                relevant_nodes.append(node)
        return relevant_nodes

    def query_graph(self, query):
        """Search the graph for relevant nodes and their relationships based on the query.
        Args:
            - query (str): The user's query.
        Returns:
            - str: A string containing the relevant graph facts.
        """
        if not self.graph: return ""
        
        nodes = self.extract_entity_from_query(query)
        graph_context = []
        
        for node in nodes:
            for neighbor in self.graph.neighbors(node):
                edge_data = self.graph.get_edge_data(node, neighbor)
                relation = edge_data.get('relation', 'is related to')
                source = edge_data.get('source', 'Unknown')
                graph_context.append(f"Graph Fact (Source: {source}): {node} [{relation}] {neighbor}")
                
        return "\n".join(graph_context)
        
    def invoke(self, query):
        """
        Execute hibrid search (Vector + BM25) + GraphRAG
        Args:
            - query (str): The user's query.
        Returns:
            - List[Document]: A list of relevant documents, potentially enriched with graph context.
        """
        chroma_results = self.vector_store.similarity_search(query, k=10)
        bm25_results = self.bm25.invoke(query)

        unique_docs = {}
        for doc in chroma_results + bm25_results:
            if doc.page_content not in unique_docs:
                unique_docs[doc.page_content] = doc
                
        candidate_docs = list(unique_docs.values())

        pairs = [[query, doc.page_content] for doc in candidate_docs]
        scores = self.cross_encoder.predict(pairs)
        
        scored_docs = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, score in scored_docs[:3]]
        
        graph_context_str = self.query_graph(query)
        if graph_context_str:
            from langchain_core.documents import Document
            graph_doc = Document(page_content=f"--- KNOWLEDGE GRAPH EXTRACT ---\n{graph_context_str}", metadata={"fuente": "Knowledge Graph"})
            top_docs.insert(0, graph_doc)

        return top_docs

def build_advanced_retriever(vector_store, documents, graph_path=None):
    return CustomAdvancedRetriever(vector_store, documents, graph_path)