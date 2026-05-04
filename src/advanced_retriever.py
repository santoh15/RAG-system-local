from langchain_community.retrievers.bm25 import BM25Retriever
from sentence_transformers import CrossEncoder

class CustomAdvancedRetriever:
    def __init__(self, vector_store, documents):
        self.vector_store = vector_store
        
        self.bm25 = BM25Retriever.from_documents(documents)
        self.bm25.k = 10

        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
    def invoke(self, query):
        """
        Executes Hybrid Search (Semantic + Lexical), removes duplicates, 
        and re-ranks the results using the Cross-Encoder model.
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
        
        return [doc for doc, score in scored_docs[:3]]

def build_advanced_retriever(vector_store, documents):
    """Factory function compatible with app.py"""
    return CustomAdvancedRetriever(vector_store, documents)