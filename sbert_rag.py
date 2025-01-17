from sentence_transformers import SentenceTransformer, util
import torch
from typing import List, Tuple, Optional
import logging
from document import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SBertRAG:
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        """
        Initialize RAG system with SBERT.
        Default model is all-mpnet-base-v2 which provides better performance than MiniLM
        """
        logging.info(f"Initializing SBertRAG with model {model_name}")
        self.model = SentenceTransformer(model_name)
        self.documents: List[Document] = []
        self.embeddings: Optional[torch.Tensor] = None
        
    def add_documents(self, texts: List[str], metadata_list: Optional[List[dict]] = None) -> None:
        """
        Add documents to the RAG system and compute their embeddings.
        
        Args:
            texts: List of document texts
            metadata_list: Optional list of metadata dictionaries for each document
        """
        logging.info("Adding documents to the RAG system")
        if metadata_list is None:
            metadata_list = [None] * len(texts)
            
        self.documents = [Document(text=text, metadata=meta) 
                         for text, meta in zip(texts, metadata_list)]
        
        logging.info(f"Computing embeddings for {len(texts)} documents")
        self.embeddings = self.model.encode(texts, 
                                          convert_to_tensor=True,
                                          show_progress_bar=True,
                                          normalize_embeddings=True)
        logging.info("Embeddings computed successfully")

    def retrieve(self, 
                query: str, 
                k: int = 3, 
                threshold: float = 0.0) -> List[Tuple[Document, float]]:
        """
        Retrieve the k most relevant documents for a given query.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            threshold: Minimum similarity score threshold
            
        Returns:
            List of tuples containing (document, similarity_score)
        """
        logging.info(f"Encoding query: {query}")
        query_embedding = self.model.encode(query, 
                                          convert_to_tensor=True,
                                          normalize_embeddings=True)
        logging.info("Query encoded successfully")

        logging.info("Calculating similarity scores")
        cos_scores = util.pytorch_cos_sim(query_embedding, self.embeddings)[0]
        logging.info("Similarity scores calculated successfully")

        top_results = torch.topk(cos_scores, k=min(k, len(cos_scores)))
        
        results = []
        for score, idx in zip(top_results[0], top_results[1]):
            score = score.item()
            if score >= threshold:
                results.append((self.documents[idx], score))
        
        return results

    def batch_retrieve(self, 
                       queries: List[str], 
                       k: int = 3, 
                       threshold: float = 0.0) -> List[List[Tuple[Document, float]]]:
        """
        Retrieve the k most relevant documents for each query in a batch of queries.
        
        Args:
            queries: List of query strings
            k: Number of documents to retrieve per query
            threshold: Minimum similarity score threshold
            
        Returns:
            List of retrieval results for each query
        """
        logging.info(f"Encoding all queries: {queries}")
        query_embeddings = self.model.encode(queries, 
                                           convert_to_tensor=True,
                                           show_progress_bar=True,
                                           normalize_embeddings=True)
        logging.info("Queries encoded successfully")
        
        logging.info("Calculating similarity scores for all queries")
        cos_scores = util.pytorch_cos_sim(query_embeddings, self.embeddings)
        logging.info("Similarity scores for all queries calculated successfully")
        
        all_results = []
        for query_scores in cos_scores:
            top_results = torch.topk(query_scores, k=min(k, len(query_scores)))
            
            query_results = []
            for score, idx in zip(top_results[0], top_results[1]):
                score = score.item()
                if score >= threshold:
                    query_results.append((self.documents[idx], score))
                    
            all_results.append(query_results)
            
        return all_results

# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag = SBertRAG()
    
    # Sample documents with metadata
    documents = [
        "Document 1 text",
        "Document 2 text",
        "Document 3 text"
    ]
    metadata = [
        {"id": 1, "source": "source1"},
        {"id": 2, "source": "source2"},
        {"id": 3, "source": "source3"}
    ]
    
    # Add documents to the RAG system
    rag.add_documents(documents, metadata)
    
    # Retrieve documents for a sample query
    query = "Sample query text"
    results = rag.retrieve(query)
    for doc, score in results:
        print(f"Document: {doc.text}, Score: {score}")