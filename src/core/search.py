import os
from core.embedder import Embedder
from core.vector_store import VectorStore
from common.config import TOP_K_RESULTS, FAISS_INDEX_PATH
from utils.logging import log_info, log_error
from common.config import DEVICE

class SearchEngine:
    """
    The SearchEngine class coordinates the retrieval process.
    It turns natural language queries into vectors and finds the most
    semantically similar chunks in the FAISS database.
    """
    def __init__(self, vector_store=None, embedder=None):
        """
        Initializes the search engine by loading the embedding model 
        and connecting to the vector database.
        """
        log_info("Initializing Search Engine...")
        self.embedder = embedder if embedder else Embedder()
        self.vector_store = vector_store if vector_store else VectorStore()
        
        # Ensure we have data to search through
        if os.path.exists(FAISS_INDEX_PATH):
            self.vector_store.load(FAISS_INDEX_PATH)
            log_info("Existing index loaded.")
        else:
            log_info("No existing index found. Search will be unavailable until files are uploaded.")

    def query(self, user_query, top_k=TOP_K_RESULTS):
        """
        Executes a semantic search.
        1. Encodes the query using the same E5 model used for ingestion.
        2. Queries the FAISS index for the 'top_k' closest vectors.
        3. Maps vector IDs back to their text and source metadata.
        """
        # Multilingual-E5 models require a "query: " prefix to distinguish 
        # the user's question from the "passage: " chunks being searched.
        query_text = f"query: {user_query}"
        
        # Generate the embedding vector for the query
        # device=DEVICE ensures it runs on CPU or GPU based on your config
        query_vector = self.embedder.model.encode([query_text], convert_to_numpy=True, device=DEVICE)
        
        # Perform the L2 distance search in FAISS
        # Distances: How similar they are (smaller = better for L2)
        # Indices: The position of the result in the metadata list
        distances, indices = self.vector_store.index.search(
            query_vector.astype('float32'), 
            top_k
        )
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            # FAISS returns -1 if it can't find enough matches
            if idx != -1:
                # Retrieve the actual text and source info from metadata
                result_item = self.vector_store.metadata[idx].copy()
                # Store the distance score for potential filtering/ranking
                result_item["score"] = float(distances[0][i])
                results.append(result_item)
        
        log_info(f"Search complete. Found {len(results)} relevant chunks.")
        return results