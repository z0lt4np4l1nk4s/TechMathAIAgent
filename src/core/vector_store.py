import faiss
import numpy as np
import os
import pickle
from common.config import EMBEDDING_DIMENSION, FAISS_INDEX_PATH, METADATA_PATH, VECTOR_STORE_FOLDER
from utils.logging import log_success, log_info, log_error, log_warning

class VectorStore:
    """
    Manages the FAISS index and associated metadata. 
    Handles storage, retrieval, and persistence of vector embeddings.
    """
    def __init__(self, embedding_dim=EMBEDDING_DIMENSION):
        """
        Initializes the VectorStore on CPU (Optimized for Windows stability).
        Automatically attempts to load existing data from disk to maintain continuity.
        """
        self.embedding_dim = embedding_dim
        self.metadata = []
        os.makedirs(VECTOR_STORE_FOLDER, exist_ok=True)

        # Automatic check for existing data to prevent loss on application restart
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
            self.load()
        else:
            # Initialize a new empty L2 index on CPU. 
            # IndexFlatL2 calculates exact Euclidean distance (best for smaller datasets).
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            log_info("Initialized new empty FAISS index on CPU.")

    def add_to_index(self, embeddings, chunks):
        """
        Appends new embeddings and their corresponding metadata (text + source) to the store.
        """
        if embeddings is None or len(embeddings) == 0:
            log_warning("No embeddings provided to add_to_index.")
            return

        # Ensure data is in float32 format, as required by the FAISS C++ backend
        data = np.array(embeddings).astype('float32')
        
        # Add vectors to the CPU index
        self.index.add(data)
        
        # Extend the metadata list to keep it synchronized with the vector IDs (indices)
        self.metadata.extend(chunks)
        log_info(f"Successfully added {len(embeddings)} vectors to index.")

    def save(self, index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH):
        """
        Persists the current FAISS index and metadata list to disk using binary formats.
        """
        try:
            # Save the structural FAISS index
            faiss.write_index(self.index, index_path)
            
            # Save the Python metadata list using Pickle
            with open(metadata_path, "wb") as f:
                pickle.dump(self.metadata, f)

            log_success(f"Vector store saved. Total vectors in index: {self.index.ntotal}")
        except Exception as e:
            log_error(f"Failed to save vector store: {e}")

    def load(self, index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH):
        """
        Loads the FAISS index and the associated metadata from disk back into RAM.
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found at: {index_path}")
            
        # Read the FAISS index directly into CPU memory
        self.index = faiss.read_index(index_path)
        
        # Unpickle the metadata list
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        
        log_info(f"Loaded existing database: {self.index.ntotal} vectors found.")