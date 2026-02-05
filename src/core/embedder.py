from sentence_transformers import SentenceTransformer
from utils.logging import log_info
from common.config import EMBEDDING_MODEL_NAME, DEVICE

class Embedder:
    """
    The Embedder class transforms text strings into high-dimensional vectors.
    These vectors capture the semantic meaning of the text.
    """
    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        """
        Initializes the SentenceTransformer model.
        A multilingual model (like E5 or LaBSE) is used to support Croatian.
        'device=cuda' pushes the model to your GPU for faster processing.
        """
        log_info(f"Loading Embedding Model: {model_name}")
        # The model is loaded onto the GPU to leverage parallel processing
        self.model = SentenceTransformer(model_name, device=DEVICE)

    def generate_embeddings(self, chunks):
        """
        Converts a list of text chunks (dictionaries/strings) into numerical vectors.
        
        Args:
            chunks: A list of text strings extracted from your documents.
            
        Returns:
            A numpy array where each row is a vector representing a chunk.
        """
        if not chunks:
            return []
        
        log_info(f"Generating embeddings for {len(chunks)} text chunks...")
        
        # It processes 32 sentences at a time to maximize GPU throughput.
        embeddings = self.model.encode(
            chunks, 
            batch_size=32, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        log_info("Embedding generation complete.")
        return embeddings