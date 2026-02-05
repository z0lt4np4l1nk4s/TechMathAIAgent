import re
from common.config import CHUNK_DEFAULT_SIZE, CHUNK_DEFAULT_OVERLAP
from utils.logging import log_info, log_warning

class TextChunker:
    """
    Handles the fragmentation of long documents into smaller, manageable pieces (chunks).
    This is essential for RAG systems to fit context into the LLM's limited window.
    """
    def __init__(self, chunk_size=CHUNK_DEFAULT_SIZE, chunk_overlap=CHUNK_DEFAULT_OVERLAP):
        self.chunk_size = chunk_size        # Maximum characters per chunk
        self.chunk_overlap = chunk_overlap  # Overlap between chunks to preserve context

    def chunk_text(self, text, source_name="Unknown"):
        """
        Splits text into meaningful chunks and attaches source metadata.
        Prioritizes splitting at newlines to avoid cutting sentences or formulas in half.
        """
        if not text:
            log_warning(f"Received empty text for source: {source_name}")
            return []

        # Clean up OCR noise (multiple spaces) to normalize text for embedding
        text = re.sub(r' +', ' ', text)
        
        chunks = []
        start = 0
        text_len = len(text)

        # Logging the start of the process
        log_info(f"Starting chunking for: {source_name} (Total length: {text_len} chars)")

        while start < text_len:
            # Determine the theoretical end of the chunk
            end = min(start + self.chunk_size, text_len)
            
            # Smart Splitting: If we are not at the end of the text, look for a newline 
            # to split the chunk naturally instead of cutting a word/formula.
            if end < text_len:
                last_newline = text.rfind('\n', start, end)
                # Only split at newline if it's reasonably close to the chunk size boundary
                if last_newline > start + (self.chunk_size // 2):
                    end = last_newline

            chunk_content = text[start:end].strip()
            if chunk_content:
                # Attach source metadata so we can display citations in the UI
                chunks.append({
                    "text": chunk_content,
                    "source": source_name
                })
            
            # Slide the window forward, keeping the overlap for continuity
            start = end - self.chunk_overlap
            
            # Break if we've reached the end to avoid infinite loops
            if end >= text_len:
                break

        # Log the final result for this document
        log_info(f"Finished: {source_name} split into {len(chunks)} chunks.")
        return chunks