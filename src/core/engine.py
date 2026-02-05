import os
import shutil
import streamlit as st
from core.generator import LocalGenerator
from preprocessing.document_parser import DocumentParser
from preprocessing.text_chunker import TextChunker
from core.embedder import Embedder
from core.vector_store import VectorStore
from utils.logging import log_info, log_success, log_warning, log_error
from utils.code_executor import execute_python_code
from common.config import (
    INPUT_DATA_FOLDER, 
    OUTPUT_DATA_FOLDER, 
    PROCESSED_DATA_FOLDER,
    VECTOR_STORE_FOLDER
)

class Engine:
    """
    The Orchestrator class that manages the lifecycle of the RAG application.
    It handles initialization, file ingestion, and query routing.
    """
    def __init__(self, clear_on_start=False):
        self.generator = None
        self.is_indexing = False
        self.processed_files = set()

        # Placeholders for sub-modules
        self.parser = None
        self.chunker = None
        self.embedder = None
        self.vector_store = None

        if clear_on_start:
            self._wipe_database()

        self._ensure_folders()

    def boot(self):
        """
        Loads the heavy AI models and existing database into memory.
        This is separated from __init__ to allow for faster app startup 
        and selective model loading.
        """   
        log_info("Booting Engine sub-modules...")
        self.parser = DocumentParser()
        self.chunker = TextChunker()
        self.embedder = Embedder()
        self.vector_store = VectorStore()

        if not self.generator:
            self.generator = LocalGenerator(vector_store=self.vector_store, embedder=self.embedder)
        
        # Load the set of already processed files from the vector store metadata
        if self.vector_store.metadata:
            indexed_files = {m.get('source') for m in self.vector_store.metadata if m.get('source')}
            self.processed_files.update(indexed_files)
            log_info(f"Engine booted. {len(self.processed_files)} files already in memory.")
    
        return True

    def process_uploads(self, uploaded_files):
        """
        Handles new file uploads. It saves them to disk and triggers 
        the incremental indexing pipeline if the files are new.
        """
        if not uploaded_files:
            return False

        current_names = {f.name for f in uploaded_files}
        new_files = current_names - self.processed_files

        if new_files:
            self.is_indexing = True
            log_info(f"Found {len(new_files)} new files to index.")
            
            # Physical save of the uploaded buffers to the local input folder
            for f in uploaded_files:
                if f.name in new_files:
                    save_path = os.path.join(INPUT_DATA_FOLDER, f.name)
                    with open(save_path, "wb") as out:
                        out.write(f.getbuffer())
            
            # Start the conversion pipeline (Parsing -> Chunking -> Embedding -> FAISS)
            self._process_files()
            
            self.processed_files.update(new_files)
            self.is_indexing = False
            return True
        return False

    def generate_response(self, query):
        """
        Main RAG entry point: 
        1. Search the index. 
        2. Format prompt. 
        3. Generate LLM answer.
        """
        log_info(f"Processing query: {query}")
        return self.generator.generate_answer(query)
    
    def _process_files(self):
        """
        Internal incremental pipeline. Only processes files that 
        do not yet exist in the FAISS index.
        """
        existing_sources = {m.get('source') for m in self.vector_store.metadata if m.get('source')}
        all_new_chunks = []

        for filename in os.listdir(INPUT_DATA_FOLDER):
            if filename in existing_sources:
                continue

            input_path = os.path.join(INPUT_DATA_FOLDER, filename)
            # Step 1: Parse (PDF/Image to Text)
            text_content = self.parser.parse(input_path)

            if text_content:
                # Save a .txt copy of the OCR for debugging
                name = os.path.splitext(filename)[0]
                output_path = os.path.join(OUTPUT_DATA_FOLDER, f"{name}.txt")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(text_content)

                # Step 2: Chunking
                chunks = self.chunker.chunk_text(text_content, source_name=filename)
                all_new_chunks.extend(chunks)

        if all_new_chunks:
            # Step 3: Embedding & Indexing
            embeddings = self.embedder.generate_embeddings(all_new_chunks)
            self.vector_store.add_to_index(embeddings, all_new_chunks)
            self.vector_store.save()
            log_success("Incremental indexing complete.")

    def _wipe_database(self):
        """Deletes all local data to reset the application state."""
        folders = [INPUT_DATA_FOLDER, OUTPUT_DATA_FOLDER, PROCESSED_DATA_FOLDER, VECTOR_STORE_FOLDER]
        for folder in folders:
            if os.path.exists(folder):
                shutil.rmtree(folder)
        self._ensure_folders()

    def _ensure_folders(self):
        folders = [INPUT_DATA_FOLDER, OUTPUT_DATA_FOLDER, PROCESSED_DATA_FOLDER, VECTOR_STORE_FOLDER]
        for folder in folders:
            os.makedirs(folder, exist_ok=True)

        log_info("Ensured all necessary folders exist.")