import os
import fitz  # PyMuPDF
from preprocessing.ocr_engine import OCREngine
from utils.logging import log_info

class DocumentParser:
    """
    Unified parser that handles both PDF documents and standalone images.
    It automatically routes files to the appropriate extraction method.
    """
    def __init__(self):
        # Initialize the OCR engine once to keep it in memory
        self.ocr_engine = OCREngine()
        self.supported_images = ('.png', '.jpg', '.jpeg')

    def parse(self, file_path):
        """
        Main entry point: Automatically detects file type and extracts text.
        Returns the full string content of the document.
        """
        ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)

        log_info(f"Processing file: {file_name} (Format: {ext})")

        if ext == '.pdf':
            return self._handle_pdf(file_path)
        elif ext in self.supported_images:
            log_info(f"Routing {file_name} to direct Image OCR.")
            return self._handle_image(file_path)
        else:
            log_info(f"Skipping unsupported format: {ext}")
            return None

    def _handle_pdf(self, file_path):
        """
        Internal method to process PDF pages. Uses a hybrid approach:
        native text extraction with OCR fallback for scans.
        """
        doc = fitz.open(file_path)
        full_text = []

        for page in doc:
            # Try fast digital extraction first
            text = page.get_text().strip()
            
            # Logic: If text is sparse (less than 50 chars), it's likely an image/scan
            if len(text) < 50:
                # Render page at 2x scale for better OCR accuracy
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_bytes = pix.tobytes("png")
                text = self.ocr_engine.extract_from_image(img_bytes)
            
            full_text.append(text)
        
        doc.close()
        return "\n\n".join(full_text)

    def _handle_image(self, file_path):
        """
        Directly sends an image file to the OCR engine.
        """
        return self.ocr_engine.extract_from_image(file_path)