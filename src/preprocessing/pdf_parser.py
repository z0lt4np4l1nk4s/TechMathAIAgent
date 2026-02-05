import fitz  # PyMuPDF
import os
from preprocessing.ocr_engine import OCREngine
from common.config import INPUT_DATA_FOLDER
from utils.logging import log_info

class PDFParser:
    """
    Handles PDF document parsing by combining native text extraction 
    with OCR fallback for scanned pages.
    """
    def __init__(self):
        """
        Initializes the PDF parser and the underlying OCR engine (e.g., EasyOCR).
        """
        self.ocr_engine = OCREngine()
        # Threshold: if a page has fewer than 50 characters, we assume it's a scan
        self.min_text_threshold = 50  

    def parse_file(self, file_path):
        """
        Extracts full text from a PDF. Processes page by page to decide 
        between digital extraction and OCR.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_name = os.path.basename(file_path)
        log_info(f"Opening PDF for parsing: {file_name}")

        doc = fitz.open(file_path)
        extracted_content = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            # Attempt to extract embedded digital text
            text = page.get_text().strip()

            # Logic: If the digital text is sparse, treat the page as an image
            if len(text) < self.min_text_threshold:
                log_info(f"[{file_name}] Page {page_num + 1}: Scanned page detected. Running OCR...")
                
                # Render page to a high-resolution image for the OCR engine
                # Matrix(2, 2) creates a 2x zoom (300 DPI equivalent) for better character recognition
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  
                img_bytes = pix.tobytes("png")
                
                # Use the OCR engine to read text from the rendered image
                text = self.ocr_engine.extract_from_image(img_bytes)
            else:
                log_info(f"[{file_name}] Page {page_num + 1}: Digital text extracted.")

            extracted_content.append(text)

        doc.close()
        full_text = "\n\n".join(extracted_content)
        log_info(f"Successfully parsed {file_name}. Total length: {len(full_text)} characters.")
        
        return full_text