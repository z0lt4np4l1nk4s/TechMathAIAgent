import easyocr
from utils.logging import log_info, log_error

class OCREngine:
    """
    Core engine for Optical Character Recognition.
    Converts image data (pixels) into structured text strings.
    """
    def __init__(self, languages=['hr', 'en']):
        """
        Initializes the EasyOCR reader. 
        Note: gpu=False is used here to stay within your 6GB VRAM limit,
        leaving the GPU memory available for the Mistral LLM.
        """
        log_info(f"Initializing EasyOCR for languages: {languages}")
        # The model weights will be downloaded on the first run if not present
        self.reader = easyocr.Reader(languages, gpu=False)

    def extract_from_image(self, image_data):
        """
        Performs OCR on various inputs: file path, bytes, or numpy array.
        
        Args:
            image_data: The image source (path string or byte buffer).
            
        Returns:
            A cleaned, single string containing all detected text.
        """
        try:
            log_info("OCREngine: Processing image data...")
            # detail=0 returns a simple list of strings, skipping bounding box coordinates
            results = self.reader.readtext(image_data, detail=0)
            
            extracted_text = " ".join(results)
            log_info(f"OCREngine: Extraction complete ({len(extracted_text)} characters found).")
            
            return extracted_text
        except Exception as e:
            # Captures potential issues like corrupted image buffers or memory limits
            log_error(f"OCR failed: {e}")
            return ""