"""PDF processing utilities for encoding and compression."""

import base64
from typing import Optional
import pymupdf  # PyMuPDF
from .config import MAX_PDF_SIZE_MB


def encode_pdf_to_base64(pdf_path: str, max_size_mb: int = MAX_PDF_SIZE_MB) -> str:
    """Convert PDF to base64 string with size limit and compression.
    
    Args:
        pdf_path: Path to the PDF file
        max_size_mb: Maximum size in MB for the base64 encoded string
        
    Returns:
        Base64 encoded PDF string
    """
    # Read and potentially compress the PDF
    doc = pymupdf.open(pdf_path)
    
    # Try to optimize the PDF if it's too large
    pdf_bytes = doc.write()
    encoded = base64.b64encode(pdf_bytes).decode('utf-8')
    
    # If the encoded size is too large, compress by reducing image quality
    if len(encoded) > max_size_mb * 1024 * 1024:
        # Create a new PDF with compressed images
        new_doc = pymupdf.open()
        for page in doc:
            new_page = new_doc.new_page(width=page.rect.width, height=page.rect.height)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(1, 1))  # Reduce resolution
            new_page.insert_image(page.rect, pixmap=pix)
        
        # Get the compressed version
        pdf_bytes = new_doc.write()
        new_doc.close()
        encoded = base64.b64encode(pdf_bytes).decode('utf-8')
    
    doc.close()
    return encoded
