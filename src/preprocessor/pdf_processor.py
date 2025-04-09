import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import fitz  # PyMuPDF
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Process PDF documents into structured text.
    
    This class handles the extraction and structuring of text
    from PDF documents downloaded from the RBI website.
    """
    
    def __init__(self, input_dir: str = "data/raw", output_dir: str = "data/processed"):
        """Initialize the PDF processor.
        
        Args:
            input_dir: Directory containing raw PDF documents
            output_dir: Directory to store processed text
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text from PDF
        """
        logger.info(f"Extracting text from PDF: {pdf_path}")
        
        try:
            # Open the PDF file
            doc = fitz.open(pdf_path)
            
            # Extract text from each page
            text = ""
            for page_num, page in enumerate(doc):
                # Extract text from the page
                page_text = page.get_text()
                
                # Add page number as a marker
                text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            
            # Close the document
            doc.close()
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return ""
    
    def _extract_document_structure(self, text: str, metadata: Dict) -> List[Dict]:
        """Extract the document structure with hierarchy from PDF text.
        
        Args:
            text: Extracted text from PDF
            metadata: Document metadata
            
        Returns:
            List of dictionaries representing document chunks with hierarchy
        """
        chunks = []
        
        # Split text into pages based on the markers we added
        pages = re.split(r"--- Page \d+ ---", text)
        pages = [page.strip() for page in pages if page.strip()]
        
        # Initialize variables to track current section
        current_section = {"title": "Main", "level": 0}
        current_subsection = None
        
        # Process each page
        for page_num, page_text in enumerate(pages):
            # Split page into paragraphs
            paragraphs = [p.strip() for p in page_text.split('\n') if p.strip()]
            
            for para in paragraphs:
                # Skip very short paragraphs or known boilerplate
                if len(para) < 10 or re.search(r"^(Yours faithfully|Yours sincerely|Encl:|Enclosure:)", para):
                    continue
                
                # Check if this is a major heading (e.g., CHAPTER)
                # Case-insensitive matching for chapter
                chapter_match = re.match(r"^(CHAPTER)\s+([\dIVX]+)[:\.\s]?\s*(.*)", para, re.IGNORECASE)
                if chapter_match:
                    title = f"Chapter {chapter_match.group(2)}: {chapter_match.group(3)}".strip().rstrip(':')
                    current_section = {"title": title, "level": 1}
                    current_subsection = None
                    logger.debug(f"PDF Section Found: {title}")
                    continue # Don't treat heading itself as a chunk
                
                # Check if this is a subsection heading (e.g., starts with number.number or roman numeral.)
                subsection_match = re.match(r"^(\d+\.\d+|[ivx]+\.)\s+(.*)", para)
                if subsection_match:
                    title = para # Use the full line as subsection title for now
                    current_subsection = {"title": title, "level": 2}
                    logger.debug(f"PDF Subsection Found: {title}")
                    # Decide whether to continue or treat as chunk - for now, treat as chunk
                    # continue
                
                # Check if this is a numbered clause (similar to HTML processor)
                # Use a simpler regex for PDF as formatting is less reliable
                clause_match = re.match(r"^((?:Clause|Para|Section)\s+)?(\d+\.\d*|[ivx]+\.|[a-z]\.|\([a-z]\))\s*", para, re.IGNORECASE)
                clause_id = None
                if clause_match:
                    clause_id = "".join(filter(None, clause_match.groups()))
                    clause_id = clause_id.strip() # Keep trailing dot if present
                
                # Create chunk
                chunk = {
                    "text": para,
                    "metadata": {
                        **metadata,
                        "page": page_num + 1,
                        "section": current_section["title"],
                        "section_level": current_section["level"]
                    }
                }
                
                # Add subsection if available
                if current_subsection:
                    chunk["metadata"]["subsection"] = current_subsection["title"]
                    chunk["metadata"]["subsection_level"] = current_subsection["level"]
                
                # Add clause ID if available
                if clause_id:
                    chunk["metadata"]["clause_id"] = clause_id
                
                chunks.append(chunk)
        
        return chunks
    
    def process_document(self, pdf_path: str, metadata_path: Optional[str] = None) -> List[Dict]:
        """Process a single PDF document.
        
        Args:
            pdf_path: Path to PDF file
            metadata_path: Path to metadata file (optional)
            
        Returns:
            List of dictionaries representing document chunks
        """
        logger.info(f"Processing PDF document: {pdf_path}")
        
        # Extract text from PDF
        text = self._extract_text_from_pdf(pdf_path)
        
        if not text:
            logger.warning(f"No text extracted from PDF: {pdf_path}")
            return []
        
        # Load metadata if available
        metadata = {}
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        metadata[key.strip()] = value.strip()
        
        # Add document type to metadata
        metadata["document_type"] = "pdf"
        metadata["source_file"] = pdf_path
        
        # Extract document structure
        chunks = self._extract_document_structure(text, metadata)
        
        return chunks
    
    def process_all_documents(self, doc_type: str = "circulars") -> List[Dict]:
        """Process all PDF documents of a specific type.
        
        Args:
            doc_type: Type of documents to process (circulars, notifications)
            
        Returns:
            List of dictionaries representing all document chunks
        """
        logger.info(f"Processing all PDF {doc_type}")
        
        all_chunks = []
        input_dir = self.input_dir / doc_type
        
        # Get all PDF files
        pdf_files = list(input_dir.glob("*.pdf"))
        
        for pdf_file in tqdm(pdf_files, desc=f"Processing PDF {doc_type}"):
            # Find corresponding metadata file
            metadata_file = input_dir / f"{pdf_file.stem}.metadata.txt"
            
            # Process document
            chunks = self.process_document(str(pdf_file), str(metadata_file) if metadata_file.exists() else None)
            all_chunks.extend(chunks)
        
        # Save all chunks to a single file
        output_file = self.output_dir / f"{doc_type}_pdf_processed.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk) + "\n")
        
        logger.info(f"Processed {len(all_chunks)} chunks from {len(pdf_files)} PDF {doc_type}")
        return all_chunks


if __name__ == "__main__":
    # Example usage
    processor = PDFProcessor()
    
    # Process all PDF circulars
    circular_chunks = processor.process_all_documents("circulars")
    
    # Process all PDF notifications
    notification_chunks = processor.process_all_documents("notifications")
    
    print(f"Processed {len(circular_chunks)} PDF circular chunks")
    print(f"Processed {len(notification_chunks)} PDF notification chunks")