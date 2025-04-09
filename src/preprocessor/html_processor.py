import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from bs4 import BeautifulSoup, NavigableString
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HTMLProcessor:
    """Process HTML documents into structured text.
    
    This class handles the parsing and structuring of text
    from HTML documents downloaded from the RBI website.
    """
    
    def __init__(self, input_dir: str = "data/raw", output_dir: str = "data/processed"):
        """Initialize the HTML processor.
        
        Args:
            input_dir: Directory containing raw HTML documents
            output_dir: Directory to store processed text
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def _clean_text(self, text: str) -> str:
        """Clean extracted text.
        
        Args:
            text: Raw extracted text
        
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Add more cleaning rules as needed
        return text

    def _extract_main_content(self, soup: BeautifulSoup) -> Optional[BeautifulSoup]:
        """Extract the main content area from the RBI HTML structure.
        
        Args:
            soup: BeautifulSoup object representing the parsed HTML
            
        Returns:
            BeautifulSoup object for the main content, or None if not found
        """
        # RBI pages often have a table structure containing the main content
        # Try to find a specific table or div ID/class that holds the circular text
        # Example selectors (need inspection of actual RBI pages):
        content_div = soup.find("div", id="doublescroll")  # Found in 2022-06-10_.html
        if content_div:
            table_content = content_div.find("table", class_="tablebg")
            if table_content:
                content_td = table_content.find("td", class_="tablecontent2")
                if content_td:
                    inner_table = content_td.find("table", class_="td")
                    if inner_table:
                        return inner_table # Return the innermost table containing the text
            # Fallback if the specific structure isn't found
            return content_div 
            
        # Add more fallback logic if needed based on other RBI page structures
        logger.warning("Could not identify main content area using known selectors.")
        return None

    def _extract_structure_recursive(self, tag: BeautifulSoup, metadata: Dict, current_section: str = "Main", current_level: int = 0, clause_counter: Dict = None) -> List[Dict]:
        """Recursively extract structured text chunks from HTML tags.
        
        Args:
            tag: Current HTML tag to process
            metadata: Document metadata
            current_section: Title of the current section
            current_level: Hierarchy level of the current section
            clause_counter: Dictionary to track clause numbering (e.g., {'1': 0, '1.1': 0})
            
        Returns:
            List of dictionaries representing document chunks
        """
        chunks = []
        if clause_counter is None:
            clause_counter = {'level1': 0, 'level2': 0, 'level3': 0} # Simple counter for now

        for child in tag.children:
            if isinstance(child, NavigableString):
                text = self._clean_text(child.string)
                if text and len(text) > 10: # Avoid very short strings
                    # Attempt to identify clause based on surrounding tags or text patterns
                    clause_id = None
                    # Example: Check if previous sibling was a bold number
                    prev_sibling = child.find_previous_sibling()
                    if prev_sibling and prev_sibling.name == 'b':
                        # Basic check for numbered lists like (i), (a), 1.
                        num_match = re.match(r'^\(?([ivxlcdm\d]+|[a-z])\)?\.?$', prev_sibling.get_text(strip=True).lower())
                        if num_match:
                           clause_id = prev_sibling.get_text(strip=True)
                           # Maybe consume the sibling text so it's not duplicated? Careful needed.
                           
                    # Fallback pattern matching on the text itself
                    if not clause_id:
                        clause_match = re.match(r"^((?:Clause|Para|Section)\s+)?(\d+\.\d*|[ivx]+\.|[a-z]\.|\([a-z]\))\s+", text, re.IGNORECASE)
                        if clause_match:
                             clause_id = clause_match.group(2)

                    chunk = {
                        "text": text,
                        "metadata": {
                            **metadata,
                            "section": current_section,
                            "section_level": current_level,
                        }
                    }
                    if clause_id:
                        chunk["metadata"]["clause_id"] = clause_id
                        
                    chunks.append(chunk)
            elif child.name == 'p' or child.name == 'div': # Paragraphs or divs might contain text or further structure
                chunks.extend(self._extract_structure_recursive(child, metadata, current_section, current_level, clause_counter))
            elif child.name in ['h1', 'h2', 'h3', 'h4']: # Handle headings
                section_title = self._clean_text(child.get_text())
                new_level = current_level + 1
                # Don't treat heading itself as a chunk, just update context for subsequent chunks
                chunks.extend(self._extract_structure_recursive(child, metadata, section_title, new_level, clause_counter))
            elif child.name in ['table', 'ul', 'ol']: # Handle tables and lists
                # For simplicity, extract text content from tables/lists as a single chunk
                list_text = self._clean_text(child.get_text())
                if list_text and len(list_text) > 15:
                     chunks.append({
                        "text": list_text,
                        "metadata": {
                            **metadata,
                            "section": current_section,
                            "section_level": current_level,
                            "tag_type": child.name # Indicate it came from a list/table
                        }
                    })
            elif child.name in ['br', 'hr']: # Ignore breaks and rules
                continue
            elif hasattr(child, 'children'): # Recurse into other tags that might have content
                 chunks.extend(self._extract_structure_recursive(child, metadata, current_section, current_level, clause_counter))
                 
        return chunks

    def process_document(self, html_path: str, metadata_path: Optional[str] = None) -> List[Dict]:
        """Process a single HTML document.
        
        Args:
            html_path: Path to HTML file
            metadata_path: Path to metadata file (optional)
            
        Returns:
            List of dictionaries representing document chunks
        """
        logger.info(f"Processing HTML document: {html_path}")
        
        try:
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Load metadata if available
            metadata = {}
            if metadata_path and os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if ":" in line:
                            key, value = line.split(":", 1)
                            metadata[key.strip()] = value.strip()
            
            # Add document type to metadata
            metadata["document_type"] = "html"
            metadata["source_file"] = html_path
            
            # Extract main content area
            main_content_tag = self._extract_main_content(soup)
            
            if not main_content_tag:
                logger.warning(f"No main content found in HTML: {html_path}. Processing entire body.")
                main_content_tag = soup.body if soup.body else soup # Fallback to body or whole soup
            
            # Extract structured text
            chunks = self._extract_structure_recursive(main_content_tag, metadata)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing HTML file {html_path}: {e}")
            return []

    def process_all_documents(self, doc_type: str = "circulars") -> List[Dict]:
        """Process all HTML documents of a specific type.
        
        Args:
            doc_type: Type of documents to process (circulars, notifications)
            
        Returns:
            List of dictionaries representing all document chunks
        """
        logger.info(f"Processing all HTML {doc_type}")
        
        all_chunks = []
        input_dir = self.input_dir / doc_type
        
        # Get all HTML files
        html_files = list(input_dir.glob("*.html")) + list(input_dir.glob("*.htm"))
        
        for html_file in tqdm(html_files, desc=f"Processing HTML {doc_type}"):
            # Find corresponding metadata file
            metadata_file = input_dir / f"{html_file.stem}.metadata.txt"
            
            # Process document
            chunks = self.process_document(str(html_file), str(metadata_file) if metadata_file.exists() else None)
            all_chunks.extend(chunks)
        
        # Save all chunks to a single file
        output_file = self.output_dir / f"{doc_type}_html_processed.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk) + "\n")
        
        logger.info(f"Processed {len(all_chunks)} chunks from {len(html_files)} HTML {doc_type}")
        return all_chunks


if __name__ == "__main__":
    # Example usage
    processor = HTMLProcessor()
    
    # Process all HTML circulars
    circular_chunks = processor.process_all_documents("circulars")
    
    # Process all HTML notifications
    notification_chunks = processor.process_all_documents("notifications")
    
    print(f"Processed {len(circular_chunks)} HTML circular chunks")
    print(f"Processed {len(notification_chunks)} HTML notification chunks") 