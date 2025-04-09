import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import fitz  # PyMuPDF
from bs4 import BeautifulSoup, NavigableString
from tqdm import tqdm
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseProcessor:
    """Base class for document processing."""
    
    def __init__(self, input_dir: str = "data/raw", output_dir: str = "data/processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove residual page numbers or common headers/footers if possible
        text = re.sub(r"Master Circular \u2013 .*dated April \d, \d{4}", "", text).strip()
        text = re.sub(r"Page \d+ of \d+", "", text).strip()
        # Further cleaning rules can be added here
        return text

    def load_metadata(self, metadata_path: Optional[str]) -> Dict:
        metadata = {}
        if metadata_path and os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if ":" in line:
                            key, value = line.split(":", 1)
                            metadata[key.strip()] = value.strip()
            except Exception as e:
                 logger.error(f"Error reading metadata file {metadata_path}: {e}")
        return metadata

    def process_document(self, file_path: str, metadata_path: Optional[str] = None) -> List[Dict]:
        raise NotImplementedError

    def process_all_documents(self, doc_type: str = "circulars") -> List[Dict]:
        raise NotImplementedError

class PDFProcessor(BaseProcessor):
    """Process PDF documents into structured text."""
    
    def _extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """Extract text from each page of a PDF file."""
        logger.info(f"Extracting text from PDF: {pdf_path}")
        page_texts = {}
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                page_texts[page_num + 1] = self._clean_text(page_text)
            doc.close()
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
        return page_texts

    def _extract_document_structure(self, page_texts: Dict[int, str], metadata: Dict) -> List[Dict]:
        """Extract the document structure with hierarchy from PDF text.
           Attempts to identify sections and clauses based on common patterns.
        """
        chunks = []
        current_section = {"title": metadata.get("title", "Main"), "level": 0}
        current_subsection = None
        
        # Regex for potential headings/clauses
        # Matches patterns like: CHAPTER II:, 2., 2.1, 2.1.1, (a), (i), etc.
        clause_heading_pattern = re.compile(
            r"^\s*" + 
            r"(?:(?:CHAPTER|SECTION|PART)\s+[IVXLCDM\d]+[:\.\s]*)?" + # Optional Chapter/Section/Part
            r"(?:(\d+(?:\.\d+)*)\.?\s+)?" + # Optional numeric prefix (e.g., 1., 1.1, 1.1.1)
            r"(?:(?:\([a-z]+\)|\([ivxlcdm]+\))\s+)?" + # Optional alpha/roman in parens (e.g., (a), (i))
            r"([A-Z][A-Za-z\s,()&\-/]+?)" + # Heading text (Starts with Cap)
            r"\s*$", re.IGNORECASE
        )
        # Simpler pattern for just clause numbers
        clause_num_pattern = re.compile(r"^\s*(\d+(?:\.\d+)*|[a-z]\.|\([a-z]\)|[ivxlcdm]+\.|\([ivxlcdm]+\))\s*")

        for page_num, page_text in sorted(page_texts.items()):
            paragraphs = [p.strip() for p in page_text.split('\n') if p.strip()]
            
            for para in paragraphs:
                # Skip very short paragraphs or known boilerplate
                if len(para) < 15 or re.match(r"^(?:Yours faithfully|Encl:|Annex(?:ure)?[:.]?$)", para, re.IGNORECASE):
                    continue

                chunk_text = self._clean_text(para)
                if not chunk_text:
                    continue
                
                heading_match = clause_heading_pattern.match(chunk_text)
                clause_num_match = clause_num_pattern.match(chunk_text)
                clause_id = None

                # Is it a potential heading?
                if heading_match and len(heading_match.group(2)) > 5: # Check if title part is reasonably long
                    num_prefix = heading_match.group(1)
                    title_text = heading_match.group(2).strip()
                    
                    if num_prefix:
                        level = num_prefix.count('.') + 1
                        if level <= 1:
                            current_section = {"title": title_text, "level": 1}
                            current_subsection = None
                            logger.debug(f"PDF L1 Section: {num_prefix} {title_text}")
                        else:
                            current_subsection = {"title": title_text, "level": level}
                            logger.debug(f"PDF L{level} Subsection: {num_prefix} {title_text}")
                        clause_id = num_prefix.rstrip('.')
                        # Don't treat the heading itself as a chunk if it looks structural
                        # continue
                    elif chunk_text.isupper() and len(chunk_text.split()) < 8: # Assume all caps = heading
                        current_section = {"title": chunk_text, "level": 1}
                        current_subsection = None
                        logger.debug(f"PDF L1 Section (Caps): {chunk_text}")
                        # continue
                # Is it just a clause number? 
                elif clause_num_match:
                    clause_id = clause_num_match.group(1).rstrip('.')
                
                chunk = {
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "page": page_num,
                        "section": current_section["title"],
                        "section_level": current_section["level"]
                    }
                }
                
                if current_subsection:
                    chunk["metadata"]["subsection"] = current_subsection["title"]
                    chunk["metadata"]["subsection_level"] = current_subsection.get("level", 2)
                
                if clause_id:
                    chunk["metadata"]["clause_id"] = clause_id
                
                chunks.append(chunk)
        
        return chunks
    
    def process_document(self, pdf_path: str, metadata_path: Optional[str] = None) -> List[Dict]:
        logger.info(f"Processing PDF document: {pdf_path}")
        page_texts = self._extract_text_from_pdf(pdf_path)
        if not page_texts:
            return []
        metadata = self.load_metadata(metadata_path)
        metadata["document_type"] = "pdf"
        metadata["source_file"] = str(pdf_path)
        return self._extract_document_structure(page_texts, metadata)

    def process_all_documents(self, doc_type: str = "notifications") -> List[Dict]:
        logger.info(f"Processing all PDF {doc_type}")
        all_chunks = []
        input_dir = self.input_dir / doc_type
        pdf_files = list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.PDF"))
        
        for pdf_file in tqdm(pdf_files, desc=f"Processing PDF {doc_type}"):
            metadata_file = input_dir / f"{pdf_file.stem}.metadata.txt"
            chunks = self.process_document(str(pdf_file), str(metadata_file) if metadata_file.exists() else None)
            all_chunks.extend(chunks)
        
        output_file = self.output_dir / f"{doc_type}_pdf_processed.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk) + "\n")
        
        logger.info(f"Processed {len(all_chunks)} chunks from {len(pdf_files)} PDF {doc_type}")
        return all_chunks

class HTMLProcessor(BaseProcessor):
    """Process HTML documents into structured text."""
    
    def _extract_main_content(self, soup: BeautifulSoup) -> Optional[BeautifulSoup]:
        content_div = soup.find("div", id="doublescroll")
        if content_div:
            table_content = content_div.find("table", class_="tablebg")
            if table_content:
                content_td = table_content.find("td", class_="tablecontent2")
                if content_td:
                    inner_table = content_td.find("table", class_="td")
                    if inner_table: return inner_table
            return content_div 
        logger.warning("Could not identify main content area.")
        return soup.body # Fallback to body

    def _extract_structure_recursive(self, tag: BeautifulSoup, metadata: Dict, current_section: str = "Main", current_level: int = 0) -> List[Dict]:
        chunks = []
        for child in tag.children:
            if isinstance(child, NavigableString):
                text = self._clean_text(child.string)
                if text and len(text) > 15: 
                    clause_id = None
                    # Basic check for numbered lists like (i), (a), 1.
                    prev = child.find_previous_sibling()
                    if prev and prev.name == 'b':
                        num_match = re.match(r'^\(?([ivxlcdm\d]+|[a-z])\)?\.?$', prev.get_text(strip=True).lower())
                        if num_match: clause_id = prev.get_text(strip=True)

                    if not clause_id:
                         # Pattern like 1. or 1.1 or (i) or i.
                        clause_match = re.match(r"^\s*(\d+(?:\.\d+)*\.?|\([a-z]+\)|[a-z]\.|\([ivxlcdm]+\)|[ivxlcdm]+\.)\s+", text, re.IGNORECASE)
                        if clause_match: clause_id = clause_match.group(1).strip().rstrip('.')
                            
                    chunk = {"text": text, "metadata": {**metadata, "section": current_section, "section_level": current_level}}
                    if clause_id: chunk["metadata"]["clause_id"] = clause_id
                    chunks.append(chunk)
                    
            elif child.name in ['p', 'div']: 
                chunks.extend(self._extract_structure_recursive(child, metadata, current_section, current_level))
            elif child.name in ['h1', 'h2', 'h3', 'h4']: 
                section_title = self._clean_text(child.get_text())
                new_level = current_level + 1 # Simple level increment
                # Don't treat heading itself as chunk, update context for subsequent ones
                # Recursive call with new section context
                # Need to handle siblings after the heading correctly
                # For now, we just pass the new context down. A better approach might be needed.
                # print(f"Found heading {new_level}: {section_title}")
                # Pass context down - siblings will inherit incorrectly currently
                # A better approach is needed to handle siblings after a heading
                # Simple approach: recurse on children of heading, assuming content is nested?
                # No, siblings should be processed at current level. Let's just pass context down for now.
                chunks.extend(self._extract_structure_recursive(child, metadata, section_title, new_level)) 

            elif child.name in ['table', 'ul', 'ol']: 
                list_text = self._clean_text(child.get_text(separator='\n'))
                if list_text and len(list_text) > 20:
                     chunks.append({"text": list_text, "metadata": {**metadata, "section": current_section, "section_level": current_level, "tag_type": child.name}})
            elif child.name in ['br', 'hr', 'script', 'style']: 
                continue
            elif hasattr(child, 'children'): # Recurse into other tags
                 chunks.extend(self._extract_structure_recursive(child, metadata, current_section, current_level))
        return chunks

    def process_document(self, html_path: str, metadata_path: Optional[str] = None) -> List[Dict]:
        logger.info(f"Processing HTML document: {html_path}")
        try:
            with open(html_path, "r", encoding="utf-8", errors='ignore') as f:
                html_content = f.read()
            soup = BeautifulSoup(html_content, 'html.parser')
            metadata = self.load_metadata(metadata_path)
            metadata["document_type"] = "html"
            metadata["source_file"] = str(html_path)
            main_content_tag = self._extract_main_content(soup)
            if not main_content_tag:
                return []
            return self._extract_structure_recursive(main_content_tag, metadata)
        except Exception as e:
            logger.error(f"Error processing HTML file {html_path}: {e}")
            return []
    
    def process_all_documents(self, doc_type: str = "notifications") -> List[Dict]:
        logger.info(f"Processing all HTML {doc_type}")
        all_chunks = []
        input_dir = self.input_dir / doc_type
        html_files = list(input_dir.glob("*.html")) + list(input_dir.glob("*.htm"))
        
        for html_file in tqdm(html_files, desc=f"Processing HTML {doc_type}"):
            metadata_file = input_dir / f"{html_file.stem}.metadata.txt"
            chunks = self.process_document(str(html_file), str(metadata_file) if metadata_file.exists() else None)
            all_chunks.extend(chunks)
        
        output_file = self.output_dir / f"{doc_type}_html_processed.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk) + "\n")
        
        logger.info(f"Processed {len(all_chunks)} chunks from {len(html_files)} HTML {doc_type}")
        return all_chunks

class TextProcessor:
    """Combines processing from different source types."""
    def __init__(self, input_dir: str = "data/raw", output_dir: str = "data/processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.pdf_processor = PDFProcessor(input_dir, output_dir)
        self.html_processor = HTMLProcessor(input_dir, output_dir)

    def process_all(self, doc_types: List[str] = ["circulars", "notifications"]):
        logger.info("Starting processing for all document types...")
        all_processed_chunks = []
        for doc_type in doc_types:
            logger.info(f"Processing type: {doc_type}")
            pdf_chunks = self.pdf_processor.process_all_documents(doc_type)
            html_chunks = self.html_processor.process_all_documents(doc_type)
            
            combined_chunks = pdf_chunks + html_chunks
            
            # Consolidate into a single file per doc_type
            output_file = self.output_dir / f"{doc_type}_processed.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                 # Deduplicate based on text content? Might be too slow. 
                 # Simple consolidation for now.
                 unique_texts = set()
                 final_chunks_for_type = []
                 for chunk in combined_chunks:
                    # Basic deduplication
                    if chunk['text'] not in unique_texts:
                        f.write(json.dumps(chunk) + "\n")
                        unique_texts.add(chunk['text'])
                        final_chunks_for_type.append(chunk)

            logger.info(f"Finished processing {doc_type}. Total unique chunks: {len(final_chunks_for_type)}")
            all_processed_chunks.extend(final_chunks_for_type)
            
        logger.info(f"Total processed chunks across all types: {len(all_processed_chunks)}")
        return all_processed_chunks

if __name__ == "__main__":
    processor = TextProcessor()
    processor.process_all()