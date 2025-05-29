import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import fitz  # PyMuPDF
from bs4 import BeautifulSoup, NavigableString
from tqdm import tqdm
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import multiprocessing # Import multiprocessing

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
        """Process all documents of a given type in the input directory."""
        doc_dir = self.input_dir / doc_type
        if not doc_dir.is_dir():
            logger.warning(f"Directory not found for doc_type '{doc_type}': {doc_dir}")
            return []
        
        all_chunks = []
        files_to_process = list(doc_dir.glob("*.pdf")) + list(doc_dir.glob("*.html")) + list(doc_dir.glob("*.htm"))
        
        logger.info(f"Found {len(files_to_process)} documents in {doc_dir}")
        
        for file_path in tqdm(files_to_process, desc=f"Processing {doc_type}"):
            # Basic metadata loading (assuming no specific .metadata files for now)
            metadata_path = None 
            try:
                # process_document should handle file type detection or be specific
                processed_data = self.process_document(str(file_path), metadata_path)
                if processed_data:
                     # Ensure metadata includes doc_type and filename
                    for chunk_dict in processed_data:
                        chunk_dict['metadata']['doc_type'] = doc_type
                        chunk_dict['metadata']['file_name'] = file_path.name # Add filename here
                    all_chunks.extend(processed_data)
            except Exception as e:
                logger.error(f"Error processing document {file_path}: {e}", exc_info=True)
                
        logger.info(f"Completed processing for {doc_type}. Total chunks: {len(all_chunks)}")
        return all_chunks

class PDFProcessor(BaseProcessor):
    """Process PDF documents, attempting to identify sections based on potential headings."""
    
    # Regex to identify potential headings (e.g., "1.", "1.1 ", "A.", "(i)", "Introduction", "Annexure")
    # This is a basic heuristic and might need refinement
    HEADING_PATTERN = re.compile(r"^\s*(?:(?:\d+\.\d*\s+)|(?:[A-Z]\.\s+)|(?:\([a-z]+\)\s+)|(?:\([ivx]+\)\s+)|(?:Introduction)|(?:Conclusion)|(?:Annex(?:ure)?(?:\s*[A-Z0-9]+)?))", re.IGNORECASE)
    
    def _extract_sections_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text page by page and group into sections based on HEADING_PATTERN."""
        logger.info(f"Extracting sections from PDF: {pdf_path}")
        sections = []
        current_section_title = f"Introduction ({Path(pdf_path).name})" # Default title
        current_section_texts = []
        current_page_num = 1
        
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                current_page_num = page_num + 1
                page_text = page.get_text("text") # Get plain text
                lines = page_text.split('\n')
                
                for line in lines:
                    cleaned_line = self._clean_text(line)
                    if not cleaned_line:
                        continue
                        
                    match = self.HEADING_PATTERN.match(cleaned_line)
                    # Check if it looks like a heading and isn't excessively long (likely paragraph)
                    if match and len(cleaned_line) < 100: 
                        # Found a potential heading, save the previous section
                        if current_section_texts:
                            sections.append({
                                "section_title": current_section_title,
                                "section_text": "\n".join(current_section_texts).strip(),
                                "page": sections[-1]["page"] if sections else 1, # Use page of last section start
                                # Metadata will be added later in process_document
                            })
                        # Start new section
                        current_section_title = cleaned_line # Use the heading line as title
                        current_section_texts = []
                        # Store page number where new section starts
                        if sections:
                             sections[-1]['end_page'] = current_page_num # Mark end page of previous
                        current_section_start_page = current_page_num
                        
                    else: # Not a heading, add to current section text
                        current_section_texts.append(cleaned_line)
                        
            # Add the last section
            if current_section_texts:
                 sections.append({
                     "section_title": current_section_title,
                     "section_text": "\n".join(current_section_texts).strip(),
                     "page": current_section_start_page if 'current_section_start_page' in locals() else 1
                 })
            if sections: sections[-1]['end_page'] = current_page_num # Mark end page of last section
                 
            doc.close()
            # Remove empty sections
            sections = [s for s in sections if s["section_text"]]
            logger.info(f"Extracted {len(sections)} potential sections from PDF {pdf_path}.")
            return sections
        
        except Exception as e:
            logger.error(f"Error extracting sections from PDF {pdf_path}: {e}", exc_info=True)
            return []

    def process_document(self, pdf_path_str: str, metadata_path: Optional[str] = None) -> List[Dict]:
        pdf_path = Path(pdf_path_str)
        logger.debug(f"Processing PDF document: {pdf_path.name}")
        base_metadata = self.load_metadata(metadata_path)
        # Add filename and default reference ID early
        base_metadata['file_name'] = pdf_path.name
        base_metadata.setdefault('reference_id', f'PDF_REF_{pdf_path.stem}')
        
        sections = self._extract_sections_from_pdf(pdf_path_str)
        
        if not sections:
            logger.warning(f"No sections extracted from {pdf_path.name}")
            return []
            
        # Apply text splitting to each section's text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Adjust as needed
            chunk_overlap=200, # Adjust as needed
            length_function=len,
            add_start_index=True,
        )
        
        final_chunks = []
        chunk_id_counter = 0
        for section in sections:
            section_text = section["section_text"]
            section_title = section["section_title"]
            section_page = section.get("page", 1) # Get page number if available

            split_texts = text_splitter.split_text(section_text)
            
            for i, chunk_text in enumerate(split_texts):
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    "section_title": section_title,
                    "page_number": section_page, # Store page number
                    "chunk_index_in_section": i,
                    "text": self._clean_text(chunk_text) # Store cleaned chunk text in metadata
                })
                # Generate a unique ID for the chunk within the document
                chunk_id = f"{base_metadata['reference_id']}_chunk_{chunk_id_counter}"
                chunk_id_counter += 1
                
                final_chunks.append({
                    "id": chunk_id,
                    "text": self._clean_text(chunk_text), # Text content for embedding
                    "metadata": chunk_metadata
                })
                
        logger.debug(f"Generated {len(final_chunks)} chunks from {pdf_path.name}")
        return final_chunks
        
    def _process_single_pdf_wrapper(self, args: Tuple[str, Optional[str], str]) -> List[Dict]:
        """Wrapper to process a single PDF, to be used with multiprocessing.Pool."""
        pdf_path_str, metadata_path, doc_type = args
        try:
            processed_data = self.process_document(pdf_path_str, metadata_path)
            if processed_data:
                for chunk_dict in processed_data:
                    # Ensure file_name is present if not already added by process_document
                    if 'file_name' not in chunk_dict['metadata']:
                         chunk_dict['metadata']['file_name'] = Path(pdf_path_str).name
                    chunk_dict['metadata']['doc_type'] = doc_type
                return processed_data
        except Exception as e:
            logger.error(f"Error processing PDF document {pdf_path_str} in worker: {e}", exc_info=True)
        return []
        
    def process_all_documents(self, doc_type: str = "circulars") -> List[Dict]:
        """Process all PDF documents of a given type in the input directory using multiprocessing."""
        doc_dir = self.input_dir / doc_type
        if not doc_dir.is_dir():
            logger.warning(f"PDF Directory not found for doc_type '{doc_type}': {doc_dir}")
            return []
        
        pdf_files = list(doc_dir.glob("*.pdf")) + list(doc_dir.glob("*.PDF"))
        logger.info(f"Found {len(pdf_files)} PDF documents in {doc_dir} for parallel processing.")
        
        if not pdf_files:
            return []

        all_chunks = []
        # Prepare arguments for the pool
        # Assuming no specific metadata files for now, so metadata_path is None
        tasks = [(str(pdf_path), None, doc_type) for pdf_path in pdf_files]

        # Determine number of processes (e.g., number of CPU cores)
        num_processes = multiprocessing.cpu_count()
        logger.info(f"Using {num_processes} processes for PDF processing.")

        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap(self._process_single_pdf_wrapper, tasks), total=len(tasks), desc=f"Processing PDF {doc_type}"))
        
        for result_list in results:
            if result_list: # Filter out empty lists from failed processing
                all_chunks.extend(result_list)
                
        logger.info(f"Completed parallel processing PDFs for {doc_type}. Total chunks: {len(all_chunks)}")
        return all_chunks

class HTMLProcessor(BaseProcessor):
    """Process HTML documents into structured sections based on headings."""
    
    HEADING_TAGS = [f"h{i}" for i in range(1, 7)] # h1, h2, ..., h6

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

    def _process_single_html_wrapper(self, args: Tuple[str, Optional[str], str]) -> List[Dict]:
        """Wrapper to process a single HTML, to be used with multiprocessing.Pool."""
        html_path_str, metadata_path, doc_type = args
        try:
            processed_data = self.process_document(html_path_str, metadata_path)
            if processed_data:
                for chunk_dict in processed_data:
                     # Ensure file_name is present if not already added by process_document
                    if 'file_name' not in chunk_dict['metadata']:
                         chunk_dict['metadata']['file_name'] = Path(html_path_str).name
                    chunk_dict['metadata']['doc_type'] = doc_type
                return processed_data
        except Exception as e:
            logger.error(f"Error processing HTML document {html_path_str} in worker: {e}", exc_info=True)
        return []

    def process_document(self, html_path: str, metadata_path: Optional[str] = None) -> List[Dict]:
        logger.info(f"Processing HTML document into sections: {html_path}")
        try:
            with open(html_path, "r", encoding="utf-8", errors='ignore') as f:
                html_content = f.read()
            soup = BeautifulSoup(html_content, 'html.parser')
            
            base_metadata = self.load_metadata(metadata_path)
            base_metadata["document_type"] = "html"
            base_metadata["source_file"] = str(html_path)
            base_metadata.setdefault('reference_id', f'UNKNOWN_HTML_REF_{Path(html_path).stem}')

            main_content_tag = self._extract_main_content(soup)
            if not main_content_tag:
                logger.warning(f"No main content found for {html_path}, returning empty list.")
                return []

            sections = []
            current_section_title = f"Introduction ({Path(html_path).name})" # Default title
            current_section_texts = []

            for element in main_content_tag.find_all(recursive=False): # Iterate direct children
                element_text = self._clean_text(element.get_text(separator=' ', strip=True))
                
                if element.name in self.HEADING_TAGS and element_text:
                    # Found a heading, save the previous section if it has text
                    if current_section_texts:
                        sections.append({
                            "section_title": current_section_title,
                            "section_text": "\n".join(current_section_texts).strip(),
                            "metadata": base_metadata.copy() # Base metadata for the section
                        })
                    # Start a new section
                    current_section_title = element_text
                    current_section_texts = [] 
                elif element_text: # Non-heading element with text
                    current_section_texts.append(element_text)

            # Add the last section if it has text
            if current_section_texts:
                sections.append({
                    "section_title": current_section_title,
                    "section_text": "\n".join(current_section_texts).strip(),
                     "metadata": base_metadata.copy()
                })
                
            # Remove empty sections just in case
            sections = [s for s in sections if s["section_text"]]

            logger.info(f"Extracted {len(sections)} sections from HTML {html_path}.")
            return sections

        except Exception as e:
            logger.error(f"Error processing HTML file {html_path} into sections: {e}", exc_info=True)
            return []

    def process_all_documents(self, doc_type: str = "circulars") -> List[Dict]:
        """Process all HTML documents of a given type in the input directory using multiprocessing."""
        doc_dir = self.input_dir / doc_type
        if not doc_dir.is_dir():
            logger.warning(f"HTML Directory not found for doc_type '{doc_type}': {doc_dir}")
            return []
        
        html_files = list(doc_dir.glob("*.html")) + list(doc_dir.glob("*.htm")) + list(doc_dir.glob("*.HTML")) + list(doc_dir.glob("*.HTM"))
        logger.info(f"Found {len(html_files)} HTML documents in {doc_dir} for parallel processing.")

        if not html_files:
            return []

        all_chunks = []
        # Prepare arguments for the pool
        tasks = [(str(html_file), None, doc_type) for html_file in html_files]

        num_processes = multiprocessing.cpu_count()
        logger.info(f"Using {num_processes} processes for HTML processing.")

        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap(self._process_single_html_wrapper, tasks), total=len(tasks), desc=f"Processing HTML {doc_type}"))

        for result_list in results:
            if result_list:
                all_chunks.extend(result_list)
                
        logger.info(f"Completed parallel processing HTML for {doc_type}. Total chunks: {len(all_chunks)}")
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
            # Explicitly call the implemented methods
            pdf_chunks = self.pdf_processor.process_all_documents(doc_type)
            html_chunks = self.html_processor.process_all_documents(doc_type)
            
            combined_chunks = pdf_chunks + html_chunks
            
            output_file = self.output_dir / f"{doc_type}_processed.jsonl"
            unique_texts = set() # Track unique chunk texts to avoid duplicates
            final_chunks_for_type = []
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                     for chunk in combined_chunks:
                        # Ensure 'text' key exists and is not empty before checking uniqueness
                        if 'text' in chunk and chunk['text'] and chunk['text'] not in unique_texts:
                            # Add required metadata if missing
                            if 'metadata' not in chunk:
                                chunk['metadata'] = {}
                            chunk['metadata'].setdefault('doc_type', doc_type)
                            chunk['metadata'].setdefault('file_name', chunk.get('metadata',{}).get('file_name', 'Unknown'))
                            chunk['metadata'].setdefault('reference_id', chunk.get('metadata',{}).get('reference_id', f'UNKNOWN_REF_{doc_type}'))
                            
                            # Ensure the main 'text' field is also present for embedding/storage
                            chunk.setdefault('text', chunk.get('metadata',{}).get('text', ''))
                            
                            f.write(json.dumps(chunk) + "\n")
                            unique_texts.add(chunk['text'])
                            final_chunks_for_type.append(chunk)
                        elif 'text' not in chunk or not chunk['text']:
                            logger.warning(f"Skipping chunk with missing or empty text: {chunk.get('id', 'N/A')}")
            except IOError as e:
                 logger.error(f"Error writing processed file {output_file}: {e}")
                 continue # Move to the next doc_type if writing fails

            logger.info(f"Finished processing {doc_type}. Total unique chunks: {len(final_chunks_for_type)}")
            all_processed_chunks.extend(final_chunks_for_type)
            
        logger.info(f"Total processed chunks across all types: {len(all_processed_chunks)}")
        return all_processed_chunks

if __name__ == "__main__":
    processor = TextProcessor()
    processor.process_all()