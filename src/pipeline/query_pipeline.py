import os

import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import tempfile
import json
from tqdm import tqdm

import numpy as np
from sentence_transformers import SentenceTransformer

from src.preprocessor.text_processor import TextProcessor
from src.embedding.embedder import DocumentEmbedder
from src.retrieval.qdrant_retriever import QdrantRetriever
from src.llm.gemini_interface import GeminiInterface

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Range
from langchain_community.vectorstores import Qdrant
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Text Splitter Configuration
TEXT_SPLITTER_CHUNK_SIZE = 1000
TEXT_SPLITTER_CHUNK_OVERLAP = 200
TEXT_SPLITTER_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

class QueryPipeline:
    """End-to-end pipeline for analyzing document compliance.
    
    This class orchestrates the process of analyzing user documents for compliance
    by retrieving relevant regulations and generating insights using LLM.
    """
    
    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        qdrant_collection: str = "rbi_regulations",
        embedding_model_name: str = "msmarco-distilbert-base-tas-b",
        llm_model_name: str = "models/gemini-1.5-flash",
        google_api_key: Optional[str] = None,
    ):
        """Initialize the query pipeline.
        
        Args:
            qdrant_url: Qdrant Cloud URL (e.g., https://your-cluster.cloud.qdrant.io). Defaults to local if not set.
            qdrant_api_key: Qdrant API Key. Required if qdrant_url is for a secured cloud instance.
            qdrant_collection: Qdrant collection name
            embedding_model_name: Name of the embedding model to use
            llm_model_name: Name of the LLM model to use
            google_api_key: Google API key for Gemini (defaults to GOOGLE_API_KEY env var)
        """
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.qdrant_collection = qdrant_collection
        self.embedding_model_name = embedding_model_name # Ensure this is used for ST
        
        # Initialize Qdrant client (used for direct search)
        qdrant_conn_port = None
        if self.qdrant_url:
            if "cloud.qdrant.io" in self.qdrant_url and self.qdrant_url.startswith("https://"):
                qdrant_conn_port = 6333 # Explicitly set for cloud HTTPS REST
            self.qdrant_client = QdrantClient(
                url=self.qdrant_url, 
                port=qdrant_conn_port,
                api_key=self.qdrant_api_key,
                prefer_grpc=False 
            )
            logger.info(f"QueryPipeline: QdrantClient connected to {self.qdrant_url}:{qdrant_conn_port if qdrant_conn_port else 'default'}")
        else:
            # Fallback to local Qdrant instance
            self.qdrant_client = QdrantClient(host="localhost", port=6333)
            logger.info("QueryPipeline: QdrantClient connected to local Qdrant at localhost:6333 (Cloud URL not provided)")

        # Initialize embedding model - FORCE SentenceTransformer for now
        self.embeddings = None  # Ensure Langchain Google Embeddings are not used
        self._st_model = None
        logger.info(f"QueryPipeline: Forcing use of SentenceTransformer model: '{self.embedding_model_name}'")
        try:
            self._st_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"QueryPipeline: Successfully loaded SentenceTransformer '{self.embedding_model_name}'.")
        except ImportError:
            logger.error("SentenceTransformers library not found. Install it (`pip install sentence-transformers`) for embedding.")
            raise
        except Exception as e:
            logger.error(f"QueryPipeline: Failed to load SentenceTransformer model '{self.embedding_model_name}': {e}")
            raise
        
        # LangChain Vector Store (mainly for potential future use or other LangChain integrations)
        # We will primarily use direct Qdrant client search for retrieval with SentenceTransformer
        # Note: The ValueError about embeddings=None is avoided by not relying on it for ST retrieval
        try:
            self.vector_store = Qdrant(
                client=self.qdrant_client,
                collection_name=self.qdrant_collection,
                # Pass embeddings only if using a Langchain compatible one (like Google's)
                embeddings=self.embeddings if self.embeddings else None 
            )
            logger.info(f"Initialized LangChain Qdrant vector store wrapper for collection '{self.qdrant_collection}'.")
        except ValueError as e:
             logger.warning(f"Could not initialize LangChain Qdrant VectorStore (this is expected if using SentenceTransformer directly for retrieval): {e}")
             self.vector_store = None # Indicate it's not fully usable for retrieval in ST mode

        # Initialize LLM (Gemini)
        if not google_api_key:
            google_api_key = os.environ.get("GOOGLE_API_KEY")
        if not google_api_key:
             logger.warning("GOOGLE_API_KEY not set. LLM analysis will not function.")
             self.llm = None
             self.compliance_chain = None
        else:
            self.llm = ChatGoogleGenerativeAI(model=llm_model_name, temperature=0.1) # Add temperature
            logger.info(f"Initialized LLM with model: {llm_model_name}")

            # Define prompt template
            self.compliance_prompt = PromptTemplate(
                input_variables=["context", "document"],
                template="""You are an expert AI assistant specialized in RBI regulations and banking compliance.
                You will be given relevant excerpts from RBI regulatory documents ('Context') and a section of a bank's internal policy document ('Bank's Policy Section').
                Your task is to provide a clear, structured, and accurate compliance assessment.

                **Instructions:**
                1.  **Compare Policy to Context:** Carefully compare the Bank's Policy Section against the specific points in the RBI Regulations Context.
                2.  **Strictly Context-Based:** Base your analysis STRICTLY on the provided text. Do NOT use outside knowledge or make assumptions.
                3.  **Precise Classification Rules:** Apply the following classification rules for EACH policy aspect assessed:
                    - **Compliant:** The policy explicitly covers all requirements specified in the regulation in a manner that aligns with regulatory intent. A relevant regulation MUST be cited from the context.
                    - **Partially Compliant:** The policy addresses some aspects of the requirement but has gaps, is unclear, or lacks necessary detail. A relevant regulation MUST be cited from the context.
                    - **Non-Compliant:** The policy DIRECTLY CONTRADICTS a clear requirement in the regulation (e.g., policy says annual review when regulation requires quarterly). A relevant regulation MUST be cited from the context.
                    - **Unable to Assess:** Use this ONLY when:
                       a) The regulation requires something, but the policy doesn't mention it at all.
                       b) There is insufficient context (either policy or regulation) to determine compliance.
                       c) **CRITICAL:** No relevant regulation addressing the specific policy aspect was found in the provided Context ('No relevant regulation provided in context').
                4.  **Mandatory Citation:** For EVERY point you assess, you MUST cite the specific regulation text from the Context that supports your finding (e.g., citing the regulation number/clause if available, or quoting a relevant phrase) OR explicitly state 'No relevant regulation provided in context'.
                5.  **'Unable to Assess' Rule:** If your supporting regulation states 'No relevant regulation provided in context', then your finding MUST ALWAYS be 'Unable to Assess'.
                6.  **Compliance Percentage Calculation:**
                    - Calculate an overall compliance percentage for this section.
                    - Count the number of items assessed as Compliant (C), Partially Compliant (P), and Non-Compliant (N).
                    - Items marked 'Unable to Assess' (U) are EXCLUDED from the percentage calculation.
                    - Total Assessable Items = C + P + N.
                    - If Total Assessable Items is 0, the percentage is 0% and the verdict is 'Unable to Assess'.
                    - Otherwise, Percentage = ((C * 1) + (P * 0.5) + (N * 0)) / Total Assessable Items * 100.
                    - Round the percentage to the nearest whole number.
                7.  **Overall Verdict Determination:** Assign a verdict based on the calculated percentage AND the presence of high-severity issues:
                    - 'Fully Compliant': 100% and no High/Medium severity issues.
                    - 'Largely Compliant': 70-99% and no High severity issues.
                    - 'Minor Issues Found': 50-69% OR <70% with only Low severity issues.
                    - 'Significant Issues Found': < 50% OR any High severity issues present.
                    - 'Unable to Assess': If Total Assessable Items = 0.
                8.  **Severity Assignment:** Assign a severity level (High, Medium, Low) to each Non-Compliant or Partially Compliant item based on potential regulatory risk, financial impact, or operational disruption.
                9.  **Section Title Generation:** Determine the most appropriate title for this policy section:
                    - Identify the main subject matter.
                    - Title should be brief (2-6 words), descriptive, and in Title Case.
                    - Examples: "Customer Due Diligence Procedures", "Suspicious Transaction Monitoring", "Account Opening Requirements".
                    - **AVOID:** Starting with numbers/letters (e.g., "1. KYC"), copying policy text, generic titles ("Policy Section").
                10. **Response Structure:** Structure your response EXACTLY using the following Markdown format. Ensure all fields are populated correctly based on your analysis.

                **1. Compliance Summary:**
                - Section Title: [Generated Section Title]
                - Overall Compliance: [Calculated Percentage]% ([Determined Verdict])
                - Number of Policy Aspects Assessed: [Total C+P+N+U]
                - Compliant Items: [Count C] ([Percentage C]% of Assessable)
                - Partially Compliant Items: [Count P] ([Percentage P]% of Assessable)
                - Non-Compliant Items: [Count N] ([Percentage N]% of Assessable)
                - Items Unable to Assess: [Count U]

                **2. Prioritized Action Items:**
                [If no action items (No Non-Compliant or Partially Compliant items), state: "No action items required based on this assessment."]
                [Otherwise, list action items ordered by severity (High > Medium > Low):]
                1. [High Severity] [Policy Aspect]: [Brief, actionable recommendation to achieve compliance]
                2. [Medium Severity] [Policy Aspect]: [Brief, actionable recommendation to achieve compliance]
                3. [Low Severity] [Policy Aspect]: [Brief, actionable recommendation to achieve compliance]

                **3. Detailed Assessment:**
                [Evaluate key aspects of the Bank's Policy Section against the Context. List ALL assessed aspects.]
                - Policy Aspect: [e.g., Customer identification procedure]
                  - Finding: [Compliant | Partially Compliant | Non-Compliant | Unable to Assess]
                  - Severity: [High | Medium | Low] (Only if Partially or Non-Compliant)
                  - Explanation: [Brief justification for the finding, linking policy to regulation or lack thereof.]
                  - Supporting Regulation: [Cite specific text/reference from Context OR 'No relevant regulation provided in context']
                
                [Repeat for each assessed aspect...]

                **4. Referenced Regulations:**
                [List all unique regulations cited in the Detailed Assessment section. If none were cited (e.g., all 'Unable to Assess' due to no context), state: 'No specific regulations were found highly relevant to this section based on the current threshold.']
                - [Reference ID / Title of Regulation 1]
                - [Reference ID / Title of Regulation 2]
                ...

                **Context (RBI Regulations):**
                ```
                {context}
                ```

                **Bank's Policy Section:**
                ```
                {document}
                ```

                **Compliance Analysis:**
                """
            )
            
            # Create LLMChain
            self.compliance_chain = LLMChain(
                llm=self.llm,
                prompt=self.compliance_prompt,
                verbose=True
            )

        # Initialize document processor (placeholder - needs actual implementation)
        # This should handle reading PDF/DOCX/TXT and potentially chunking
        self.doc_processor = TextProcessor() # Using the combined processor for now
        logger.info("Initialized document processor.")

    def preprocess_document_for_analysis(self, document_path: str) -> List[str]:
        """
        Processes a document and returns a list of its text sections/chunks.
        This is the first step in the incremental analysis workflow.
        """
        logger.info(f"Step 1: Processing document into chunks: {document_path}")
        # The _process_user_document method already returns a list of chunk dictionaries.
        # We just need to extract the 'text' from each dictionary.
        chunk_dicts = self._process_user_document(document_path)
        
        # Group chunks by section title to represent the document's structure
        sections = defaultdict(list)
        for chunk in chunk_dicts:
            section_title = chunk.get('metadata', {}).get('section_title', 'Unnamed Section')
            sections[section_title].append(chunk['text'])
        
        # Combine the texts of chunks within the same section
        section_texts = ["\n".join(texts) for texts in sections.values()]
        
        logger.info(f"Preprocessed document into {len(section_texts)} sections.")
        return section_texts

    def _process_user_document(self, document_path: str) -> List[Dict]:
        """Processes the user's document into chunks, preserving section context.

        For PDFs, it uses the already chunked output from PDFProcessor.
        For other types (HTML, TXT, DOCX), it gets sections and chunks them here.

        Returns:
            List of chunk dictionaries, each containing at least:
            {'id': chunk_id, 'text': chunk_content, 'metadata': {..., 'section_title': title, ...}}
        """
        logger.info(f"Processing user document into section-aware chunks: {document_path}")
        file_ext = Path(document_path).suffix.lower()
        all_chunks = [] # Initialize chunk list

        try:
            # 1. Process based on file type
            if file_ext == ".pdf":
                # PDF processor ALREADY returns chunk dictionaries with id, text, metadata
                all_chunks = self.doc_processor.pdf_processor.process_document(document_path)
                if not all_chunks:
                     logger.warning(f"PDF document {document_path} yielded no chunks.")
                # No further splitting needed for PDF files here

            elif file_ext in [".html", ".htm", ".txt", ".docx"]: # Handle types that need splitting here
                sections = []
                # --- Get sections for these types ---
                if file_ext == ".html" or file_ext == ".htm":
                    # Assuming html_processor.process_document returns list of section dicts
                    # with keys like 'section_title', 'section_text', 'metadata'
                    sections = self.doc_processor.html_processor.process_document(document_path)
                elif file_ext == ".txt":
                    # Treat the whole document as one section
                    with open(document_path, "r", encoding="utf-8") as f:
                        full_text = f.read()
                    doc_name = Path(document_path).name
                    sections = [{
                        "section_title": doc_name,
                        "section_text": full_text,
                        "metadata": { # Basic metadata for TXT
                             "document_type": "txt",
                             "source_file": str(document_path),
                             "file_name": doc_name, # Ensure file_name is present
                             "reference_id": f'TXT_REF_{Path(document_path).stem}'
                        }
                    }]
                elif file_ext == ".docx":
                    # Treat the whole document as one section
                    try:
                        import docx
                        doc = docx.Document(document_path)
                        full_text = "\n".join([para.text for para in doc.paragraphs])
                        doc_name = Path(document_path).name
                        sections = [{
                            "section_title": doc_name,
                            "section_text": full_text,
                            "metadata": { # Basic metadata for DOCX
                                 "document_type": "docx",
                                 "source_file": str(document_path),
                                 "file_name": doc_name, # Ensure file_name is present
                                 "reference_id": f'DOCX_REF_{Path(document_path).stem}'
                            }
                        }]
                    except ImportError:
                        logger.error("python-docx not found. Install it (`pip install python-docx`) to process .docx files.")
                        raise
                    except Exception as e:
                        logger.error(f"Error processing DOCX file {document_path}: {e}")
                        raise
                # --- End getting sections ---

                if not sections:
                    logger.warning(f"Document {document_path} (type: {file_ext}) yielded no sections.")
                    return []

                # --- Splitting logic needed ONLY for non-PDF types that provide sections ---
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=TEXT_SPLITTER_CHUNK_SIZE,
                    chunk_overlap=TEXT_SPLITTER_CHUNK_OVERLAP,
                    length_function=len,
                    is_separator_regex=False,
                    separators=TEXT_SPLITTER_SEPARATORS
                )

                # Iterate through sections and chunk their text
                chunk_counter = 0
                for section in sections:
                    section_title = section.get("section_title", "Unnamed Section")
                    section_text = section.get("section_text", "")
                    # Start with metadata from the section itself
                    base_chunk_metadata = section.get("metadata", {}).copy()
                    base_chunk_metadata["section_title"] = section_title # Ensure section title is in metadata

                    if not section_text.strip():
                        continue # Skip empty sections

                    chunks_from_section = text_splitter.split_text(section_text)

                    # Create chunk dictionaries mimicking PDFProcessor output structure
                    for i, chunk_text in enumerate(chunks_from_section):
                        chunk_metadata = base_chunk_metadata.copy()
                        chunk_metadata["chunk_index_in_section"] = i

                        # Clean the text for the final chunk - ONLY for non-PDF types
                        cleaned_chunk_text = chunk_text # Default: no cleaning
                        if file_ext == ".pdf":
                            cleaned_chunk_text = chunk_text # PDFs are already cleaned
                        else:
                            cleaned_chunk_text = self.doc_processor.pdf_processor._clean_text(chunk_text) # Reuse cleaning

                        chunk_metadata["text"] = cleaned_chunk_text # Also store cleaned text in metadata like PDFProcessor does

                        # Generate a unique ID
                        ref_id = chunk_metadata.get('reference_id', f'UNKNOWN_{file_ext.upper()}')
                        chunk_id = f"{ref_id}_chunk_{chunk_counter}"
                        chunk_counter += 1

                        chunk_dict = {
                            "id": chunk_id,
                            "text": cleaned_chunk_text, # Text for embedding
                            "metadata": chunk_metadata
                        }
                        all_chunks.append(chunk_dict)
                # --- End splitting logic for non-PDF ---

            else:
                logger.error(f"Unsupported document format: {file_ext}")
                raise ValueError(f"Unsupported document format: {file_ext}")

            # Log final count using the unified all_chunks list
            logger.info(f"Processed document {document_path} into {len(all_chunks)} chunks.")
            return all_chunks

        except Exception as e:
             logger.error(f"Failed to process document {document_path}: {e}", exc_info=True)
             return [] # Return empty list on error

    def retrieve_relevant_regulations(self, query_text: str, top_k: int = 5, score_threshold: Optional[float] = None) -> List[Dict]:
        """Retrieve relevant regulations from Qdrant.
        
        Args:
            query_text: Text query (e.g., a chunk's content from the user's document)
            top_k: Number of relevant documents to retrieve
            score_threshold: Minimum similarity score to consider
        Returns:
            List of retrieved regulations (documents with metadata and score)
        """
        logger.info(f"Retrieving top {top_k} regulations for query: '{query_text[:100]}...'")
        
        results_with_scores = []

        # Use SentenceTransformer if it's initialized
        if self._st_model:
            logger.info("Using SentenceTransformer and direct Qdrant client for retrieval.")
            query_vector = self._st_model.encode(query_text).tolist()
            try:
                search_result = self.qdrant_client.search(
                    collection_name=self.qdrant_collection,
                    query_vector=query_vector,
                    query_filter=None, # Add filters if needed
                    limit=top_k,
                    score_threshold=score_threshold # Let Qdrant handle None threshold if needed
                )
                results_with_scores = [
                    # Payload is already flat due to ingestion fix
                    {"text": hit.payload.get("text", ""), "metadata": hit.payload, "score": hit.score} 
                    for hit in search_result
                ]
            except Exception as e:
                 logger.error(f"Qdrant search failed: {e}")
                 # Handle error appropriately, maybe return empty list or raise
                 return []

        # Use Langchain retriever if Google embeddings are available and ST model is not
        elif self.embeddings and self.vector_store:
            logger.info("Using LangChain VectorStore with Google Embeddings for retrieval.")
            try:
                # Use similarity_search_with_score for consistency
                results_with_scores_lc = self.vector_store.similarity_search_with_score(
                    query=query_text, k=top_k, score_threshold=score_threshold or 0.0 # Adjust threshold as needed for Google Embeddings
                )
                results_with_scores = [
                    {"text": doc.page_content, "metadata": doc.metadata, "score": score}
                    for doc, score in results_with_scores_lc
                ]
            except Exception as e:
                 logger.error(f"LangChain vector store search failed: {e}")
                 return []
        
        else:
             logger.error("No valid embedding/retrieval method configured.")
             return []

        logger.info(f"Retrieved {len(results_with_scores)} documents.")
        # Sort by score descending just in case Qdrant doesn't guarantee it (though it usually does)
        results_with_scores.sort(key=lambda x: x['score'], reverse=True)
        
        return results_with_scores

    def analyze_single_chunk(self, document_chunk: str, top_k: int = 5, score_threshold: Optional[float] = None) -> str:
        """
        Analyzes a single chunk of a document for compliance.
        This is the core iterative step in the incremental analysis workflow.
        
        Args:
            document_chunk: The text content of the document section to analyze.
            top_k: The number of relevant regulations to retrieve.
            score_threshold: The minimum similarity score for retrieved regulations.

        Returns:
            A string containing the raw analysis from the LLM.
        """
        logger.info(f"Analyzing compliance for chunk: '{document_chunk[:100]}...'")
        
        # 1. Retrieve relevant regulations for the chunk
        regulations = self.retrieve_relevant_regulations(
            query_text=document_chunk,
            top_k=top_k,
            score_threshold=score_threshold
        )
        
        # 2. Generate compliance analysis using the LLM
        analysis_text = self._generate_compliance_analysis(
            policy_section=document_chunk,
            regulations=regulations
        )
        
        logger.info("LLM analysis completed.")
        return analysis_text

    def _generate_compliance_analysis(self, policy_section: str, regulations: List[Dict]) -> str:
        """Invokes the LLM chain to generate a compliance analysis for a given
        policy section against a set of regulations.

        Args:
            policy_section: The text of the bank's policy section.
            regulations: A list of dictionaries, each representing a relevant regulation.

        Returns:
            A string containing the raw analysis from the LLM.
        """
        if not self.compliance_chain:
            logger.error("LLM chain is not initialized. Cannot perform analysis.")
            return "Error: LLM not available."

        # Format the regulations context
        if regulations:
            context_text = "\n\n".join([
                f"Regulation {i+1}: [Source: {reg.get('metadata', {}).get('source', 'N/A')}]\n{reg.get('text', '')}"
                for i, reg in enumerate(regulations)
            ])
        else:
            context_text = "No relevant regulations were found in the knowledge base."

        # Run the LLM chain
        try:
            response = self.compliance_chain.run(context=context_text, document=policy_section)
            return response
        except Exception as e:
            logger.error(f"Error running LLM chain: {e}", exc_info=True)
            return f"An error occurred during LLM analysis: {e}"

    def analyze_document(self, document_path: str, top_k: int = 5, score_threshold: Optional[float] = None, progress_callback: Optional[callable] = None) -> Dict:
        """
        DEPRECATED: This method performs analysis in a single, long-running batch.
        Use the new incremental methods: preprocess_document_for_analysis and analyze_single_chunk.
        """
        raise NotImplementedError(
            "analyze_document is deprecated. Use preprocess_document_for_analysis and "
            "analyze_single_chunk for an incremental workflow."
        )

    def format_analysis_report(self, analysis_results: Dict) -> str:
        """Formats the final analysis results into a human-readable string report.

        Args:
            analysis_results: A dictionary containing the analysis results.

        Returns:
            A string containing the formatted analysis report.
        """
        report_parts = []
        skipped_sections_count = 0
        total_sections = 0

        # Add header with parameters
        params = analysis_results.get('parameters', {})
        report_parts.append(f"# Compliance Analysis Report")
        report_parts.append(f"**Document:** `{analysis_results.get('document_path', 'N/A')}`")
        report_parts.append(f"**Analysis Parameters:** Top K={params.get('top_k', 'N/A')}, Threshold={params.get('score_threshold', 'N/A')}")
        report_parts.append("***")

        if "section_analyses" in analysis_results:
            total_sections = len(analysis_results["section_analyses"])
            for i, section_data in enumerate(analysis_results["section_analyses"]):
                section_title = section_data.get("section_title", f"Section {i+1}")
                analysis_text = section_data.get("analysis_text", "Analysis not available.") 
                retrieved_regs = section_data.get("retrieved_regulations", [])
                section_text_preview = section_data.get("section_text", "")[:300] # Preview first 300 chars

                # --- Check if section should be skipped --- 
                is_fully_compliant = "Overall Assessment:**\n- Fully Compliant" in analysis_text
                no_recommendations_needed = "Actionable Recommendations:**\n- No specific recommendations required" in analysis_text

                if is_fully_compliant and no_recommendations_needed:
                    skipped_sections_count += 1
                    continue # Skip adding this section to the report

                # --- Add Section Details (Markdown) --- 
                report_parts.append(f"## Section: {section_title}")
                
                # Add policy text preview in a code block
                report_parts.append("**Original Policy Text (Preview):**")
                report_parts.append(f"```\n{section_text_preview}{'...' if len(section_data.get('section_text', '')) > 300 else ''}\n```")

                # Add retrieved regulations as a list
                report_parts.append("**Retrieved Regulations:**")
                if retrieved_regs:
                    reg_markdown = "\n".join([f"- **[{reg['metadata'].get('source', 'Unknown Source')}, Page {reg['metadata'].get('page_number', 'N/A')}]**: ...{reg['text'][:150]}..." for reg in retrieved_regs])
                    report_parts.append(reg_markdown)
                else:
                    report_parts.append("- None")

                # Add LLM Analysis (already formatted as Markdown)
                report_parts.append("### Compliance Assessment")
                report_parts.append(analysis_text)
                report_parts.append("***") # Separator between sections

        if skipped_sections_count == total_sections and total_sections > 0:
            report_parts.append("**Overall Result:** All sections were found to be fully compliant with the retrieved regulations, and no specific recommendations were necessary based on the provided context.")
        elif skipped_sections_count > 0:
             report_parts.append(f"*Note: {skipped_sections_count} section(s) were omitted from this report as they were fully compliant with no recommendations.*")

        report_str = "\n\n".join(report_parts)
        report_str += "Please review the detailed section analysis for specific action items and findings."
        return report_str


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="RBI Compliance Analysis Pipeline")
    
    parser.add_argument(
        "--document_path",
        type=str,
        required=True,
        help="Path to the document to be analyzed (PDF, DOCX, TXT, HTML)"
    )
    # Add other arguments as needed, e.g., for Qdrant connection details
    parser.add_argument("--qdrant_url", type=str, help="Qdrant Cloud URL", default=None)
    parser.add_argument("--qdrant_api_key", type=str, help="Qdrant API Key", default=None)

    return parser.parse_args()


if __name__ == '__main__':
    # This main block is for testing the pipeline from the command line.
    # It demonstrates the NEW incremental workflow.
    
    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()

    args = parse_args()

    # Get API keys from environment
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    qdrant_url = args.qdrant_url or os.environ.get("QDRANT_URL")
    qdrant_api_key = args.qdrant_api_key or os.environ.get("QDRANT_API_KEY")

    if not google_api_key:
        logger.error("GOOGLE_API_KEY environment variable not set.")
        exit(1)
    if not qdrant_url:
        logger.warning("QDRANT_URL environment variable not set. Using local Qdrant.")
        
    logger.info("Initializing QueryPipeline for command-line execution...")
    pipeline = QueryPipeline(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        google_api_key=google_api_key
    )
    
    logger.info("--- Starting Incremental Analysis ---")
    
    # Step 1: Preprocess the document to get chunks
    try:
        document_chunks = pipeline.preprocess_document_for_analysis(args.document_path)
    except FileNotFoundError:
        logger.error(f"Error: Document not found at {args.document_path}")
        exit(1)
    except Exception as e:
        logger.error(f"An error occurred during preprocessing: {e}")
        exit(1)

    # Step 2: Analyze each chunk incrementally
    all_analyses = []
    logger.info(f"Analyzing {len(document_chunks)} sections...")
    for i, chunk in enumerate(tqdm(document_chunks, desc="Analyzing Sections")):
        analysis_text = pipeline.analyze_single_chunk(chunk, top_k=5, score_threshold=0.5)
        all_analyses.append({
            "section_index": i,
            "section_text": chunk, # Include original text
            "analysis_text": analysis_text
        })
        
    # Step 3: Consolidate and save results
    final_results = {
        "document_path": args.document_path,
        "parameters": {"top_k": 5, "score_threshold": 0.5},
        "section_analyses": all_analyses
    }

    # Save the results to a JSON file
    output_filename = f"analysis_results_{Path(args.document_path).stem}.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
        
    logger.info(f"--- Analysis Complete ---")
    logger.info(f"Results saved to {output_filename}")
