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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QueryPipeline:
    """End-to-end pipeline for analyzing document compliance.
    
    This class orchestrates the process of analyzing user documents for compliance
    by retrieving relevant regulations and generating insights using LLM.
    """
    
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        qdrant_collection: str = "rbi_regulations",
        embedding_model_name: str = "msmarco-distilbert-base-tas-b",
        llm_model_name: str = "models/gemini-1.5-flash",
        google_api_key: Optional[str] = None,
    ):
        """Initialize the query pipeline.
        
        Args:
            qdrant_host: Qdrant host address
            qdrant_port: Qdrant port
            qdrant_collection: Qdrant collection name
            embedding_model_name: Name of the embedding model to use
            llm_model_name: Name of the LLM model to use
            google_api_key: Google API key for Gemini (defaults to GOOGLE_API_KEY env var)
        """
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.qdrant_collection = qdrant_collection
        
        # Initialize Qdrant client (used for direct search)
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        logger.info(f"Connected to Qdrant at {qdrant_host}:{qdrant_port}")

        # Initialize embedding model
        self.embedding_model_name = embedding_model_name
        self.embeddings = None
        self._st_model = None

        # Try Google Embeddings first if API key is available
        if google_api_key:
             try:
                 self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
                 logger.info("Using GoogleGenerativeAIEmbeddings.")
             except Exception as e:
                  logger.warning(f"Failed to initialize GoogleGenerativeAIEmbeddings: {e}. Falling back to SentenceTransformer.")
                  self.embeddings = None # Ensure fallback

        # Fallback or default to SentenceTransformer
        if not self.embeddings:
            try:
                self._st_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Using SentenceTransformer '{self.embedding_model_name}' directly.")
            except ImportError:
                logger.error("SentenceTransformers library not found. Install it (`pip install sentence-transformers`) for embedding.")
                raise
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer model '{self.embedding_model_name}': {e}")
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
            self.llm = ChatGoogleGenerativeAI(model=llm_model_name, google_api_key=google_api_key)
            logger.info(f"Initialized LLM: {llm_model_name}")

            # Define prompt template
            self.compliance_prompt = PromptTemplate(
                input_variables=["context", "document"],
                template="""You are an expert AI assistant specialized in RBI regulations and banking compliance.
                You will be given excerpts from RBI regulatory documents ('Context') and a section of a bank's internal policy ('Bank's Policy Section').
                Your task is to assess whether the Bank's Policy Section complies with the provided RBI Regulations context.

                **Instructions:**
                1.  Carefully analyze the Bank's Policy Section against EACH relevant point in the RBI Regulations context.
                2.  Base your analysis STRICTLY on the provided text. Do NOT assume prior knowledge or regulations not present in the Context.
                3.  Identify any specific areas of non-compliance, potential compliance issues, or confirm compliance.
                4.  For each point of analysis (compliance confirmed, issue identified), CLEARLY CITE the relevant regulation from the Context using the provided source identifier (e.g., [Source: Doc 1], [Source: Doc 2, Clause 5]).
                5.  If the provided Context does not contain information to assess a specific part of the policy, explicitly state that you cannot determine compliance for that part based on the given context.
                6.  Structure your response EXACTLY as follows:

                **1. Summary:** A brief one-sentence overview (e.g., Compliant, Non-Compliant, Potential Issues Found).
                **2. Compliance Issues:** List each identified issue OR state 'No compliance issues identified based on the provided context.'
                   - Issue 1: [Description of issue] (Relevant Regulation: [Citation from Context]).
                   - Issue 2: [Description of issue] (Relevant Regulation: [Citation from Context]).
                **3. Compliance Confirmation:** List areas where the policy IS compliant with the context OR state 'No specific compliant areas noted' if focusing only on issues.
                   - Point 1: [Description of compliant aspect] (Relevant Regulation: [Citation from Context]).
                **4. Areas of Uncertainty:** List parts of the policy that cannot be assessed due to lack of relevant information in the Context OR state 'All parts of the policy section could be assessed against the provided context.'
                   - Uncertainty 1: [Description of policy part] (Reason: No relevant regulation provided in context).
                **5. Recommendations:** (Optional) Suggest specific changes ONLY if non-compliance was found.

                **Context:**
                RBI Regulations:
                {context}

                **Bank's Policy Section:**
                {document}

                **Analysis:**
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

    def _process_user_document(self, document_path: str) -> List[str]:
        """Process the user's document into text chunks using RecursiveCharacterTextSplitter."""
        logger.info(f"Processing user document: {document_path}")
        file_ext = Path(document_path).suffix.lower()
        full_text = ""

        try:
            if file_ext == ".pdf":
                pdf_chunks = self.doc_processor.pdf_processor.process_document(document_path)
                full_text = "\n\n".join([chunk['text'] for chunk in pdf_chunks]) # Join pages/chunks first
            elif file_ext == ".html" or file_ext == ".htm":
                html_chunks = self.doc_processor.html_processor.process_document(document_path)
                full_text = " ".join([chunk['text'] for chunk in html_chunks])
            elif file_ext == ".txt":
                with open(document_path, "r", encoding="utf-8") as f:
                    full_text = f.read()
            elif file_ext == ".docx":
                try:
                    import docx
                    doc = docx.Document(document_path)
                    full_text = "\n".join([para.text for para in doc.paragraphs])
                except ImportError:
                    logger.error("python-docx library not found. Install it (`pip install python-docx`) to process .docx files.")
                    raise
                except Exception as e:
                    logger.error(f"Error processing DOCX file {document_path}: {e}")
                    raise
            else:
                logger.error(f"Unsupported document format: {file_ext}")
                raise ValueError(f"Unsupported document format: {file_ext}")

            if not full_text.strip():
                logger.warning(f"Document {document_path} appears to be empty after text extraction.")
                return []

            # Use RecursiveCharacterTextSplitter for chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Target size for chunks
                chunk_overlap=200, # Overlap between chunks
                length_function=len,
                is_separator_regex=False,
                separators=["\n\n", "\n", ". ", " ", ""] # Common separators
            )
            
            final_chunks = text_splitter.split_text(full_text)
            logger.info(f"Processed document into {len(final_chunks)} chunks using RecursiveCharacterTextSplitter.")
            return final_chunks

        except Exception as e:
             logger.error(f"Failed to process document {document_path}: {e}")
             return [] # Return empty list on error

    def retrieve_relevant_regulations(self, query: str, top_k: int = 5, score_threshold: Optional[float] = None) -> List[Dict]:
        """Retrieve relevant regulations from Qdrant.
        
        Args:
            query: Text query (e.g., a chunk from the user's document)
            top_k: Number of relevant documents to retrieve
            score_threshold: Minimum similarity score to consider
        Returns:
            List of retrieved regulations (documents with metadata and score)
        """
        logger.info(f"Retrieving top {top_k} regulations for query: '{query[:100]}...'")
        
        results_with_scores = []

        # Use SentenceTransformer if it's initialized
        if self._st_model:
            logger.info("Using SentenceTransformer and direct Qdrant client for retrieval.")
            query_vector = self._st_model.encode(query).tolist()
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
                    query=query, k=top_k, score_threshold=score_threshold or 0.0 # Adjust threshold as needed for Google Embeddings
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

    def analyze_compliance(self, document_chunk: str, regulations: List[Dict]) -> str:
        """Analyze compliance using LLM.
        
        Args:
            document_chunk: Text chunk from the user's document
            regulations: List of relevant regulations retrieved from Qdrant
            
        Returns:
            Compliance analysis text generated by LLM
        """
        if not self.compliance_chain:
            return "LLM analysis is not available (API key missing or LLM initialization failed)."
            
        logger.info(f"Analyzing compliance for chunk: '{document_chunk[:100]}...'")
        
        # Format context for the prompt
        context_str = ""
        for i, reg in enumerate(regulations):
            metadata = reg.get("metadata", {})
            ref = metadata.get("reference_id", f"Doc {i+1}")
            clause = metadata.get("clause_id", None)
            page = metadata.get("page", None)
            citation = f"[Source: {ref}" + (f", Clause {clause}" if clause else "") + (f", Page {page}" if page else "") + "]"
            context_str += f"Regulation {i+1}: {citation}\n{reg['text']}\n\n"
        
        if not context_str:
            context_str = "No relevant regulations found in the knowledge base."
            
        try:
            # Run the LLM chain
            response = self.compliance_chain.run(context=context_str, document=document_chunk)
            logger.info("LLM analysis completed.")
            return response
        except Exception as e:
            logger.error(f"Error during LLM analysis: {e}")
            return f"Error during analysis: {e}"

    def analyze_document(self, document_path: str, top_k: int = 5, score_threshold: Optional[float] = None) -> Dict:
        """Run the full query pipeline for a given document.
        
        Args:
            document_path: Path to the user document
            top_k: Number of relevant regulations to retrieve per chunk
            score_threshold: Minimum similarity score
            
        Returns:
            Dictionary containing analysis results for each chunk and overall info
        """
        logger.info(f"Starting analysis for document: {document_path}")
        
        try:
            # Process document into chunks
            doc_chunks = self._process_user_document(document_path)
            
            if not doc_chunks:
                return {"error": "Failed to process document or document is empty."}
            
            chunk_analyses = []
            all_retrieved_regulations = []
            
            # Analyze each chunk
            for i, chunk in enumerate(tqdm(doc_chunks, desc="Analyzing document chunks")):
                # Retrieve relevant regulations for the chunk
                regulations = self.retrieve_relevant_regulations(
                    query=chunk,
                    top_k=top_k,
                    score_threshold=score_threshold
                )
                all_retrieved_regulations.extend(regulations) # Collect all for summary
            
            # Analyze compliance
                analysis_text = self.analyze_compliance(chunk, regulations)
            
            chunk_analyses.append({
                "chunk_index": i,
                "chunk_text": chunk,
                "regulations": regulations,
                    "analysis": analysis_text
                })
            
            # Consolidate results
            # Find unique regulations cited across all chunks
            unique_regs = {json.dumps(reg['metadata']): reg for reg in all_retrieved_regulations}
            
            results = {
            "document_path": document_path,
                "num_chunks": len(doc_chunks),
            "chunk_analyses": chunk_analyses,
                "all_regulations": list(unique_regs.values()) # List of unique regulations used
            }
            
            logger.info(f"Analysis complete for {document_path}")
            return results
            
        except Exception as e:
            logger.error(f"Error during query pipeline execution for {document_path}: {e}")
            return {"error": str(e)}
    
    def format_analysis_report(self, analysis_result: Dict[str, Any]) -> str:
        """Format analysis result into a readable report.
        
        Args:
            analysis_result: Result from analyze_document
            
        Returns:
            Formatted report as text
        """
        document_path = analysis_result["document_path"]
        num_chunks = analysis_result["num_chunks"]
        all_regulations = analysis_result["all_regulations"]
        chunk_analyses = analysis_result["chunk_analyses"]
        
        # Build report
        report = []
        report.append(f"# Compliance Analysis Report")
        report.append(f"## Document: {Path(document_path).name}")
        report.append(f"")
        
        # Summary of regulations referenced
        report.append(f"## Referenced RBI Regulations")
        for i, reg in enumerate(all_regulations, 1):
            metadata = reg.get("metadata", {})
            ref_id = metadata.get("reference_id", "Unknown Reference")
            report.append(f"{i}. {ref_id}")
        report.append(f"")
        
        # Detailed analysis by chunk
        report.append(f"## Detailed Analysis")
        
        for chunk_analysis in chunk_analyses:
            chunk_index = chunk_analysis["chunk_index"]
            chunk_text = chunk_analysis["chunk_text"]
            analysis = chunk_analysis["analysis"]
            
            report.append(f"### Section {chunk_index + 1}")
            report.append(f"")
            report.append(f"**Text Excerpt:**")
            report.append(f"```")
            report.append(chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text)
            report.append(f"```")
            report.append(f"")
            report.append(f"**Analysis:**")
            report.append(analysis)
            report.append(f"")
        
        return "\n".join(report)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RBI Compliance Analysis Pipeline")
    
    parser.add_argument(
        "document",
        type=str,
        help="Path to the document file to analyze"
    )
    
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="msmarco-distilbert-base-tas-b",
        help="Name of the embedding model to use"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of relevant regulations to retrieve per chunk"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the analysis report (defaults to stdout)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    # Example Usage (requires GOOGLE_API_KEY)
    # Ensure Qdrant is running (e.g., via docker-compose up)
    
    # Make sure GOOGLE_API_KEY is set as an environment variable
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set.")
    else:
        pipeline = QueryPipeline()
        
        # Create a dummy test document file
        dummy_doc_path = "dummy_policy.txt"
        with open(dummy_doc_path, "w") as f:
            f.write("Our bank policy allows unsecured loans up to 15% of total assets.\n")
            f.write("Loans to directors are permitted if fully secured by government bonds.")
        
        # Analyze the dummy document
        analysis_results = pipeline.analyze_document(dummy_doc_path)
        
        # Print results (simplified)
        if "error" in analysis_results:
            print(f"Error: {analysis_results['error']}")
        else:
            print(f"Analysis Results for: {analysis_results['document_path']}")
            for chunk_result in analysis_results["chunk_analyses"]:
                print(f"\n--- Chunk {chunk_result['chunk_index'] + 1} ---")
                print(f"Text: {chunk_result['chunk_text'][:100]}...")
                print("Analysis:")
                print(chunk_result["analysis"])
                print(f"Retrieved {len(chunk_result['regulations'])} regulations.")
                
        # Clean up dummy file
        # os.remove(dummy_doc_path)