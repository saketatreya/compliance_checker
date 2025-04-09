import os
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import tempfile
import json

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
        embedding_model_name: str = "all-mpnet-base-v2",
        llm_model_name: str = "gemini-pro",
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
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        logger.info(f"Connected to Qdrant at {qdrant_host}:{qdrant_port}")

        # Initialize embedding function using LangChain
        # For consistency, should use the same SentenceTransformer model used in ingestion.
        # Langchain might not directly support SentenceTransformer for retrieval this way.
        # Option 1: Use a compatible LangChain embedding (like Google's)
        if google_api_key:
             self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
             logger.info("Using GoogleGenerativeAIEmbeddings for retrieval.")
        else:
            # Option 2: Fallback or raise error if key is missing
            # This requires SentenceTransformer integration with LangChain's VectorStoreRetriever
            # which might need custom code or a different approach.
            # For now, we'll rely on Google's embedding for retrieval if API key is present.
             logger.warning("Google API Key not provided. Retrieval might not work as expected if embeddings in Qdrant are from a different model.")
             # As a placeholder, let's try to use the SentenceTransformer directly for query embedding
             # Note: This requires `sentence-transformers` to be installed
             try:
                 self._st_model = SentenceTransformer(embedding_model_name)
                 self.embeddings = None # Indicate we'll embed manually
                 logger.info(f"Using SentenceTransformer '{embedding_model_name}' directly for query embedding.")
             except ImportError:
                 logger.error("SentenceTransformers library not found. Install it (`pip install sentence-transformers`) or provide a Google API Key.")
                 raise
             except Exception as e:
                  logger.error(f"Failed to load SentenceTransformer model '{embedding_model_name}': {e}")
                  raise

        # Initialize Qdrant LangChain vector store
        self.vector_store = Qdrant(
            client=self.qdrant_client,
            collection_name=self.qdrant_collection,
            # embeddings=self.embeddings # Pass embeddings function if available and compatible
            # If using manual embedding: embeddings are not needed here, but used in retriever.
            embedding_function=self.embeddings # Try passing it, Langchain might handle it
        )
        logger.info(f"Initialized LangChain Qdrant vector store for collection '{self.qdrant_collection}'")
        
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
            # (Using the template similar to the one in GeminiInterface example)
            self.compliance_prompt = PromptTemplate(
                input_variables=["context", "document"],
                template="""You are an expert AI assistant specialized in RBI regulations and banking compliance. 
                You will be given excerpts from RBI regulatory documents and a section of a bank's internal policy. 
                Your job is to assess whether the policy section complies with the regulations, point out any compliance issues, 
                and cite the relevant RBI rules.

                When you refer to an RBI regulation, include the document reference or clause number from the provided context.
                If a relevant regulation is not provided in context, say you are not certain, rather than guessing.
                Base your analysis ONLY on the provided context and document section.
                
                Context:
                RBI Regulations:
                {context}
                
                Bank's Policy Section:
                {document}
                
                Analysis:
                Identify any compliance issues in the bank's policy section with respect to RBI regulations. 
                Cite the relevant RBI guidelines in your answer. If no issues are found, confirm compliance and cite the relevant guidelines.
                
                Format your response as follows:
                1. Summary: A brief overview of your compliance assessment (Compliant / Non-Compliant / Potential Issues).
                2. Issues: List each compliance issue with citation (if any). If none, state "No issues found".
                3. Recommendations: Suggested changes to ensure compliance (if applicable).
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
        """Process the user's document into text chunks.
        For now, treats the whole document as one chunk after text extraction.
        """
        logger.info(f"Processing user document: {document_path}")
        file_ext = Path(document_path).suffix.lower()
        chunks = []

        if file_ext == ".pdf":
            # Use PDFProcessor logic (assuming it returns text per page or similar)
            pdf_chunks = self.doc_processor.pdf_processor.process_document(document_path)
            # Consolidate PDF chunks into logical sections if possible, or just join text
            # Simple approach: join all text for now
            full_text = " ".join([chunk['text'] for chunk in pdf_chunks])
            chunks.append(full_text)
        elif file_ext == ".html" or file_ext == ".htm":
            html_chunks = self.doc_processor.html_processor.process_document(document_path)
            full_text = " ".join([chunk['text'] for chunk in html_chunks])
            chunks.append(full_text)
        elif file_ext == ".txt":
            with open(document_path, "r", encoding="utf-8") as f:
                chunks.append(f.read())
        elif file_ext == ".docx":
            try:
                import docx
                doc = docx.Document(document_path)
                full_text = "\n".join([para.text for para in doc.paragraphs])
                chunks.append(full_text)
            except ImportError:
                logger.error("python-docx library not found. Install it (`pip install python-docx`) to process .docx files.")
                raise
            except Exception as e:
                logger.error(f"Error processing DOCX file {document_path}: {e}")
                raise
        else:
            logger.error(f"Unsupported document format: {file_ext}")
            raise ValueError(f"Unsupported document format: {file_ext}")

        # Basic chunking (e.g., by paragraphs or fixed size) can be added here if needed
        # For now, return the full text as a single chunk
        # Placeholder: simple paragraph splitting
        final_chunks = []
        for text in chunks:
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()] # Split by double newline
            final_chunks.extend(paragraphs)
            
        logger.info(f"Processed document into {len(final_chunks)} chunks.")
        return final_chunks if final_chunks else chunks # Return paragraphs if split, else original

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
        if self.embeddings:
             # Use LangChain retriever with built-in embedding function
             retriever = self.vector_store.as_retriever(
                 search_type="similarity_score_threshold",
                 search_kwargs={'k': top_k, 'score_threshold': score_threshold or 0.7} # Default threshold 0.7
             )
             docs = retriever.get_relevant_documents(query)
             # Note: Langchain retriever might not return scores directly in this mode easily.
             # We might need to use similarity_search_with_score instead.
             results_with_scores_lc = self.vector_store.similarity_search_with_score(
                  query=query, k=top_k, score_threshold=score_threshold or 0.7
             )
             results_with_scores = [
                 {"text": doc.page_content, "metadata": doc.metadata, "score": score}
                 for doc, score in results_with_scores_lc
             ]

        elif hasattr(self, '_st_model'):
            # Manually embed the query and search Qdrant directly
            query_vector = self._st_model.encode(query).tolist()
            search_result = self.qdrant_client.search(
                collection_name=self.qdrant_collection,
                query_vector=query_vector,
                query_filter=None, # Add filters if needed
                limit=top_k,
                score_threshold=score_threshold or 0.7 # Default threshold 0.7
            )
            results_with_scores = [
                {"text": hit.payload.get("text", ""), "metadata": hit.payload, "score": hit.score}
                for hit in search_result
            ]
        else:
             logger.error("No valid embedding method available for retrieval.")
             return []
             
        logger.info(f"Retrieved {len(results_with_scores)} regulations.")
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
        default="all-mpnet-base-v2",
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