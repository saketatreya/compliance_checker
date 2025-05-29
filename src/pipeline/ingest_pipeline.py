import os
import logging
import argparse
from pathlib import Path
from typing import List, Optional

from src.scraper.rbi_scraper import RBIScraper
from src.preprocessor.text_processor import TextProcessor
from src.embedding.embedder import DocumentEmbedder
from src.retrieval.qdrant_retriever import QdrantRetriever

from qdrant_client import QdrantClient, models
from tqdm import tqdm
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IngestionPipeline:
    """End-to-end pipeline for ingesting RBI regulatory documents.
    
    This class orchestrates the entire process of scraping, preprocessing,
    embedding, and storing RBI documents in the vector database.
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        embedding_model: str = "msmarco-distilbert-base-tas-b",
        start_year: int = 2015,
        end_year: Optional[int] = None,
        doc_types: List[str] = ["circulars", "notifications"],
        qdrant_url_param: Optional[str] = None,      # Renamed to avoid clash with argparse
        qdrant_api_key_param: Optional[str] = None, # Renamed to avoid clash with argparse
        qdrant_host_arg: str = "localhost",       # From argparse, for fallback
        qdrant_port_arg: int = 6333,            # From argparse, for fallback
        qdrant_collection: str = "rbi_regulations",
        vector_size: int = 768,
        embedding_batch_size: int = 32, # Renamed from batch_size for clarity
        qdrant_batch_size: int = 100    # Added for Qdrant upsert batching
    ):
        """Initialize the ingestion pipeline.
        
        Args:
            data_dir: Base directory for data storage
            embedding_model: Name of the embedding model to use
            start_year: First year to scrape documents from
            end_year: Last year to scrape documents from (defaults to current year)
            doc_types: Types of documents to process
            qdrant_url_param: Qdrant Cloud URL (direct parameter)
            qdrant_api_key_param: Qdrant API Key (direct parameter)
            qdrant_host_arg: Qdrant host address (from argparse, for fallback)
            qdrant_port_arg: Qdrant port (from argparse, for fallback)
            qdrant_collection: Qdrant collection name
            vector_size: Vector size for Qdrant
            embedding_batch_size: Batch size for embedding processing
            qdrant_batch_size: Batch size for Qdrant client upserts
        """
        self.data_dir = Path(data_dir)
        self.embedding_model = embedding_model
        self.start_year = start_year
        self.end_year = end_year
        self.doc_types = doc_types
        
        # Determine Qdrant connection details (Priority: params -> env vars -> argparse defaults)
        qdrant_url = qdrant_url_param or os.environ.get("QDRANT_URL")
        qdrant_api_key = qdrant_api_key_param or os.environ.get("QDRANT_API_KEY")

        final_qdrant_url = None
        final_qdrant_port = None # Port is usually part of the URL for cloud

        if qdrant_url:
            final_qdrant_url = qdrant_url
            logger.info(f"IngestionPipeline: Attempting to use Qdrant URL (from param/env): {final_qdrant_url}")
            if qdrant_api_key:
                logger.info("IngestionPipeline: Qdrant API Key IS provided (from param/env).")
            else:
                logger.warning("IngestionPipeline: Qdrant URL is set, but API Key is NOT provided. This might fail for secured cloud instances.")
        else:
            final_qdrant_url = qdrant_host_arg
            final_qdrant_port = qdrant_port_arg
            logger.info(f"IngestionPipeline: Attempting to use Qdrant host/port from arguments/defaults: {final_qdrant_url}:{final_qdrant_port}")
            
        # Create data directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.data_dir / "raw", exist_ok=True)
        os.makedirs(self.data_dir / "processed", exist_ok=True)
        os.makedirs(self.data_dir / "embeddings", exist_ok=True)
        
        # Initialize components
        self.scraper = RBIScraper(output_dir=str(self.data_dir / "raw"))
        self.processor = TextProcessor(
            input_dir=str(self.data_dir / "raw"),
            output_dir=str(self.data_dir / "processed")
        )
        self.embedder = DocumentEmbedder(
            model_name=embedding_model,
            input_dir=str(self.data_dir / "processed"),
            output_dir=str(self.data_dir / "embeddings"),
            batch_size=embedding_batch_size # Use renamed embedding_batch_size
        )
        self.retriever = QdrantRetriever(
            url=final_qdrant_url,
            port=final_qdrant_port if not qdrant_url else None, # Port only if not using cloud URL
            api_key=qdrant_api_key if qdrant_url else None,      # API key only if using cloud URL
            collection_name=qdrant_collection,
            embedding_dim=vector_size,
            input_dir=str(self.data_dir / "embeddings"),
            qdrant_batch_size=qdrant_batch_size # Pass qdrant_batch_size
        )
        
        # Initialize Qdrant client (or use the retriever's client)
        # For simplicity, we'll re-initialize, but ideally, you'd use self.retriever.client
        if qdrant_url: # Cloud or specific URL
            ingest_qdrant_port = None
            if "cloud.qdrant.io" in qdrant_url and qdrant_url.startswith("https://"):
                ingest_qdrant_port = 6333 # Try Qdrant Cloud REST API over HTTPS on port 6333
                logger.info(f"IngestionPipeline QdrantClient: Detected Qdrant Cloud URL (HTTPS). Setting port to {ingest_qdrant_port} for REST-like API.")
            elif final_qdrant_port: # if a port was determined from args (e.g. for local non-default)
                ingest_qdrant_port = final_qdrant_port

            self.qdrant_client = QdrantClient(url=final_qdrant_url, port=ingest_qdrant_port, api_key=qdrant_api_key, prefer_grpc=False)
            logger.info(f"IngestionPipeline QdrantClient: Successfully initialized for {final_qdrant_url}:{ingest_qdrant_port if ingest_qdrant_port else '(default)'} with prefer_grpc=False")
        else: # Local default
             self.qdrant_client = QdrantClient(host=final_qdrant_url, port=final_qdrant_port)
             logger.info(f"IngestionPipeline QdrantClient: Successfully initialized for LOCAL at {final_qdrant_url}:{final_qdrant_port}")

        self.vector_size = vector_size
        # It seems 'batch_size' in __init__ was intended for embedding.
        # If Qdrant client also uses a batch_size, it's managed internally or in QdrantRetriever.
        # self.qdrant_batch_size = batch_size # If needed for Qdrant client directly
        
        logger.info(f"Initialized ingestion pipeline with model: {embedding_model}")
    
    def _setup_qdrant_collection(self):
        """Create or recreate the Qdrant collection."""
        logger.info(f"Ensuring Qdrant collection '{self.retriever.collection_name}' exists.")
        self.retriever.create_collection(recreate=True)
        logger.info(f"Qdrant collection setup completed.")

    def run_scraping(self):
        """Run the scraping step of the pipeline."""
        logger.info("Starting document scraping")
        
        # Scrape all documents
        results = self.scraper.scrape_all_documents(
            start_year=self.start_year,
            end_year=self.end_year
        )
        
        # Log results
        for doc_type, docs in results.items():
            logger.info(f"Scraped {len(docs)} {doc_type}")
        
        return results
    
    def run_preprocessing(self):
        """Run the preprocessing step of the pipeline."""
        logger.info("Starting document preprocessing")
        
        all_chunks = self.processor.process_all(self.doc_types)
        logger.info(f"Processed {len(all_chunks)} total chunks across specified types")
        
        return all_chunks
    
    def run_embedding(self):
        """Run the embedding step of the pipeline."""
        logger.info("Starting document embedding")
        
        # process_all_documents in DocumentEmbedder now uses its own configured batch_size
        self.embedder.process_all_documents(self.doc_types) 
        
        return True
    
    def run_indexing(self):
        """Run the indexing step of the pipeline."""
        logger.info("Starting document indexing in Qdrant")
        
        # Ensure collection is created before indexing
        self._setup_qdrant_collection()
        
        self.retriever.ingest_documents(self.doc_types)
        
        # Get collection info
        collection_info = self.retriever.get_collection_info()
        logger.info(f"Indexed documents in Qdrant: {collection_info}")
        
        return collection_info
    
    def run_pipeline(self, steps: List[str] = ["scrape", "preprocess", "embed", "index"]):
        """Run the complete ingestion pipeline or specific steps.
        
        Args:
            steps: List of pipeline steps to run
        
        Returns:
            Dictionary with results of each step
        """
        results = {}
        
        if "scrape" in steps:
            results["scrape"] = self.run_scraping()
        
        if "preprocess" in steps:
            results["preprocess"] = self.run_preprocessing()
        
        if "embed" in steps:
            results["embed"] = self.run_embedding()
        
        if "index" in steps:
            results["index"] = self.run_indexing()
        
        logger.info("Ingestion pipeline completed successfully")
        return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RBI Document Ingestion Pipeline")
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Base directory for data storage"
    )
    
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="msmarco-distilbert-base-tas-b",
        help="Name of the embedding model to use"
    )
    
    parser.add_argument(
        "--start-year",
        type=int,
        default=2015,
        help="First year to scrape documents from"
    )
    
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Last year to scrape documents from (defaults to current year)"
    )
    
    parser.add_argument(
        "--doc-types",
        nargs="+",
        default=["circulars", "notifications"],
        help="Types of documents to process"
    )
    
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["scrape", "preprocess", "embed", "index"],
        choices=["scrape", "preprocess", "embed", "index"],
        help="Pipeline steps to run"
    )
    
    parser.add_argument(
        "--qdrant-host",
        type=str,
        default="localhost",
        help="Qdrant host address (fallback if QDRANT_URL env var or --qdrant-url not set)"
    )
    
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=6333,
        help="Qdrant port (fallback if QDRANT_URL env var or --qdrant-url not set)"
    )
    
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default=os.environ.get("QDRANT_URL"),
        help="Qdrant Cloud URL (overrides host/port, also checks QDRANT_URL env var)"
    )

    parser.add_argument(
        "--qdrant-api-key",
        type=str,
        default=os.environ.get("QDRANT_API_KEY"),
        help="Qdrant API Key (used with --qdrant-url, also checks QDRANT_API_KEY env var)"
    )

    parser.add_argument(
        "--qdrant-collection",
        type=str,
        default="rbi_regulations",
        help="Qdrant collection name"
    )
    
    parser.add_argument(
        "--vector-size",
        type=int,
        default=768,
        help="Vector size for Qdrant"
    )
    
    parser.add_argument(
        "--embedding-batch-size", # Renamed argument
        type=int,
        default=32,
        help="Batch size for embedding processing"
    )
    
    parser.add_argument(
        "--qdrant-batch-size",
        type=int,
        default=100,
        help="Batch size for Qdrant upsert operations"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Initialize and run pipeline
    pipeline = IngestionPipeline(
        data_dir=args.data_dir,
        embedding_model=args.embedding_model,
        start_year=args.start_year,
        end_year=args.end_year,
        doc_types=args.doc_types,
        qdrant_url_param=args.qdrant_url,
        qdrant_api_key_param=args.qdrant_api_key,
        qdrant_host_arg=args.qdrant_host,
        qdrant_port_arg=args.qdrant_port,
        qdrant_collection=args.qdrant_collection,
        vector_size=args.vector_size,
        embedding_batch_size=args.embedding_batch_size, # Use renamed arg
        qdrant_batch_size=args.qdrant_batch_size      # Use new arg
    )
    
    # Run pipeline with specified steps
    results = pipeline.run_pipeline(steps=args.steps)
    
    # Print summary
    print("\nPipeline Execution Summary:")
    for step, result in results.items():
        if step == "index" and isinstance(result, dict):
            print(f"  {step.capitalize()}: {result.get('points_count', 0)} points indexed in Qdrant")
        elif step == "preprocess" and isinstance(result, list):
            print(f"  {step.capitalize()}: {len(result)} chunks processed")
        elif step == "scrape" and isinstance(result, dict):
            total_docs = sum(len(docs) for docs in result.values())
            print(f"  {step.capitalize()}: {total_docs} documents scraped")
        else:
            print(f"  {step.capitalize()}: Completed successfully")