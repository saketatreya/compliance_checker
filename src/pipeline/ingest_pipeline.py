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
        embedding_model: str = "all-mpnet-base-v2",
        start_year: int = 2015,
        end_year: Optional[int] = None,
        doc_types: List[str] = ["circulars", "notifications"],
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        qdrant_collection: str = "rbi_regulations",
        vector_size: int = 768,
        batch_size: int = 32
    ):
        """Initialize the ingestion pipeline.
        
        Args:
            data_dir: Base directory for data storage
            embedding_model: Name of the embedding model to use
            start_year: First year to scrape documents from
            end_year: Last year to scrape documents from (defaults to current year)
            doc_types: Types of documents to process
            qdrant_host: Qdrant host address
            qdrant_port: Qdrant port
            qdrant_collection: Qdrant collection name
            vector_size: Vector size for Qdrant
            batch_size: Batch size for embedding processing
        """
        self.data_dir = Path(data_dir)
        self.embedding_model = embedding_model
        self.start_year = start_year
        self.end_year = end_year
        self.doc_types = doc_types
        
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
            output_dir=str(self.data_dir / "embeddings")
        )
        self.retriever = QdrantRetriever(
            url=qdrant_host,
            port=qdrant_port,
            collection_name=qdrant_collection,
            embedding_dim=vector_size,
            input_dir=str(self.data_dir / "embeddings")
        )
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        logger.info(f"Connected to Qdrant at {qdrant_host}:{qdrant_port}")
        
        self.vector_size = vector_size
        self.batch_size = batch_size
        
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
        
        self.embedder.process_all_documents(self.doc_types, self.batch_size)
        
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
        default="all-mpnet-base-v2",
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
        help="Qdrant host address"
    )
    
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=6333,
        help="Qdrant port"
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
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding processing"
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
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        qdrant_collection=args.qdrant_collection,
        vector_size=args.vector_size,
        batch_size=args.batch_size
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