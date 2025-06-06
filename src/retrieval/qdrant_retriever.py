import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QdrantRetriever:
    """Vector storage and retrieval using Qdrant.
    
    This class handles the storage of document embeddings in Qdrant
    and provides functionality for semantic search.
    """
    
    def __init__(
        self, 
        collection_name: str = "rbi_regulations",
        embedding_dim: int = 768,
        url: Optional[str] = None,
        port: Optional[int] = None, # Qdrant Cloud URL usually includes the port or uses a default
        api_key: Optional[str] = None, # Added for Qdrant Cloud
        input_dir: str = "data/embeddings",
        qdrant_batch_size: int = 100 # Added qdrant_batch_size
    ):
        """Initialize the Qdrant retriever.
        
        Args:
            collection_name: Name of the Qdrant collection
            embedding_dim: Dimension of the embedding vectors
            url: URL of the Qdrant server (None for local). For Qdrant Cloud, include http/https.
            port: Port of the Qdrant server (often not needed if url includes it or for cloud default).
            api_key: API key for Qdrant Cloud.
            input_dir: Directory containing embedded chunks (for local ingestion to cloud)
            qdrant_batch_size: Batch size for upserting points to Qdrant
        """
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.input_dir = Path(input_dir)
        self.qdrant_batch_size = qdrant_batch_size # Store batch size
        
        # Initialize Qdrant client
        if url:
            _port = port # Use the provided port if any
            if "cloud.qdrant.io" in url: # It's a Qdrant Cloud URL
                if url.startswith("https://"):
                    _port = 6333 # Try Qdrant Cloud REST API over HTTPS on port 6333
                    logger.info(f"QdrantRetriever: Detected Qdrant Cloud URL (HTTPS). Forcing port to {_port} for REST-like API.")
                elif url.startswith("http://"): # Though unlikely for cloud
                    _port = 6333 
                    logger.info(f"QdrantRetriever: Detected Qdrant Cloud URL (HTTP). Forcing port to {_port}.")

            if _port:
                 self.client = QdrantClient(url=url, port=_port, api_key=api_key, prefer_grpc=False)
                 logger.info(f"QdrantRetriever: Initialized client for {url}:{_port} with prefer_grpc=False")
            else: # URL likely includes port or is a cloud URL (api_key might be used) - fallback if not cloud
                 self.client = QdrantClient(url=url, api_key=api_key, prefer_grpc=False)
                 logger.info(f"QdrantRetriever: Initialized client for {url} (default port) with prefer_grpc=False.")
        else:
            # Fallback to local Qdrant instance with persistence
            default_qdrant_path = "data/qdrant"
            os.makedirs(default_qdrant_path, exist_ok=True)
            self.client = QdrantClient(path=default_qdrant_path)
            logger.info(f"QdrantRetriever: Initialized local client at path ./{default_qdrant_path}")
    
    def create_collection(self, recreate: bool = False):
        """Create a collection in Qdrant.
        
        Args:
            recreate: Whether to recreate the collection if it exists
        """
        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name in collection_names:
            if recreate:
                logger.info(f"Recreating collection: {self.collection_name}")
                self.client.delete_collection(collection_name=self.collection_name)
            else:
                logger.info(f"Collection already exists: {self.collection_name}")
                return
        
        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=Distance.COSINE
            )
        )
        
        # Create payload indexes for efficient filtering
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="metadata.doc_type",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="metadata.reference_id",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        
        logger.info(f"Created collection: {self.collection_name}")
    
    def load_embedded_chunks(self, file_path: str) -> List[Dict]:
        """Load embedded chunks from a file.
        
        Args:
            file_path: Path to JSONL file containing embedded chunks
            
        Returns:
            List of dictionaries containing chunks with embeddings
        """
        chunks = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
        
        logger.info(f"Loaded {len(chunks)} embedded chunks from {file_path}")
        return chunks
    
    def upsert_chunks(self, chunks: List[Dict], batch_size: Optional[int] = None):
        """Insert or update chunks in Qdrant.
        
        Args:
            chunks: List of dictionaries containing chunks with embeddings
            batch_size: Batch size for insertion (defaults to constructor's qdrant_batch_size)
        """
        total_chunks = len(chunks)
        logger.info(f"Upserting {total_chunks} chunks to Qdrant")
        
        current_batch_size = batch_size if batch_size is not None else self.qdrant_batch_size

        # Process in batches
        for i in tqdm(range(0, total_chunks, current_batch_size), desc="Upserting to Qdrant"):
            batch = chunks[i:i+current_batch_size]
            
            # Prepare points for upsert
            points = []
            for j, chunk in enumerate(batch):
                # Create a unique ID based on chunk content
                chunk_id = hash(f"{chunk['metadata'].get('reference_id', '')}_{i}_{j}") % (2**63)
                
                # Extract embedding
                embedding = chunk.get("embedding")
                if not embedding:
                    logger.warning(f"Chunk missing embedding, skipping: {chunk['text'][:50]}...")
                    continue
                
                # Prepare payload directly (flattened)
                payload = chunk.get("metadata", {}).copy() # Start with existing metadata
                payload["text"] = chunk.get("text", "")     # Add text directly
                
                # Create point
                point = PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload=payload # Use the flattened payload
                )
                points.append(point)
            
            # Upsert batch
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
    
    def ingest_documents(self, doc_types: List[str] = ["circulars", "notifications"]):
        """Ingest documents into Qdrant.
        
        Args:
            doc_types: List of document types to ingest
        """
        # Create collection if it doesn't exist
        self.create_collection(recreate=False)
        
        # Process each document type
        for doc_type in doc_types:
            file_path = self.input_dir / f"{doc_type}_embedded.jsonl"
            
            if not file_path.exists():
                logger.warning(f"Embedded file not found: {file_path}")
                continue
            
            # Load embedded chunks
            chunks = self.load_embedded_chunks(str(file_path))
            
            # Add doc_type to metadata
            for chunk in chunks:
                if "metadata" not in chunk:
                    chunk["metadata"] = {}
                chunk["metadata"]["doc_type"] = doc_type
            
            # Upsert chunks to Qdrant
            self.upsert_chunks(chunks)
            
            logger.info(f"Completed ingestion for {doc_type}")
    
    def search(self, query_vector: List[float], limit: int = 5, filter_dict: Optional[Dict] = None, score_threshold: float = 0.0) -> List[Dict]:
        """Search for similar documents in Qdrant.
        
        Args:
            query_vector: Embedding vector of the query
            limit: Maximum number of results to return
            filter_dict: Optional filter for metadata fields
            score_threshold: Minimum score threshold for results
            
        Returns:
            List of dictionaries containing search results
        """
        # Prepare filter if provided
        filter_condition = None
        if filter_dict:
            filter_conditions = []
            for key, value in filter_dict.items():
                if key.startswith("metadata."):
                    filter_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                    )
            
            if filter_conditions:
                filter_condition = models.Filter(
                    must=filter_conditions
                )
        
        # Perform search
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=filter_condition,
            score_threshold=score_threshold
        )
        
        logger.info(f"QdrantRetriever.search: Found {len(search_result)} hits for vector query.")
        if not search_result:
            logger.warning(f"QdrantRetriever.search: No hits found for score_threshold {score_threshold}.")
        
        # Format results
        results = []
        for scored_point in search_result:
            result = {
                "id": scored_point.id,
                "score": scored_point.score,
                "metadata": scored_point.payload.get("metadata", {}),
                "text": scored_point.payload.get("metadata", {}).get("text", "")
            }
            results.append(result)
        
        return results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection.
        
        Returns:
            Dictionary containing collection information
        """
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            # Convert the response model to a dictionary for easier handling
            # Access attributes directly from the info object
            return {
                "status": "success",
                "vector_count": info.vectors_count,
                "indexed_vector_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                # Add other relevant fields from `info` if needed
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # Example usage
    retriever = QdrantRetriever()
    
    # Ingest documents
    retriever.ingest_documents()
    
    # Get collection info
    info = retriever.get_collection_info()
    print(f"Collection info: {info}")