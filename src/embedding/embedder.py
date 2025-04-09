import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import re
import textwrap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentEmbedder:
    """Generate vector embeddings for document chunks.
    
    This class handles the conversion of text chunks into vector embeddings
    using sentence transformer models.
    """
    
    def __init__(
        self, 
        model_name: str = "all-mpnet-base-v2", 
        input_dir: str = "data/processed", 
        output_dir: str = "data/embeddings"
    ):
        """Initialize the document embedder.
        
        Args:
            model_name: Name of the sentence transformer model to use
            input_dir: Directory containing processed text chunks
            output_dir: Directory to store embeddings
        """
        self.model_name = model_name
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load the embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info(f"Model loaded with embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array containing the embedding vector
        """
        return self.model.encode(text, show_progress_bar=False)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for embedding generation
            
        Returns:
            Numpy array containing the embedding vectors
        """
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    
    def process_chunks_file(self, file_path: str, batch_size: int = 32) -> List[Dict]:
        """Process a file containing text chunks and generate embeddings.
        
        Args:
            file_path: Path to JSONL file containing text chunks
            batch_size: Batch size for embedding generation
            
        Returns:
            List of dictionaries containing chunks with embeddings
        """
        logger.info(f"Processing chunks file: {file_path}")
        
        # Load chunks from file
        chunks = []
        texts = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                chunks.append(chunk)
                texts.append(chunk["text"])
        
        logger.info(f"Loaded {len(chunks)} chunks for embedding")
        
        # Generate embeddings in batches
        embeddings = self.embed_batch(texts, batch_size=batch_size)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i].tolist()
        
        return chunks
    
    def save_embeddings(self, chunks: List[Dict], output_file: str):
        """Save chunks with embeddings to file.
        
        Args:
            chunks: List of dictionaries containing chunks with embeddings
            output_file: Path to output file
        """
        logger.info(f"Saving {len(chunks)} embedded chunks to {output_file}")
        
        with open(output_file, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")
    
    def process_all_documents(self, doc_types: List[str] = ["circulars", "notifications"], batch_size: int = 32):
        """Process all document types and generate embeddings.
        
        Args:
            doc_types: List of document types to process
            batch_size: Batch size for embedding generation
        """
        for doc_type in doc_types:
            input_file = self.input_dir / f"{doc_type}_processed.jsonl"
            output_file = self.output_dir / f"{doc_type}_embedded.jsonl"
            
            if not input_file.exists():
                logger.warning(f"Input file not found: {input_file}")
                continue
            
            # Process chunks and generate embeddings
            chunks = self.process_chunks_file(str(input_file), batch_size=batch_size)
            
            # Save embeddings
            self.save_embeddings(chunks, str(output_file))
            
            logger.info(f"Completed embedding generation for {doc_type}")
    
    def evaluate_chunk_sizes(self, text: str, chunk_sizes: List[int], overlap_ratio: float = 0.1) -> Dict[int, float]:
        """Evaluate different chunk sizes for embedding quality.
        
        This method splits a text into chunks of different sizes and evaluates
        the quality of embeddings by measuring cosine similarity between
        adjacent chunks. Higher similarity between adjacent chunks indicates
        better semantic continuity.
        
        Args:
            text: Text to evaluate
            chunk_sizes: List of chunk sizes to evaluate
            overlap_ratio: Ratio of overlap between chunks
            
        Returns:
            Dictionary mapping chunk sizes to average similarity scores
        """
        logger.info(f"Evaluating {len(chunk_sizes)} different chunk sizes for embedding quality")
        
        results = {}
        
        for chunk_size in tqdm(chunk_sizes, desc="Evaluating chunk sizes"):
            # Calculate overlap size
            overlap = int(chunk_size * overlap_ratio)
            
            # Split text into chunks
            chunks = []
            for i in range(0, len(text), chunk_size - overlap):
                if i + chunk_size <= len(text):
                    chunks.append(text[i:i+chunk_size])
                else:
                    chunks.append(text[i:])
                    break
            
            # Skip if too few chunks
            if len(chunks) < 2:
                logger.warning(f"Chunk size {chunk_size} resulted in only {len(chunks)} chunks, skipping")
                continue
            
            # Generate embeddings
            embeddings = self.embed_batch(chunks)
            
            # Calculate cosine similarity between adjacent chunks
            similarities = []
            for i in range(len(embeddings) - 1):
                # Compute cosine similarity
                similarity = np.dot(embeddings[i], embeddings[i+1]) / \
                            (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]))
                similarities.append(similarity)
            
            # Calculate average similarity
            avg_similarity = np.mean(similarities)
            results[chunk_size] = avg_similarity
            
            logger.info(f"Chunk size {chunk_size}: average similarity = {avg_similarity:.4f}")
        
        return results
    
    def recommend_chunk_size(self, text: str, min_size: int = 100, max_size: int = 2000, step: int = 100) -> int:
        """Recommend an optimal chunk size for a given text.
        
        Args:
            text: Sample text to evaluate
            min_size: Minimum chunk size to consider
            max_size: Maximum chunk size to consider
            step: Step size for chunk size evaluation
            
        Returns:
            Recommended chunk size
        """
        logger.info(f"Finding optimal chunk size between {min_size} and {max_size}")
        
        # Generate range of chunk sizes to evaluate
        chunk_sizes = list(range(min_size, max_size + 1, step))
        
        # Evaluate chunk sizes
        results = self.evaluate_chunk_sizes(text, chunk_sizes)
        
        if not results:
            logger.warning("Could not determine optimal chunk size, using default of 1000")
            return 1000
        
        # Find chunk size with highest similarity score
        optimal_size = max(results.items(), key=lambda x: x[1])[0]
        
        logger.info(f"Recommended chunk size: {optimal_size}")
        return optimal_size

if __name__ == "__main__":
    # Example usage
    embedder = DocumentEmbedder()
    
    # Process all document types
    embedder.process_all_documents()
    
    # Example of chunk size optimization
    # Uncomment to run chunk size optimization
    """
    # Load a sample document for chunk size optimization
    sample_file_path = "tests/test_document.txt"
    with open(sample_file_path, "r", encoding="utf-8") as f:
        sample_text = f.read()
    
    # Find optimal chunk size
    optimal_size = embedder.recommend_chunk_size(
        text=sample_text,
        min_size=200,
        max_size=1500,
        step=100
    )
    print(f"Recommended chunk size for optimal embedding quality: {optimal_size}")
    """