 """
Mock embedding generator for the BankRAG system.

This module provides a simple mock implementation of embedding generators
to avoid dependency issues with HuggingFace/transformers.
"""

import hashlib
import numpy as np
from typing import Dict, List, Optional

from bankrag.core.models import TextChunk
from bankrag.processing.embeddings import BaseEmbeddingGenerator
from bankrag.utils.logger import get_logger

logger = get_logger()


class MockEmbeddingGenerator(BaseEmbeddingGenerator):
    """
    Mock embedding generator for demonstration purposes.
    
    This generator creates deterministic embeddings based on text hash,
    allowing for consistent retrieval without external dependencies.
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        seed: int = 42,
    ):
        """
        Initialize the MockEmbeddingGenerator.
        
        Args:
            embedding_dim: Dimension of the embeddings to generate.
            seed: Random seed for reproducibility.
        """
        self.embedding_dim = embedding_dim
        self.rng = np.random.RandomState(seed)
        logger.info(f"Initialized MockEmbeddingGenerator with {embedding_dim} dimensions")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate a deterministic embedding for a single text.
        
        Args:
            text: Text to embed.
            
        Returns:
            Numpy array containing the embedding vector.
        """
        # Create a deterministic hash of the text
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert the hash to a seed for random number generation
        seed = int(text_hash, 16) % (2**32)
        
        # Create a random generator with this seed
        rng = np.random.RandomState(seed)
        
        # Generate a random vector
        embedding = rng.randn(self.embedding_dim)
        
        # Normalize the vector to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of numpy arrays containing the embedding vectors.
        """
        return [self.embed_text(text) for text in texts]
    
    def embed_chunk(self, chunk: TextChunk) -> np.ndarray:
        """
        Generate embedding for a TextChunk.
        
        Args:
            chunk: TextChunk to embed.
            
        Returns:
            Numpy array containing the embedding vector.
        """
        return self.embed_text(chunk.content)
    
    def embed_chunks(self, chunks: List[TextChunk]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for multiple TextChunks.
        
        Args:
            chunks: List of TextChunks to embed.
            
        Returns:
            Dictionary mapping chunk IDs to embedding vectors.
        """
        logger.info(f"Generating mock embeddings for {len(chunks)} chunks")
        
        result = {}
        for chunk in chunks:
            result[chunk.id] = self.embed_chunk(chunk)
        
        logger.info(f"Generated embeddings for {len(result)} chunks")
        return result
