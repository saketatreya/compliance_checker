import os
import sys
import argparse
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.embedding.embedder import DocumentEmbedder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def optimize_chunk_size(sample_text_path: str, min_size: int = 100, max_size: int = 2000, step: int = 100):
    """Optimize chunk size for embedding quality using a sample text.
    
    Args:
        sample_text_path: Path to sample text file
        min_size: Minimum chunk size to consider
        max_size: Maximum chunk size to consider
        step: Step size for chunk size evaluation
    """
    # Initialize embedder
    logger.info("Initializing document embedder")
    embedder = DocumentEmbedder()
    
    # Load sample text
    logger.info(f"Loading sample text from {sample_text_path}")
    with open(sample_text_path, "r", encoding="utf-8") as f:
        sample_text = f.read()
    
    # Generate range of chunk sizes to evaluate
    chunk_sizes = list(range(min_size, max_size + 1, step))
    logger.info(f"Evaluating chunk sizes from {min_size} to {max_size} with step {step}")
    
    # Evaluate chunk sizes
    results = embedder.evaluate_chunk_sizes(sample_text, chunk_sizes)
    
    # Print results
    logger.info("\nChunk Size Evaluation Results:")
    logger.info("-" * 40)
    logger.info("Chunk Size | Average Similarity")
    logger.info("-" * 40)
    for size, similarity in sorted(results.items()):
        logger.info(f"{size:10d} | {similarity:.4f}")
    
    # Find optimal chunk size
    if results:
        optimal_size = max(results.items(), key=lambda x: x[1])[0]
        logger.info(f"\nRecommended optimal chunk size: {optimal_size}")
    else:
        logger.warning("Could not determine optimal chunk size")

def main():
    """Main function for chunk size optimization."""
    parser = argparse.ArgumentParser(description="Optimize chunk size for embedding quality")
    
    parser.add_argument(
        "--sample-text",
        type=str,
        default="tests/test_document.txt",
        help="Path to sample text file"
    )
    
    parser.add_argument(
        "--min-size",
        type=int,
        default=200,
        help="Minimum chunk size to consider"
    )
    
    parser.add_argument(
        "--max-size",
        type=int,
        default=1500,
        help="Maximum chunk size to consider"
    )
    
    parser.add_argument(
        "--step",
        type=int,
        default=100,
        help="Step size for chunk size evaluation"
    )
    
    args = parser.parse_args()
    
    # Run optimization
    optimize_chunk_size(
        sample_text_path=args.sample_text,
        min_size=args.min_size,
        max_size=args.max_size,
        step=args.step
    )

if __name__ == "__main__":
    main()