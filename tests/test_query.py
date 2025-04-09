import os
import sys
import argparse
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.query_pipeline import QueryPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_query_pipeline():
    """Test the query pipeline with a sample document."""
    # Initialize the pipeline
    logger.info("Initializing query pipeline")
    pipeline = QueryPipeline()
    
    # Path to test document
    test_doc_path = Path(__file__).parent / "test_document.txt"
    
    if not test_doc_path.exists():
        logger.error(f"Test document not found: {test_doc_path}")
        return False
    
    # Analyze document
    logger.info(f"Analyzing document: {test_doc_path}")
    try:
        results = pipeline.analyze_document(
            document_path=str(test_doc_path),
            top_k=3  # Limit to 3 regulations for faster testing
        )
        
        # Check results
        if not results or "chunk_analyses" not in results:
            logger.error("No analysis results returned")
            return False
        
        # Print summary
        chunk_analyses = results.get("chunk_analyses", [])
        logger.info(f"Successfully analyzed document into {len(chunk_analyses)} chunks")
        
        # Print first chunk insights
        if chunk_analyses:
            logger.info("Sample insights from first chunk:")
            insights = chunk_analyses[0].get("insights", "No insights available")
            logger.info(insights[:500] + "..." if len(insights) > 500 else insights)
        
        return True
        
    except Exception as e:
        logger.error(f"Error analyzing document: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function for testing."""
    parser = argparse.ArgumentParser(description="Test the BankRAG query pipeline")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run test
    logger.info("Starting query pipeline test")
    success = test_query_pipeline()
    
    if success:
        logger.info("Test completed successfully")
        return 0
    else:
        logger.error("Test failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())