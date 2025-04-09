import logging
import sys
from src.pipeline.query_pipeline import QueryPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_query_pipeline(doc_path: str):
    """Test the full QueryPipeline with a document."""
    logger.info(f"Testing QueryPipeline with document: {doc_path}")
    
    try:
        # Initialize pipeline (using default SentenceTransformer model)
        pipeline = QueryPipeline()
        
        # Analyze the document
        # Using default threshold and top_k from the pipeline method
        analysis_results = pipeline.analyze_document(doc_path)
        
        print("\n--- Analysis Results ---")
        # Check for error dictionary before proceeding
        if isinstance(analysis_results, dict) and "error" in analysis_results:
            print(f"Pipeline execution failed: {analysis_results['error']}")
            return
        
        # Check if results are empty (might happen if doc processing failed earlier)
        if not analysis_results or not analysis_results.get("chunk_analyses"):
             print("No analysis results generated or format is unexpected.")
             return

        # Iterate through chunk analyses from the results dictionary
        for i, result in enumerate(analysis_results.get("chunk_analyses", [])):
             print(f"\n--- Chunk {i+1} Analysis ---")
             # Use the correct key 'chunk_text' from the analysis result dict
             print(f"Original Chunk Text (Snippet):\n{result.get('chunk_text', '')[:200]}...")
             print("\nRetrieved Regulations:")
             # Use the correct key 'regulations'
             retrieved_regs = result.get('regulations')
             if retrieved_regs:
                 for j, reg in enumerate(retrieved_regs):
                     print(f"  Regulation {j+1} (Score: {reg.get('score', 0.0):.4f}):")
                     print(f"    Source: {reg.get('metadata', {}).get('source_file', 'Unknown')}")
                     print(f"    Text: {reg.get('text', '')[:200]}...")
             else:
                 print("  No relevant regulations retrieved for this chunk.")
             
             print("\nLLM Analysis:")
             # Use the correct key 'analysis'
             print(result.get('analysis', "No analysis provided."))
             print("-" * 50)

    except Exception as e:
        logger.error(f"Error during QueryPipeline test: {e}", exc_info=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/test_query_pipeline.py <path_to_document>")
        # Default to one of the sample docs for testing if none provided
        # Use the HTML file as it's simpler than PDF/DOCX for this test
        default_doc = "data/raw/notifications/2022-06-10_.html"
        print(f"No document specified, using default: {default_doc}")
        test_doc_path = default_doc
    else:
        test_doc_path = sys.argv[1]
        
    test_query_pipeline(test_doc_path) 