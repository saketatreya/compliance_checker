import logging
import sys
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_retrieval(query: str = "card tokenisation", top_k: int = 5, threshold: float = 0.5):
    """Test retrieval from Qdrant using a query."""
    logger.info(f"Testing retrieval for query: '{query}' with similarity threshold: {threshold}, top_k: {top_k}")
    
    # Initialize Qdrant client
    qdrant_client = QdrantClient(host="localhost", port=6333)
    logger.info(f"Connected to Qdrant at localhost:6333")
    
    # Use SentenceTransformer for embeddings
    embedding_model_name = "msmarco-distilbert-base-tas-b"
    st_model = SentenceTransformer(embedding_model_name)
    logger.info(f"Using SentenceTransformer '{embedding_model_name}' for query embedding.")
    
    # Embed the query
    query_vector = st_model.encode(query).tolist()
    
    # Search in Qdrant
    collection_name = "rbi_regulations"
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=None,  # Add filters if needed
        limit=top_k,
        score_threshold=threshold
    )
    
    logger.info(f"Retrieved {len(search_result)} documents.")
    
    # Display results
    print(f"\nResults for query: '{query}' (threshold: {threshold})")
    print("=" * 50)
    
    for i, hit in enumerate(search_result):
        score = hit.score
        text = hit.payload.get("text", "")
        metadata = hit.payload
        
        # Extract metadata for display
        doc_source = metadata.get("source_file", "Unknown")
        doc_ref = metadata.get("reference_id", "Unknown Reference")
        doc_date = metadata.get("date", "Unknown Date")
        
        print(f"\nDocument {i+1} (Score: {score:.4f}):")
        print(f"Source: {doc_source}")
        print(f"Reference: {doc_ref}")
        print(f"Date: {doc_date}")
        print("-" * 40)
        print(text[:500] + "..." if len(text) > 500 else text)
        print("-" * 40)

if __name__ == "__main__":
    # Use command-line argument as query if provided
    query = "card tokenisation"
    if len(sys.argv) > 1:
        query = sys.argv[1]
    
    test_retrieval(query) 