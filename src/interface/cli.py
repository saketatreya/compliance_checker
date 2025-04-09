import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# Add src directory to Python path to allow imports
current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir.parent))

from src.pipeline.ingest_pipeline import IngestionPipeline
from src.pipeline.query_pipeline import QueryPipeline
from src.utils.env_loader import load_environment_variables

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComplianceCheckerCLI:
    """Command-line interface for the BankRAG compliance checker.
    
    This class provides a user-friendly CLI for analyzing documents
    for compliance with RBI regulations.
    """
    
    def __init__(self):
        """Initialize the CLI."""
        self.parser = self._create_parser()
        self.pipeline = None
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser.
        
        Returns:
            Configured argument parser
        """
        parser = argparse.ArgumentParser(
            description="BankRAG: RBI Compliance Checker CLI"
        )
        
        subparsers = parser.add_subparsers(dest="command", required=True)
        
        # --- Ingest Command ---
        ingest_parser = subparsers.add_parser("ingest", help="Run the data ingestion pipeline")
        ingest_parser.add_argument(
            "--doc-types", 
            nargs='+', 
            default=["circulars", "notifications"],
            help="Types of documents to process (e.g., circulars notifications)"
        )
        ingest_parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host")
        ingest_parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
        ingest_parser.add_argument("--collection", default="rbi_regulations", help="Qdrant collection name")
        ingest_parser.add_argument("--vector-size", type=int, default=768, help="Embedding vector size")
        ingest_parser.add_argument("--batch-size", type=int, default=32, help="Processing batch size")
        ingest_parser.add_argument("--model", default="all-mpnet-base-v2", help="Embedding model name")

        # --- Analyze Command ---
        analyze_parser = subparsers.add_parser("analyze", help="Analyze a document for compliance")
        analyze_parser.add_argument("document_path", help="Path to the document file (txt, pdf, docx)")
        analyze_parser.add_argument(
            "--top-k", 
            type=int, 
            default=5, 
            help="Number of relevant regulations to retrieve per chunk"
        )
        analyze_parser.add_argument(
            "--threshold", 
            type=float, 
            default=None, # Use retriever's default or pipeline default
            help="Minimum similarity score threshold for retrieval"
        )
        analyze_parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host")
        analyze_parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
        analyze_parser.add_argument("--collection", default="rbi_regulations", help="Qdrant collection name")
        analyze_parser.add_argument("--embedding-model", default="all-mpnet-base-v2", help="Embedding model used for retrieval (must match ingestion)")
        analyze_parser.add_argument("--llm-model", default="gemini-pro", help="LLM model for analysis")

        return parser
    
    def _setup_logging(self, verbose: bool, debug: bool):
        """Set up logging based on verbosity level.
        
        Args:
            verbose: Whether to enable verbose output
            debug: Whether to enable debug mode
        """
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")
        elif verbose:
            logging.getLogger().setLevel(logging.INFO)
            logger.info("Verbose mode enabled")
        else:
            logging.getLogger().setLevel(logging.WARNING)
    
    def _format_analysis_results(self, results: Dict) -> str:
        """Format analysis results for display.
        
        Args:
            results: Analysis results from the pipeline
            
        Returns:
            Formatted results as a string
        """
        output = []
        output.append("BankRAG: RBI Compliance Analysis")
        output.append("=" * 50)
        
        # Document info
        doc_path = results.get("document_path", "Unknown document")
        output.append(f"Document: {Path(doc_path).name}")
        output.append("-" * 50)
        
        # Chunk analyses
        chunk_analyses = results.get("chunk_analyses", [])
        for i, analysis in enumerate(chunk_analyses):
            output.append(f"\nChunk {i+1}/{len(chunk_analyses)}")
            output.append("-" * 40)
            
            # Show a brief excerpt of the chunk text
            chunk_text = analysis.get("chunk_text", "")
            if chunk_text:
                excerpt = chunk_text[:150] + "..." if len(chunk_text) > 150 else chunk_text
                output.append(f"Excerpt: {excerpt}\n")
            
            # Compliance insights
            analysis_text = analysis.get("analysis", "No analysis available")
            output.append(analysis_text)
            
            # Referenced regulations
            regulations = analysis.get("regulations", [])
            if regulations:
                output.append("\nRelevant Regulations:")
                for j, reg in enumerate(regulations[:3], 1):
                    metadata = reg.get("metadata", {})
                    ref_id = metadata.get("reference_id", "Unknown")
                    title = metadata.get("title", "")
                    date = metadata.get("date", "")
                    citation = f"{ref_id}"
                    if title:
                        citation += f": {title}"
                    if date:
                        citation += f" ({date})"
                    output.append(f"  {j}. {citation}")
                
                if len(regulations) > 3:
                    output.append(f"  ... and {len(regulations) - 3} more")
        
        # Summary of all unique regulations
        all_regulations = results.get("all_regulations", [])
        if all_regulations:
            output.append("\n" + "-" * 50)
            output.append("Summary of Referenced Regulations:")
            unique_refs = {}
            for reg in all_regulations:
                metadata = reg.get("metadata", {})
                ref_id = metadata.get("reference_id", "Unknown")
                if ref_id not in unique_refs:
                    unique_refs[ref_id] = metadata
            
            for i, (ref_id, metadata) in enumerate(unique_refs.items(), 1):
                title = metadata.get("title", "")
                output.append(f"  {i}. {ref_id}{': ' + title if title else ''}")
        
        # Summary
        output.append("\n" + "=" * 50)
        output.append("Analysis complete")
        
        return "\n".join(output)
    
    def run(self):
        """Parse arguments and execute the corresponding command."""
        args = self.parser.parse_args()
        
        if args.command == "ingest":
            self.run_ingestion(args)
        elif args.command == "analyze":
            self.run_analysis(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            self.parser.print_help()

    def run_ingestion(self, args):
        """Execute the ingestion pipeline."""
        logger.info("Starting ingestion process...")
        try:
            pipeline = IngestionPipeline(
                qdrant_host=args.qdrant_host,
                qdrant_port=args.qdrant_port,
                qdrant_collection=args.collection,
                embedding_model=args.model,
                vector_size=args.vector_size,
                batch_size=args.batch_size
            )
            pipeline.run_pipeline(doc_types=args.doc_types)
            logger.info("Ingestion process completed successfully.")
        except Exception as e:
            logger.error(f"Ingestion process failed: {e}", exc_info=True)

    def run_analysis(self, args):
        """Execute the document analysis pipeline."""
        logger.info(f"Starting analysis for document: {args.document_path}")
        
        if not Path(args.document_path).is_file():
            logger.error(f"Document file not found: {args.document_path}")
            return
            
        # Check for Google API key
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
             logger.warning("GOOGLE_API_KEY environment variable not set. LLM analysis may fail.")
             # Proceed anyway, QueryPipeline handles internal checks

        try:
            pipeline = QueryPipeline(
                qdrant_host=args.qdrant_host,
                qdrant_port=args.qdrant_port,
                qdrant_collection=args.collection,
                embedding_model_name=args.embedding_model,
                llm_model_name=args.llm_model,
                google_api_key=api_key # Pass explicitly if needed, or let pipeline use env var
            )
            
            results = pipeline.analyze_document(
                document_path=args.document_path,
                top_k=args.top_k,
                score_threshold=args.threshold
            )
            
            # Format and print results
            self._print_analysis_results(results)
            
        except Exception as e:
            logger.error(f"Analysis process failed: {e}", exc_info=True)

    def _print_analysis_results(self, results: Dict):
        """Format and print the analysis results to the console."""
        if not results or "error" in results:
            print(f"\nAnalysis failed: {results.get('error', 'Unknown error')}")
            return
        
        print(f"\n--- Compliance Analysis Report for: {results.get('document_path')} ---")
        print(f"Document processed into {results.get('num_chunks', 0)} sections/chunks.")
        
        chunk_analyses = results.get("chunk_analyses", [])
        if not chunk_analyses:
            print("\nNo analysis results generated.")
            return

        for i, analysis in enumerate(chunk_analyses):
            print(f"\n=== Section {i+1} == worrisome?{len(chunk_analyses)} ===")
            
            # Show a brief excerpt of the chunk text
            chunk_text = analysis.get("chunk_text", "")
            if chunk_text:
                excerpt = chunk_text[:200] + ("..." if len(chunk_text) > 200 else "")
                print(f"\nDocument Text Excerpt:\n'''\n{excerpt}\n'''")
            
            # Compliance insights
            analysis_text = analysis.get("analysis", "No analysis available")
            print("\nCompliance Analysis:")
            print(analysis_text)
            
            # Referenced regulations
            regulations = analysis.get("regulations", [])
            if regulations:
                print("\nRelevant Regulations Considered:")
                for j, reg in enumerate(regulations, 1):
                    metadata = reg.get("metadata", {})
                    score = reg.get("score", 0.0)
                    ref_id = metadata.get("reference_id", metadata.get("source_file", "Unknown")) # Use ref_id or source
                    clause = metadata.get("clause_id", None)
                    page = metadata.get("page", None)
                    
                    citation = f"{ref_id}"
                    if clause: citation += f" (Clause: {clause})"
                    if page: citation += f" (Page: {page})"
                    
                    print(f"  {j}. {citation} (Score: {score:.3f})")
                    # Optionally print regulation text excerpt
                    # reg_excerpt = reg.get("text", "")[:100] + "..."
                    # print(f"     Text: {reg_excerpt}")
        
        # Summary of all unique regulations used in the analysis
        all_regulations = results.get("all_regulations", [])
        if all_regulations:
             print("\n--- Summary of Unique Regulations Referenced ---")
             unique_citations = set()
             for reg in all_regulations:
                 metadata = reg.get("metadata", {})
                 ref_id = metadata.get("reference_id", metadata.get("source_file", "Unknown"))
                 unique_citations.add(ref_id)
             for citation in sorted(list(unique_citations)):
                 print(f"- {citation}")
                 
        print("\n--- End of Report ---")


def main():
    """Main entry point for the CLI."""
    # Load environment variables from .env file
    load_environment_variables()
    
    cli = ComplianceCheckerCLI()
    sys.exit(cli.run())


if __name__ == "__main__":
    main()