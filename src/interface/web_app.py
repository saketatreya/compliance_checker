import os
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import sys
import re

import streamlit as st
from streamlit.components.v1 import html

from src.pipeline.query_pipeline import QueryPipeline
from src.utils.env_loader import load_environment_variables

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src directory to Python path
current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir.parent.parent)) # Adjust based on actual structure

# Set page configuration
st.set_page_config(
    page_title="BankRAG: RBI Compliance Checker",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .citation {
        background-color: #E5E7EB;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    .insight-box {
        background-color: #F0FDF4;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #10B981;
    }
    .issue-box {
        background-color: #FEF2F2;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #EF4444;
    }
    .recommendation-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "selected_chunk" not in st.session_state:
        st.session_state.selected_chunk = 0


def load_pipeline():
    """Load the query pipeline if not already loaded."""
    if st.session_state.pipeline is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            st.error("GOOGLE_API_KEY environment variable not set. Analysis will not work.")
            st.stop()
        try:
            # Load environment variables before initializing pipeline
            load_environment_variables()
            st.session_state.pipeline = QueryPipeline(google_api_key=api_key)
            st.success("Compliance checker initialized!")
        except Exception as e:
            st.error(f"Failed to initialize analysis pipeline: {e}")
            logger.error(f"Pipeline initialization error: {e}", exc_info=True)
            st.stop()


def process_uploaded_file(uploaded_file, top_k, score_threshold):
    """Process an uploaded file and generate compliance analysis.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        top_k: Number of relevant regulations to retrieve per chunk
        score_threshold: Minimum similarity score for retrieved regulations
    """
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        # Analyze document
        with st.spinner("Analyzing document for compliance..."):
            results = st.session_state.pipeline.analyze_document(
                document_path=tmp_file_path,
                top_k=top_k,
                score_threshold=score_threshold
            )
            
            if "error" in results:
                st.error(f"Analysis failed: {results['error']}")
                st.session_state.analysis_results = None
            else:
                # Add document name to results
                results["document_name"] = uploaded_file.name
                
                # Store results in session state
                st.session_state.analysis_results = results
                st.session_state.selected_chunk = 0
    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
        logger.error(f"Analysis execution error: {e}", exc_info=True)
        st.session_state.analysis_results = None
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_file_path)
        except Exception as e_unlink:
            logger.error(f"Error removing temporary file {tmp_file_path}: {e_unlink}")


def display_sidebar():
    """Display and handle the sidebar components."""
    st.sidebar.markdown("<h2>BankRAG</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("RBI Compliance Checker")
    st.sidebar.markdown("---")
    
    # File upload
    st.sidebar.markdown("### Upload Document")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a document to analyze",
        type=["txt", "pdf", "docx"],
        help="Supported formats: TXT, PDF, DOCX"
    )
    
    # Analysis parameters
    st.sidebar.markdown("### Analysis Parameters")
    top_k = st.sidebar.slider(
        "Number of regulations to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="Higher values may provide more comprehensive analysis but take longer"
    )
    score_threshold = st.sidebar.slider(
        "Minimum Similarity Score",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Minimum relevance score for retrieved regulations (higher = stricter)"
    )
    
    # Analyze button
    if uploaded_file is not None:
        if st.sidebar.button("Analyze Document"):
            load_pipeline()
            if st.session_state.pipeline:
                process_uploaded_file(uploaded_file, top_k, score_threshold)
            else:
                st.error("Analysis pipeline could not be loaded. Check logs or API key.")
    
    # Display chunk selector if results are available
    if st.session_state.analysis_results is not None and "error" not in st.session_state.analysis_results:
        st.sidebar.markdown("### Document Sections")
        chunk_analyses = st.session_state.analysis_results.get("chunk_analyses", [])
        
        if chunk_analyses:
            # Use chunk text excerpt as option identifier if available
            def format_chunk_option(i):
                chunk = chunk_analyses[i]
                text = chunk.get("chunk_text", "")
                excerpt = text[:50] + ("..." if len(text) > 50 else "")
                return f"Section {i+1}: '{excerpt}'"

            selected_index = st.sidebar.selectbox(
                "Select section to view",
                range(len(chunk_analyses)),
                format_func=format_chunk_option,
                key="chunk_selector"
            )
            # Update selected chunk based on selection box
            if st.session_state.selected_chunk != selected_index:
                st.session_state.selected_chunk = selected_index
                st.experimental_rerun() # Rerun to update main display
        else:
            st.sidebar.info("Document analysis yielded no sections.")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "BankRAG is a Retrieval-Augmented Generation system for RBI regulatory compliance. "
        "It uses LLMs to analyze documents against RBI circulars and notifications."
    )


def display_analysis_results():
    """Display the analysis results."""
    if st.session_state.analysis_results is None:
        st.markdown(
            "<div class='info-box'>"
            "<h3>Welcome to BankRAG: RBI Compliance Checker</h3>"
            "<p>Upload a document using the sidebar to analyze it for compliance with RBI regulations.</p>"
            "</div>",
            unsafe_allow_html=True
        )
        return
    
    if "error" in st.session_state.analysis_results:
        # Error already shown by process_uploaded_file
        return

    results = st.session_state.analysis_results
    document_name = results.get("document_name", "Document")
    chunk_analyses = results.get("chunk_analyses", [])
    
    # Display document header
    st.markdown(f"<h1 class='main-header'>Compliance Analysis: {document_name}</h1>", unsafe_allow_html=True)
    
    if not chunk_analyses:
        st.warning("No analysis results available for this document.")
        return
    
    # Get selected chunk
    selected_index = st.session_state.selected_chunk
    if selected_index >= len(chunk_analyses):
        st.warning("Selected section index is out of bounds. Resetting to first section.")
        selected_index = 0
        st.session_state.selected_chunk = 0
    
    selected_analysis = chunk_analyses[selected_index]
    
    # Display chunk info
    st.markdown(f"<h2 class='sub-header'>Section {selected_index + 1} of {len(chunk_analyses)}</h2>", unsafe_allow_html=True)
    
    # Display document text for the current chunk
    with st.expander("Document Text (Section)", expanded=False):
        st.markdown("```\n" + selected_analysis.get("chunk_text", "No text available") + "\n```")
    
    # Display compliance insights from LLM
    analysis_text = selected_analysis.get("analysis", "No analysis available.")
    
    # Simple parsing based on expected format (can be improved)
    summary_content = ""
    issues_content = ""
    recommendations_content = ""
    current_section_parse = None

    for line in analysis_text.split('\n'):
        line_strip = line.strip()
        # Check for section headers like "1. Summary:", "2. Issues:", "3. Recommendations:"
        if re.match(r"^\d+\.\s+Summary:?", line_strip, re.IGNORECASE):
            current_section_parse = "summary"
            summary_content += re.sub(r"^\d+\.\s+Summary:?", "", line_strip, flags=re.IGNORECASE).strip() + "\n"
        elif re.match(r"^\d+\.\s+Issues:?", line_strip, re.IGNORECASE):
            current_section_parse = "issues"
            issues_content += re.sub(r"^\d+\.\s+Issues:?", "", line_strip, flags=re.IGNORECASE).strip() + "\n"
        elif re.match(r"^\d+\.\s+Recommendations:?", line_strip, re.IGNORECASE):
            current_section_parse = "recommendations"
            recommendations_content += re.sub(r"^\d+\.\s+Recommendations:?", "", line_strip, flags=re.IGNORECASE).strip() + "\n"
        elif current_section_parse == "summary":
            summary_content += line + "\n"
        elif current_section_parse == "issues":
            issues_content += line + "\n"
        elif current_section_parse == "recommendations":
            recommendations_content += line + "\n"
        elif current_section_parse is None and line_strip: # Capture any leading text as summary
             # Avoid capturing if it looks like a section header was just missed
             if not re.match(r"^\d+\.\s+(?:Issues|Recommendations):?", line_strip, re.IGNORECASE):
                summary_content += line + "\n"
                current_section_parse = "summary" # Assume start of summary

    st.markdown("<h3>Compliance Analysis</h3>", unsafe_allow_html=True)
    if summary_content.strip():
        st.markdown("**Summary**")
        st.markdown(f"<div class='insight-box'>{summary_content.strip()}</div>", unsafe_allow_html=True)
    
    if issues_content.strip():
        st.markdown("**Issues Identified**")
        # Use markdown within the div for better formatting (like bullet points)
        st.markdown(f"<div class='issue-box'>{issues_content.strip()}</div>", unsafe_allow_html=True)
    elif current_section_parse == "issues": # Handle case where Issues section is empty
         st.markdown("**Issues Identified**")
         st.markdown(f"<div class='info-box'>No specific issues mentioned in analysis.</div>", unsafe_allow_html=True)

    
    if recommendations_content.strip():
        st.markdown("**Recommendations**")
        st.markdown(f"<div class='recommendation-box'>{recommendations_content.strip()}</div>", unsafe_allow_html=True)
        
    # Fallback if parsing fails or yields no content
    if not summary_content.strip() and not issues_content.strip() and not recommendations_content.strip():
         st.markdown("**Analysis Output**")
         st.markdown(f"<div class='info-box'>{analysis_text}</div>", unsafe_allow_html=True) 

    
    # Display retrieved regulations for the current chunk
    retrieved_regulations = selected_analysis.get("regulations", [])
    if retrieved_regulations:
        st.markdown("<h3>Retrieved Regulations Used for this Section</h3>", unsafe_allow_html=True)
        for i, reg in enumerate(retrieved_regulations):
            metadata = reg.get("metadata", {})
            score = reg.get("score", 0.0)
            source_file = metadata.get("source_file", "N/A")
            source = Path(source_file).name if source_file != "N/A" else "N/A"
            ref_id = metadata.get("reference_id", metadata.get("title", "Unknown")) # Use ref_id or title
            section = metadata.get("section", None)
            subsection = metadata.get("subsection", None)
            clause = metadata.get("clause_id", None)
            page = metadata.get("page", None)
            
            citation = f"{ref_id} ({source})"
            context_parts = []
            if section and section != "Main": context_parts.append(f"Sec: {section}")
            if subsection: context_parts.append(f"Sub: {subsection}")
            if clause: context_parts.append(f"Clause: {clause}")
            if page: context_parts.append(f"Page: {page}")
            if context_parts: citation += f" [{', '.join(context_parts)}]"
            
            with st.expander(f"Regulation {i+1}: {citation} (Score: {score:.3f})"):
                # Display regulation text
                st.markdown("```\n" + reg.get("text", "No text available.") + "\n```")
                # Display full metadata for debugging/info
                # st.json(metadata)
    else:
        st.info("No specific regulations were retrieved for this section.")


def main():
    """Main function for the Streamlit app."""
    # Load environment variables
    load_environment_variables()
    
    # Initialize session state
    initialize_session_state()
    
    # Display sidebar
    display_sidebar()
    
    # Display main content
    display_analysis_results()


if __name__ == "__main__":
    main()