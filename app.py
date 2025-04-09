import streamlit as st
from pathlib import Path
import tempfile
import logging
import os

from src.pipeline.query_pipeline import QueryPipeline
from src.utils.env_loader import load_environment_variables

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_MODEL = "models/gemini-1.5-flash"
DEFAULT_THRESHOLD = 0.5
DEFAULT_TOP_K = 5

# --- Functions ---
@st.cache_resource # Cache the pipeline resource across reruns
def get_pipeline():
    """Initializes and returns the QueryPipeline instance."""
    logger.info("Initializing QueryPipeline...")
    try:
        # Load keys just before initializing pipeline
        load_environment_variables()
        pipeline = QueryPipeline() # Uses defaults including the LLM model
        logger.info("QueryPipeline initialized successfully.")
        return pipeline
    except Exception as e:
        logger.error(f"Fatal error initializing QueryPipeline: {e}", exc_info=True)
        st.error(f"Fatal error initializing analysis pipeline: {e}. Please check logs and API key configuration.")
        return None # Indicate failure

# --- App Setup ---
st.set_page_config(
    page_title="BankRAG Compliance Checker",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 BankRAG Compliance Checker")
st.caption("Upload an internal policy document (PDF, DOCX, TXT, HTML) to analyze its compliance against RBI regulations.")

# --- Session State Initialization ---
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'last_analysis_params' not in st.session_state:
    st.session_state.last_analysis_params = {}

# --- Sidebar for Configuration ---
st.sidebar.header("Analysis Configuration")
# model_choice = st.sidebar.selectbox("LLM Model", [DEFAULT_MODEL], disabled=True) # Model fixed for now
top_k = st.sidebar.slider("Top K Regulations per Chunk", 1, 15, DEFAULT_TOP_K, 1)
score_threshold = st.sidebar.slider("Minimum Similarity Threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.05)

# --- Initialize Pipeline ---
pipeline = get_pipeline() 

if pipeline is None:
    st.warning("Pipeline could not be initialized. Analysis is unavailable. Please check API Key and logs.")
    # Stop execution if pipeline failed
    st.stop() 

# --- Main Area --- 
col1, col2 = st.columns([1, 2]) # Define columns for layout

with col1:
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a document", 
        type=["pdf", "docx", "txt", "html"], 
        help="Upload your internal policy document for analysis.",
        label_visibility="collapsed" # Hide label as subheader is present
    )

    analyze_button_disabled = uploaded_file is None
    analyze_button = st.button("Analyze Compliance", key="analyze_button", disabled=analyze_button_disabled, use_container_width=True)

    if uploaded_file is not None:
        st.info(f"Uploaded: **{uploaded_file.name}**")
        
        # Store filename if it changes
        if st.session_state.uploaded_file_name != uploaded_file.name:
             st.session_state.uploaded_file_name = uploaded_file.name
             st.session_state.analysis_results = None # Clear old results on new file
             st.session_state.last_analysis_params = {}

with col2:
    st.subheader("Analysis Report")
    
    # --- Run Analysis --- 
    if analyze_button and uploaded_file is not None:
        # Check if analysis needs to be rerun (new file or changed params)
        current_params = {"top_k": top_k, "threshold": score_threshold}
        if st.session_state.analysis_results is None or st.session_state.last_analysis_params != current_params:
            # Save uploaded file temporarily 
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
                logger.info(f"Saved uploaded file to temporary path: {tmp_file_path}")

            # Use a spinner during analysis
            with st.spinner(f"Analyzing {uploaded_file.name} (Top K: {top_k}, Threshold: {score_threshold})..."):
                try:
                    # Run the analysis with selected parameters
                    logger.info(f"Running analysis with top_k={top_k}, score_threshold={score_threshold}")
                    analysis_results = pipeline.analyze_document(
                        tmp_file_path, 
                        top_k=top_k, 
                        score_threshold=score_threshold
                    )
                    st.session_state.analysis_results = analysis_results # Store results
                    st.session_state.last_analysis_params = current_params # Store params used
                    logger.info("Analysis complete.")

                except Exception as e:
                    st.error(f"An unexpected error occurred during analysis: {e}")
                    logger.error(f"Analysis button error: {e}", exc_info=True)
                    st.session_state.analysis_results = {"error": str(e)} # Store error
                finally:
                    # Clean up the temporary file
                    if os.path.exists(tmp_file_path):
                        try:
                            os.remove(tmp_file_path)
                            logger.info(f"Removed temporary file: {tmp_file_path}")
                        except Exception as e_clean:
                            logger.warning(f"Could not remove temporary file {tmp_file_path}: {e_clean}")
            
            # Rerun script to display results after analysis finishes
            st.rerun() 

    # --- Display Results --- 
    if st.session_state.analysis_results is not None:
        results = st.session_state.analysis_results
        # Check for errors stored in session state
        if isinstance(results, dict) and "error" in results:
            st.error(f"Analysis failed: {results['error']}")
        elif not results or not results.get("chunk_analyses"):
            st.warning("Analysis completed, but no results were generated or the format is unexpected.")
        else:
            # Display the report
            try:
                report_text = pipeline.format_analysis_report(results)
                st.markdown(report_text, unsafe_allow_html=True) # Allow basic HTML in markdown if needed
            except Exception as e:
                st.error(f"Failed to format or display report: {e}")
                logger.error(f"Report formatting/display error: {e}", exc_info=True)
                st.json(results) # Show raw JSON as fallback
    elif uploaded_file is None:
         st.info("Please upload a document to begin analysis.")
    else:
         # Case where file is uploaded but analysis not yet run
         st.info("Click 'Analyze Compliance' to process the uploaded document.")

# Add footer or other elements if desired
# st.markdown("---")
# st.caption("Powered by BankRAG") 