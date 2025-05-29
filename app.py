import streamlit as st
from pathlib import Path
import tempfile
import logging
import os
import re
import pandas as pd
import plotly.graph_objects as go
from collections import Counter # Import Counter

from src.pipeline.query_pipeline import QueryPipeline
from src.utils.env_loader import load_environment_variables

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_MODEL = "models/gemini-1.5-flash"
DEFAULT_THRESHOLD = 0.5
DEFAULT_TOP_K = 5
STATUS_COLORS = {
    "Compliant": "green",
    "Partially Compliant": "orange",
    "Non-Compliant": "red",
    "Unable to Assess": "gray"
}
STATUS_ICONS = {
    "Compliant": "🟢",
    "Partially Compliant": "🟡",
    "Non-Compliant": "🔴",
    "Unable to Assess": "⚪"
}

# --- Helper Functions ---

def parse_llm_analysis(analysis_text: str, section_index: int) -> dict:
    """Parses the LLM analysis text to extract structured data based on the new prompt format."""
    parsed_data = {
        "llm_section_title": f"Section {section_index+1} Analysis", # Default title
        "compliance_status": "Unable to Assess", # Default status
        "compliance_pct": 0,
        "compliance_verdict": "Unable to Assess", # Default verdict
        "action_items": [],
        "severity_counts": {"High": 0, "Medium": 0, "Low": 0},
        "metrics": {
            "aspects_assessed": 0,
            "compliant_items": 0,
            "partially_compliant_items": 0,
            "noncompliant_items": 0,
            "unable_to_assess_items": 0
        },
        "detailed_assessment": "", # Store the detailed assessment part
        "referenced_regulations": "", # Store the referenced regulations part
        "raw_analysis": analysis_text # Keep raw text for reference
    }

    # 1. Extract Section Title
    title_match = re.search(r"- Section Title: (.*?)\n", analysis_text)
    if title_match:
        title = title_match.group(1).strip()
        # Basic cleaning, remove potential leading numbers/letters and trailing punctuation
        title = re.sub(r'^[A-Za-z0-9]+\.\s*', '', title).strip()
        title = re.sub(r'[.,;:!?]$', '', title).strip()
        if title: # Ensure title is not empty after cleaning
            parsed_data["llm_section_title"] = title

    # 2. Extract Overall Compliance Percentage and Verdict
    overall_match = re.search(r"- Overall Compliance: (\d+)% \((.*?)\)\n", analysis_text)
    if overall_match:
        parsed_data["compliance_pct"] = int(overall_match.group(1))
        parsed_data["compliance_verdict"] = overall_match.group(2).strip()
        # Determine status based on verdict (more reliable than just percentage now)
        verdict = parsed_data["compliance_verdict"]
        if verdict == "Fully Compliant":
            parsed_data["compliance_status"] = "Compliant"
        elif verdict == "Largely Compliant" or verdict == "Minor Issues Found":
            parsed_data["compliance_status"] = "Partially Compliant"
        elif verdict == "Significant Issues Found":
            parsed_data["compliance_status"] = "Non-Compliant"
        else: # Includes "Unable to Assess"
            parsed_data["compliance_status"] = "Unable to Assess"
    else:
        # Fallback if overall line not found (shouldn't happen with new prompt)
        if "unable to assess" in analysis_text.lower():
            parsed_data["compliance_status"] = "Unable to Assess"
            parsed_data["compliance_verdict"] = "Unable to Assess"

    # 3. Extract Compliance Metrics (Counts)
    metrics_patterns = {
        "aspects_assessed": r"- Number of Policy Aspects Assessed: (\d+)",
        "compliant_items": r"- Compliant Items: (\d+)",
        "partially_compliant_items": r"- Partially Compliant Items: (\d+)",
        "noncompliant_items": r"- Non-Compliant Items: (\d+)",
        "unable_to_assess_items": r"- Items Unable to Assess: (\d+)"
    }
    for key, pattern in metrics_patterns.items():
        match = re.search(pattern, analysis_text)
        if match:
            parsed_data["metrics"][key] = int(match.group(1))

    # 4. Extract Prioritized Action Items
    action_items_section_match = re.search(r"\*\*2\. Prioritized Action Items:\*\*\s*(.*?)(?=\n\*\*3\. Detailed Assessment|$)", analysis_text, re.DOTALL)
    if action_items_section_match:
        action_items_text = action_items_section_match.group(1).strip()
        if "No action items required" not in action_items_text:
            # Regex to capture severity, aspect, and recommendation
            action_item_matches = re.finditer(r"\d+\. \[(High|Medium|Low) Severity\] (.*?): (.*?)(?=\n\d+\. |$)", action_items_text, re.DOTALL)
            for match in action_item_matches:
                severity = match.group(1).strip()
                policy_aspect = match.group(2).strip()
                recommendation = match.group(3).strip()
                parsed_data["severity_counts"][severity] += 1
                parsed_data["action_items"].append({
                    "section_title": parsed_data["llm_section_title"], # Use processed title
                    "severity": severity,
                    "policy_aspect": policy_aspect,
                    "recommendation": recommendation
                })

    # 5. Extract Detailed Assessment Section
    detailed_assessment_match = re.search(r"\*\*3\. Detailed Assessment:\*\*\s*(.*?)(?=\n\*\*4\. Referenced Regulations|$)", analysis_text, re.DOTALL)
    if detailed_assessment_match:
        parsed_data["detailed_assessment"] = detailed_assessment_match.group(1).strip()

    # 6. Extract Referenced Regulations Section
    referenced_regulations_match = re.search(r"\*\*4\. Referenced Regulations:\*\*\s*(.*?)(?=\n\*\*Context|$)", analysis_text, re.DOTALL)
    if referenced_regulations_match:
        parsed_data["referenced_regulations"] = referenced_regulations_match.group(1).strip()

    return parsed_data

def calculate_overall_summary(processed_sections: list) -> dict:
    """Calculates overall summary statistics from processed sections."""
    summary = {
        "total_sections": len(processed_sections),
        "avg_compliance_pct": 0,
        "section_status_counts": Counter(),
        "total_item_metrics": Counter(),
        "total_severity_counts": Counter(),
        "all_action_items": [],
        "non_compliant_sections": []
    }

    if not processed_sections:
        return summary

    total_pct = 0
    for section in processed_sections:
        # Revert to direct access - keys should be present now
        total_pct += section["compliance_pct"]
        summary["section_status_counts"][section["compliance_status"]] += 1
        summary["total_item_metrics"].update(section["metrics"])
        summary["total_severity_counts"].update(section["severity_counts"])
        summary["all_action_items"].extend(section["action_items"])
        if section["compliance_status"] == "Non-Compliant":
            summary["non_compliant_sections"].append(section["section_title"])

    # Add check for division by zero
    if summary["total_sections"] > 0:
        summary["avg_compliance_pct"] = round(total_pct / summary["total_sections"])
    else:
        summary["avg_compliance_pct"] = 0 # Ensure it's 0 if no sections

    # Sort action items by severity
    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    summary["all_action_items"].sort(key=lambda x: severity_order.get(x["severity"], 3))

    return summary

def create_compliance_donut_chart(status_counts: Counter):
    """Creates a Plotly donut chart for compliance status breakdown."""
    labels = list(status_counts.keys())
    values = list(status_counts.values())
    colors = [STATUS_COLORS.get(label, 'grey') for label in labels]

    fig = go.Figure(data=[go.Pie(labels=labels,
                                values=values,
                                hole=.4,
                                marker_colors=colors,
                                textinfo='percent+label',
                                hoverinfo='label+value+percent')])
    fig.update_layout(
        # title_text='Section Compliance Status',
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        height=250 # Adjust height
    )
    return fig

# --- Streamlit App Code ---

@st.cache_resource # Cache the pipeline resource across reruns
def get_pipeline():
    """Initializes and returns the QueryPipeline instance."""
    logger.info("Initializing QueryPipeline...")
    qdrant_url = None
    qdrant_api_key = None
    google_api_key_secret = None

    try:
        # Attempt to load from Streamlit secrets first
        logger.info(f"Checking for Streamlit secrets. hasattr(st, 'secrets'): {hasattr(st, 'secrets')}")
        if hasattr(st, 'secrets'):
            logger.info(f"st.secrets object type: {type(st.secrets)}")
            logger.info(f"Number of secrets found: {len(st.secrets) if st.secrets is not None else 'st.secrets is None'}")
            logger.info(f"Secret keys available: {list(st.secrets.keys()) if st.secrets is not None and hasattr(st.secrets, 'keys') else 'Cannot list keys'}")

        if hasattr(st, 'secrets') and len(st.secrets) > 0: # Check if secrets are available and populated
            logger.info("Attempting to load credentials from Streamlit secrets.")
            qdrant_url = st.secrets.get("QDRANT_URL")
            qdrant_api_key = st.secrets.get("QDRANT_API_KEY")
            google_api_key_secret = st.secrets.get("GOOGLE_API_KEY")
            logger.info(f"Loaded from st.secrets - QDRANT_URL: {'Found' if qdrant_url else 'Not Found'}, QDRANT_API_KEY: {'Found' if qdrant_api_key else 'Not Found'}, GOOGLE_API_KEY: {'Found' if google_api_key_secret else 'Not Found'}")
        else:
            logger.info("Streamlit secrets not found or empty, will proceed to check environment variables.")
    except Exception as e: # Catch any error during secrets access, including StreamlitSecretNotFoundError
        logger.warning(f"Could not load from Streamlit secrets (error: {e}), will proceed to check environment variables.")
        # Ensure variables are None if secrets access failed, so fallback logic triggers correctly
        qdrant_url = None
        qdrant_api_key = None
        google_api_key_secret = None

    # Fallback to environment variables if not found in secrets
    if not all([qdrant_url, qdrant_api_key, google_api_key_secret]):
        logger.info("One or more credentials not found via Streamlit secrets, attempting to load from .env or direct environment variables.")
        load_environment_variables() # Loads .env if it exists

        if not qdrant_url:
            qdrant_url = os.environ.get("QDRANT_URL")
            logger.info(f"QDRANT_URL from env: {'Found' if qdrant_url else 'Not Found'}")
        if not qdrant_api_key:
            qdrant_api_key = os.environ.get("QDRANT_API_KEY")
            logger.info(f"QDRANT_API_KEY from env: {'Found' if qdrant_api_key else 'Not Found'}")
        if not google_api_key_secret:
            google_api_key_secret = os.environ.get("GOOGLE_API_KEY")
            logger.info(f"GOOGLE_API_KEY from env: {'Found' if google_api_key_secret else 'Not Found'}")

    if not google_api_key_secret:
        st.warning("GOOGLE_API_KEY not found in Streamlit secrets or environment variables. LLM features may be limited or non-functional.")
        # Potentially fall back to a non-LLM mode or show a persistent error
    
    # Critical check for deployed environment (this part might need adjustment based on how IS_DEPLOYED is set)
    # For now, assuming it's for Streamlit Cloud deployment.
    # If running locally, we rely on the QDRANT_URL and QDRANT_API_KEY from env vars we are passing.
    is_deployed_env = os.environ.get("IS_DEPLOYED", "FALSE").upper() == "TRUE" # Example way to check if deployed
    if is_deployed_env: # Check if running in deployed environment
        logger.info("Detected deployed environment context.")
        if not qdrant_url:
            st.error("QDRANT_URL is not set in Streamlit secrets for deployed environment. App cannot connect to vector database.")
            return None
        if not qdrant_api_key:
            st.error("QDRANT_API_KEY is not set in Streamlit secrets for deployed environment. App cannot authenticate.")
            return None
    else:
        logger.info("Detected local development context (or IS_DEPLOYED is not TRUE).")
        # For local, ensure critical Qdrant vars are present (they should be, from our command line)
        if not qdrant_url:
            st.error("QDRANT_URL is not set in environment variables for local run. App cannot connect to vector database.")
            return None
        # API key might be optional for a local, unsecured Qdrant instance, but required for cloud.
        # The QdrantRetriever will handle this if api_key is None for a cloud URL.
        if "cloud.qdrant.io" in (qdrant_url or "") and not qdrant_api_key:
             st.warning("Connecting to Qdrant Cloud URL but QDRANT_API_KEY is not set. This will likely fail.")


    try:
        pipeline = QueryPipeline(
            qdrant_url=qdrant_url, 
            qdrant_api_key=qdrant_api_key, 
            google_api_key=google_api_key_secret
        )
        logger.info("QueryPipeline initialized successfully.")
        return pipeline
    except Exception as e:
        logger.error(f"Fatal error initializing QueryPipeline: {e}", exc_info=True)
        st.error(f"Fatal error initializing analysis pipeline: {e}. Please check logs and API key configuration.")
        return None # Indicate failure

# --- App Setup ---
st.set_page_config(
    page_title="Compliance Checker",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Compliance Checker")
# Add Welcome Message/Instructions
st.info("Welcome! Upload an internal policy document (PDF, DOCX, TXT, HTML) using the panel on the left. Adjust the analysis parameters in the sidebar if needed, then click 'Analyze Compliance'. The results will appear below.")

# --- Session State Initialization ---
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'last_analysis_params' not in st.session_state:
    st.session_state.last_analysis_params = {}
if 'filter_status' not in st.session_state:
    st.session_state.filter_status = "All"
if 'search_term' not in st.session_state:
    st.session_state.search_term = ""

# --- Sidebar for Configuration ---
st.sidebar.header("Analysis Configuration")
# model_choice = st.sidebar.selectbox("LLM Model", [DEFAULT_MODEL], disabled=True) # Model fixed for now
top_k = st.sidebar.slider("Top K Regulations per Chunk", 1, 15, DEFAULT_TOP_K, 1, help="How many relevant regulations to consider for each section of the document.")
score_threshold = st.sidebar.slider("Minimum Similarity Threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.05, help="How closely a regulation must match a document section to be considered relevant (0.0 = loose, 1.0 = strict).")
st.sidebar.caption("Adjust how strictly the analysis matches regulations.")

# Add Reset Button to Sidebar
st.sidebar.divider()
if st.sidebar.button("Clear Results / Start New", key="clear_button", use_container_width=True):
    st.session_state.analysis_results = None
    st.session_state.uploaded_file_name = None
    st.session_state.last_analysis_params = {}
    st.session_state.filter_status = "All"
    st.session_state.search_term = ""
    st.rerun()

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
        help="Upload your internal policy document (PDF, DOCX, TXT, HTML) for compliance analysis against RBI regulations.",
        label_visibility="collapsed" # Hide label as subheader is present
    )
    st.caption("Select the policy document you want to analyze.")

    analyze_button_disabled = uploaded_file is None
    analyze_button = st.button("Analyze Compliance", key="analyze_button", disabled=analyze_button_disabled, use_container_width=True)

    if uploaded_file is not None:
        st.info(f"Uploaded: **{uploaded_file.name}**")

        # Store filename if it changes
        if st.session_state.uploaded_file_name != uploaded_file.name:
             st.session_state.uploaded_file_name = uploaded_file.name
             st.session_state.analysis_results = None # Clear old results on new file
             st.session_state.last_analysis_params = {}
             st.session_state.filter_status = "All" # Reset filters on new file
             st.session_state.search_term = ""

with col2:
    st.subheader("Analysis Report")

    # Placeholder for the progress bar
    progress_bar_placeholder = st.empty()
    progress_text_placeholder = st.empty()

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

            # Replace spinner with progress bar
            progress_bar = progress_bar_placeholder.progress(0)
            progress_text = progress_text_placeholder.text("Starting analysis... 0%")

            # Define the callback function for progress updates
            def update_progress(value):
                # Ensure value is between 0 and 1
                value = max(0.0, min(1.0, value))
                percent_complete = int(value * 100)
                progress_bar.progress(value)
                # More specific progress text
                if value < 0.1:
                    progress_text.text(f"Preprocessing document... {percent_complete}%")
                elif value < 0.8:
                    progress_text.text(f"Analyzing sections... {percent_complete}%")
                else:
                    progress_text.text(f"Generating report... {percent_complete}%")

            try:
                # Run the analysis with selected parameters and the callback
                logger.info(f"Running analysis with top_k={top_k}, score_threshold={score_threshold}")
                analysis_results = pipeline.analyze_document(
                    tmp_file_path,
                    top_k=top_k,
                    score_threshold=score_threshold,
                    progress_callback=update_progress # Pass the callback here
                )
                st.session_state.analysis_results = analysis_results # Store results
                st.session_state.last_analysis_params = current_params # Store params used
                logger.info("Analysis complete.")
                # Clear progress bar after completion
                progress_bar_placeholder.empty()
                progress_text_placeholder.empty()

            except Exception as e:
                st.error(f"An unexpected error occurred during analysis: {e}")
                logger.error(f"Analysis button error: {e}", exc_info=True)
                st.session_state.analysis_results = {"error": str(e)} # Store error
                # Clear progress bar even on error
                progress_bar_placeholder.empty()
                progress_text_placeholder.empty()
            finally:
                # Clean up the temporary file
                if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
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
        processed_sections = [] # Store processed data for filtering

        # Check for errors stored in session state
        if isinstance(results, dict) and "error" in results:
            st.error(f"Analysis failed: {results['error']}")
        elif not results or not results.get("section_analyses"):
            st.warning("Analysis completed, but no results were generated or the format is unexpected.")
        else:
            # Display Overall Header
            st.header("Compliance Analysis Report")
            params = results.get('parameters', {})
            doc_name = st.session_state.uploaded_file_name or results.get('document_path', 'N/A')
            st.caption(f"Document: `{doc_name}` | Top K Regulations: {params.get('top_k', 'N/A')} | Similarity Threshold: {params.get('score_threshold', 'N/A')}")

            # --- Process Sections --- 
            section_analyses = results.get("section_analyses", [])
            total_sections = len(section_analyses)

            for i, section_data in enumerate(section_analyses):
                analysis_text = section_data.get("analysis_text", "Analysis not available.")
                parsed_analysis = parse_llm_analysis(analysis_text, i)

                processed_sections.append({
                    "original_index": i,
                    "section_title": parsed_analysis["llm_section_title"],
                    "compliance_status": parsed_analysis["compliance_status"],
                    "compliance_pct": parsed_analysis["compliance_pct"],
                    "compliance_verdict": parsed_analysis["compliance_verdict"],
                    "analysis_text": parsed_analysis["raw_analysis"], # Store raw for display
                    "section_text": section_data.get("section_text", ""), # Use correct key 'section_text'
                    "referenced_regulations": parsed_analysis["referenced_regulations"], # Correct source and add key
                    "detailed_assessment": parsed_analysis["detailed_assessment"], # Add missing key
                    "action_items": parsed_analysis["action_items"],
                    "metrics": parsed_analysis["metrics"],
                    "severity_counts": parsed_analysis["severity_counts"]
                })

            # --- Calculate Overall Summary --- 
            summary_data = calculate_overall_summary(processed_sections)

            # --- Display Compliance Dashboard --- 
            st.subheader("Compliance Dashboard")

            # Key Findings Summary (Non-Compliant Sections)
            if summary_data["non_compliant_sections"]:
                st.error(f"**Key Findings:** {len(summary_data['non_compliant_sections'])} section(s) identified as **Non-Compliant**:\n" +
                         "\n".join([f"- {title}" for title in summary_data['non_compliant_sections'][:5]]) +
                         ("\n- ... (see details below)" if len(summary_data['non_compliant_sections']) > 5 else ""))

            dash_col1, dash_col2, dash_col3 = st.columns([1, 1, 1]) # Use 3 columns for dashboard

            with dash_col1:
                st.metric("Overall Compliance Score", f"{summary_data['avg_compliance_pct']}%", delta=None) # No delta needed here
                st.progress(summary_data['avg_compliance_pct'] / 100)

                st.metric("Total Sections Analyzed", summary_data['total_sections'])

            with dash_col2:
                st.markdown("**Section Status Breakdown**")
                if summary_data["section_status_counts"]:
                    fig = create_compliance_donut_chart(summary_data["section_status_counts"])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("No section data for chart.")

            with dash_col3:
                st.markdown("**Action Item Severity**")
                # Use metrics for severity counts with color
                st.metric("🔴 High Severity Items", summary_data['total_severity_counts']['High'])
                st.metric("🟠 Medium Severity Items", summary_data['total_severity_counts']['Medium'])
                st.metric("🟡 Low Severity Items", summary_data['total_severity_counts']['Low'])
                # Highlight Non-Compliant count in red
                non_compliant_count = summary_data['section_status_counts']['Non-Compliant']
                st.metric("🔴 Non-Compliant Sections", non_compliant_count)

            # Display Prioritized Action Items (moved below dashboard columns)
            st.markdown("**Prioritized Action Items (Top 5)**")
            if not summary_data["all_action_items"]:
                 st.success("✅ No action items required based on this assessment.")
            else:
                actions_to_show = summary_data["all_action_items"][:5]
                for i, action in enumerate(actions_to_show):
                    severity_icon = {"High": "🔴", "Medium": "🟠", "Low": "🟡"}.get(action["severity"], "⚪")
                    st.markdown(f"{i+1}. {severity_icon} **[{action['severity']}]** {action['policy_aspect']} (Section: *{action['section_title']}*)")
                    st.markdown(f"   ↳ Recommendation: *{action['recommendation']}*")
                    # Removed the divider between items for cleaner look
                if len(summary_data["all_action_items"]) > len(actions_to_show):
                    st.caption(f"... and {len(summary_data['all_action_items']) - len(actions_to_show)} more action items in the detailed sections below.")

            st.divider()

            # --- Filtering and Search for Detailed Results ---
            st.subheader("Section Analysis Details")

            filter_col1, filter_col2 = st.columns([1, 2])
            with filter_col1:
                filter_status = st.selectbox(
                    "Filter by Compliance Status",
                    options=["All", "Compliant", "Partially Compliant", "Non-Compliant", "Unable to Assess"],
                    key='filter_status',
                    help="Show only sections with the selected compliance status."
                )
            with filter_col2:
                search_term = st.text_input(
                    "Search Section Titles",
                    key='search_term',
                    placeholder="Enter keywords to search section titles...",
                    help="Filter sections by keywords in their title."
                ).lower()

            # Status Legend
            st.markdown(
                "**Legend:** <span style='color:green;'>🟢 Compliant</span> | <span style='color:orange;'>🟡 Partially Compliant</span> | <span style='color:red;'>🔴 Non-Compliant</span> | ⚪ Unable to Assess",
                unsafe_allow_html=True
            )

            # Filter the processed sections
            filtered_sections = []
            for section in processed_sections:
                status_match = (filter_status == "All" or section["compliance_status"] == filter_status)
                search_match = (search_term == "" or search_term in section["section_title"].lower())
                if status_match and search_match:
                    filtered_sections.append(section)

            # --- Display Filtered Section Details ---
            if not filtered_sections:
                st.warning(f"No sections match the current filter criteria (Status: {filter_status}, Search: '{search_term}').")
            else:
                st.caption(f"Displaying {len(filtered_sections)} of {total_sections} sections.")
                for section in filtered_sections:
                    # Revert to direct access for title - keys should be present
                    status_icon = STATUS_ICONS.get(section["compliance_status"], "⚪")
                    expander_title = f"{status_icon} {section['compliance_status']} ({section['compliance_verdict']}): {section['section_title']} ({section['compliance_pct']}% Compliance)"
                    # Keep unique key based on index, but rename for clarity
                    policy_text_key = f"policy_text_{section['original_index']}" 

                    with st.expander(expander_title):
                        # Use columns for better layout inside expander
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**1. Policy Section Text:**")
                            # Use the correct key 'section_text' here
                            st.text_area("Policy Text", section.get('section_text', "Policy text not available."), height=200, disabled=True, label_visibility="collapsed", key=policy_text_key) 

                        with col2:
                            st.markdown("**2. AI Compliance Summary:**")
                            # Revert to direct access - keys should be present
                            st.markdown(f"""
                            - **Overall Compliance:** {section['compliance_pct']}% ({section['compliance_verdict']})
                            - **Compliant Items:** {section['metrics']['compliant_items']}
                            - **Partially Compliant Items:** {section['metrics']['partially_compliant_items']}
                            - **Non-Compliant Items:** {section['metrics']['noncompliant_items']}
                            - **Unable to Assess:** {section['metrics']['unable_to_assess_items']}
                            """)

                            # Display Action Items for this section, revert to direct access
                            if section['action_items']:
                                st.markdown("**Action Items:**")
                                for item in section['action_items']:
                                    # Direct access within loop - item dict structure assumed consistent
                                    sev_icon = {"High": "🔴", "Medium": "🟠", "Low": "🟡"}.get(item["severity"], "⚪")
                                    st.markdown(f" - {sev_icon} **[{item['severity']}]** {item['policy_aspect']}: *{item['recommendation']}*")
                            else:
                                st.markdown("**Action Items:** None")

                        st.markdown("---") # Divider below columns

                        # Display Detailed Assessment and Regulations below columns
                        st.markdown("**3. Detailed Assessment:**")
                        # Revert to direct access - key should be present
                        if section['detailed_assessment']:
                            # Use markdown for better formatting of the detailed assessment
                            st.markdown(section['detailed_assessment'])
                        else:
                            st.caption("Detailed assessment breakdown not available.")

                        st.markdown("---")

                        st.markdown("**4. Referenced Regulations:**")
                        # Revert to direct access - key should be present
                        if section['referenced_regulations'] and 'No specific regulations' not in section['referenced_regulations']:
                             # Use markdown for better formatting of regulations list
                            st.markdown(section['referenced_regulations'])
                        else:
                            st.markdown("*No specific regulations were found highly relevant to this section based on the current threshold or LLM output.*")

                        # Optionally show raw LLM output for debugging
                        # with st.popover("Show Raw LLM Output"):
                        #    st.text_area("Raw LLM Analysis", section['raw_analysis'], height=200, disabled=True)

    elif st.session_state.uploaded_file_name:
        # If a file was uploaded but no results yet (e.g., after clearing)
        st.info(f"Ready to analyze **{st.session_state.uploaded_file_name}**. Click 'Analyze Compliance' on the left.")
    else:
        # Initial state before any upload
        st.info("Upload a document using the panel on the left to begin the compliance analysis.")
