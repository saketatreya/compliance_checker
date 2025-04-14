import streamlit as st
from pathlib import Path
import tempfile
import logging
import os
import re

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
                progress_text.text(f"Analyzing document... {percent_complete}%")

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
        skipped_sections_count = 0
        total_sections = 0
        sections_displayed = 0
        
        # Collection for storing summary data
        overall_compliance_data = {
            "total_compliance_pct": 0,
            "policy_aspects_assessed": 0,
            "compliant_items": 0,
            "partially_compliant_items": 0,
            "noncompliant_items": 0,
            "unable_to_assess_items": 0,
            "high_severity_items": 0,
            "medium_severity_items": 0,
            "low_severity_items": 0,
            "all_action_items": []
        }

        # Check for errors stored in session state
        if isinstance(results, dict) and "error" in results:
            st.error(f"Analysis failed: {results['error']}")
        elif not results or not results.get("section_analyses"):
            st.warning("Analysis completed, but no results were generated or the format is unexpected.")
        else:
            # Display Overall Header
            st.header("Compliance Analysis Report")
            params = results.get('parameters', {})
            st.caption(f"Document: `{results.get('document_path', 'N/A')}` | Top K Regulations: {params.get('top_k', 'N/A')} | Similarity Threshold: {params.get('score_threshold', 'N/A')}")
            
            # Iterate through sections and collect summary data first
            section_analyses = results.get("section_analyses", [])
            total_sections = len(section_analyses)
            
            for i, section_data in enumerate(section_analyses):
                section_title = section_data.get("section_title", f"Section {i+1}")
                analysis_text = section_data.get("analysis_text", "Analysis not available.")
                
                # Extract LLM-identified section title
                llm_section_title_match = re.search(r"- Section Title: \[(.*?)\]", analysis_text)
                if llm_section_title_match:
                    llm_section_title = llm_section_title_match.group(1).strip()
                    
                    # Post-process the section title to clean up any formatting issues
                    # 1. Remove any section numbers or letters at the beginning (A., 1., etc.)
                    llm_section_title = re.sub(r'^[A-Za-z0-9]+\.\s*', '', llm_section_title)
                    
                    # 2. If title is too long (more than 50 chars), it's likely a sentence - truncate and add ellipsis
                    if len(llm_section_title) > 50:
                        # Try to find a natural breaking point (after 4-5 words)
                        words = llm_section_title.split()
                        if len(words) > 5:
                            llm_section_title = ' '.join(words[:5])
                        else:
                            llm_section_title = llm_section_title[:50]
                    
                    # 3. Convert to title case
                    llm_section_title = llm_section_title.title()
                    
                    # 4. If it ends with a period or other punctuation, remove it
                    llm_section_title = re.sub(r'[.,;:!?]$', '', llm_section_title)
                    
                    # 5. Extract the main topic using NLP-like heuristics
                    # If it contains "shall", "must", "will", it's likely a policy statement, not a title
                    # Extract just the subject and object
                    policy_verbs = ['shall', 'must', 'will', 'should', 'may', 'can', 'report', 'identify', 'verify']
                    for verb in policy_verbs:
                        if verb in llm_section_title.lower():
                            # Find what the policy is about by looking at key nouns around the verb
                            parts = llm_section_title.lower().split(verb)
                            if len(parts) > 1:
                                # Look for banking terms in the statement
                                key_terms = ['transaction', 'customer', 'account', 'risk', 'report', 
                                            'currency', 'kyc', 'aml', 'verification', 'due diligence',
                                            'monitoring', 'identification', 'document', 'policy', 'procedure',
                                            'compliance', 'suspicious', 'counterfeit', 'activity']
                                
                                found_terms = []
                                for term in key_terms:
                                    if term in llm_section_title.lower():
                                        found_terms.append(term.title())
                                
                                if found_terms:
                                    if len(found_terms) > 2:
                                        found_terms = found_terms[:2]  # Limit to 2 terms
                                    llm_section_title = " ".join(found_terms) + " Requirements"
                                    break
                                else:
                                    # Generic fallback - grab some context words
                                    context_nouns = parts[1].strip().split()[:2]
                                    if context_nouns:
                                        llm_section_title = " ".join([word.title() for word in context_nouns]) + " Policy"
                                    else:
                                        llm_section_title = "Policy Requirements"
                                    break
                    
                    # Store the processed title
                    section_data["llm_section_title"] = llm_section_title
                
                # Extract compliance percentage from LLM response using regex
                compliance_pct_match = re.search(r"Overall Compliance: (\d+)%", analysis_text)
                compliance_pct = int(compliance_pct_match.group(1)) if compliance_pct_match else 0
                
                # Extract compliance metrics using regex
                aspects_assessed_match = re.search(r"Number of Policy Aspects Assessed: (\d+)", analysis_text)
                compliant_items_match = re.search(r"Compliant Items: (\d+) \((\d+)%\)", analysis_text)
                partially_compliant_match = re.search(r"Partially Compliant Items: (\d+) \((\d+)%\)", analysis_text)
                noncompliant_match = re.search(r"Non-Compliant Items: (\d+) \((\d+)%\)", analysis_text)
                unable_to_assess_match = re.search(r"Items Unable to Assess: (\d+)", analysis_text)
                
                # Extract action items
                action_items_section = re.search(r"\*\*2\. Prioritized Action Items:\*\*\s*(.*?)(?=\*\*3\. Detailed Assessment)", analysis_text, re.DOTALL)
                
                if action_items_section:
                    action_items_text = action_items_section.group(1).strip()
                    
                    # If there are action items, extract them with severity
                    if "No action items required" not in action_items_text:
                        # Look for the numbered items with severity
                        action_item_matches = re.finditer(r"(\d+)\. \[(High|Medium|Low) Severity\] ([^:]+): ([^\n]+)", action_items_text)
                        
                        for match in action_item_matches:
                            severity = match.group(2)
                            policy_aspect = match.group(3).strip()
                            recommendation = match.group(4).strip()
                            
                            # Track severities
                            if severity == "High":
                                overall_compliance_data["high_severity_items"] += 1
                            elif severity == "Medium":
                                overall_compliance_data["medium_severity_items"] += 1
                            elif severity == "Low":
                                overall_compliance_data["low_severity_items"] += 1
                                
                            # Store action item with section data
                            overall_compliance_data["all_action_items"].append({
                                "section_title": section_title,
                                "severity": severity,
                                "policy_aspect": policy_aspect,
                                "recommendation": recommendation
                            })
                
                # Aggregate the compliance data
                if compliance_pct > 0:
                    overall_compliance_data["total_compliance_pct"] += compliance_pct
                
                if aspects_assessed_match:
                    overall_compliance_data["policy_aspects_assessed"] += int(aspects_assessed_match.group(1))
                
                if compliant_items_match:
                    overall_compliance_data["compliant_items"] += int(compliant_items_match.group(1))
                
                if partially_compliant_match:
                    overall_compliance_data["partially_compliant_items"] += int(partially_compliant_match.group(1))
                
                if noncompliant_match:
                    overall_compliance_data["noncompliant_items"] += int(noncompliant_match.group(1))
                
                if unable_to_assess_match:
                    overall_compliance_data["unable_to_assess_items"] += int(unable_to_assess_match.group(1))
                
                # Check if section should be skipped (updating criteria for new format)
                is_fully_compliant = compliance_pct == 100 if compliance_pct_match else False
                no_action_items = "No action items required" in analysis_text if action_items_section else False
                
                if is_fully_compliant and no_action_items:
                    skipped_sections_count += 1
                else:
                    sections_displayed += 1
            
            # Calculate overall average compliance percentage
            if total_sections > 0:
                overall_compliance_data["total_compliance_pct"] = round(overall_compliance_data["total_compliance_pct"] / total_sections)
            
            # --- Display Compliance Dashboard ---
            st.subheader("Compliance Dashboard")
            
            # Create dashboard columns
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # Show compliance gauge with color coding
                compliance_pct = overall_compliance_data["total_compliance_pct"]
                gauge_color = "#ff0000"  # Red for low compliance
                if compliance_pct >= 80:
                    gauge_color = "#00ff00"  # Green for high compliance
                elif compliance_pct >= 50:
                    gauge_color = "#ffff00"  # Yellow for medium compliance
                
                # Using a progress bar as a simple gauge
                st.metric("Overall Compliance", f"{compliance_pct}%")
                st.progress(compliance_pct/100)
                
                # Display counts
                st.markdown("### Compliance Breakdown")
                st.markdown(f"🟢 **Compliant Items:** {overall_compliance_data['compliant_items']}")
                st.markdown(f"🟡 **Partially Compliant:** {overall_compliance_data['partially_compliant_items']}")
                st.markdown(f"🔴 **Non-Compliant:** {overall_compliance_data['noncompliant_items']}")
                st.markdown(f"⚪ **Unable to Assess:** {overall_compliance_data['unable_to_assess_items']}")
            
            with col2:
                # Display prioritized action items
                st.markdown("### Prioritized Action Items")
                
                if not overall_compliance_data["all_action_items"]:
                    st.success("No action items required - policy is fully compliant with available regulations.")
                else:
                    # Sort action items by severity (High, Medium, Low)
                    severity_order = {"High": 0, "Medium": 1, "Low": 2}
                    sorted_actions = sorted(
                        overall_compliance_data["all_action_items"], 
                        key=lambda x: severity_order.get(x["severity"], 3)
                    )
                    
                    # Create table of action items
                    for i, action in enumerate(sorted_actions):
                        severity_color = {
                            "High": "🔴",
                            "Medium": "🟠",
                            "Low": "🟡"
                        }.get(action["severity"], "⚪")
                        
                        st.markdown(f"{i+1}. {severity_color} **[{action['severity']}]** {action['policy_aspect']}")
                        st.markdown(f"   *{action['recommendation']}*")
                        st.markdown(f"   *Section: {action['section_title']}*")
                        if i < len(sorted_actions) - 1:
                            st.markdown("---")
            
            st.divider()
            
            # --- Display Section Details with Tabs ---
            st.subheader("Section Analysis Details")
            
            if sections_displayed == 0:
                if skipped_sections_count == total_sections and total_sections > 0:
                    st.success("**All sections were found to be fully compliant with the retrieved regulations, and no specific recommendations were necessary based on the provided context.**")
                else:
                    st.warning("Analysis completed, but no sections required detailed reporting based on the criteria.")
            else:
                # Create tabs for each section to be displayed
                displayed_sections = [section_data for i, section_data in enumerate(section_analyses) 
                                     if not (
                                         re.search(r"Overall Compliance: 100%", section_data.get("analysis_text", "")) and 
                                         "No action items required" in section_data.get("analysis_text", "")
                                     )]
                
                # Create tab labels
                tab_labels = [f"Section {i+1}: {s.get('llm_section_title', s.get('section_title', 'Unnamed'))[:20]}..." 
                             for i, s in enumerate(displayed_sections)]
                
                # If we have sections to display, create tabs
                if tab_labels:
                    tabs = st.tabs(tab_labels)
                    
                    # Fill content for each tab
                    for i, (tab, section_data) in enumerate(zip(tabs, displayed_sections)):
                        with tab:
                            section_title = section_data.get("llm_section_title", section_data.get("section_title", f"Unnamed Section {i+1}"))
                            analysis_text = section_data.get("analysis_text", "Analysis not available.")
                            retrieved_regs = section_data.get("retrieved_regulations", [])
                            section_text = section_data.get("section_text", "")
                            section_text_preview = section_text[:500]  # Show more preview
                            
                            st.markdown(f"### {section_title}")
                            
                            with st.expander("Original Policy Text (Preview)"):
                                st.text_area("", section_text_preview + ("..." if len(section_text) > 500 else ""), 
                                            height=150, disabled=True, label_visibility="collapsed", 
                                            key=f"text_exp_{i}")
                            
                            if retrieved_regs:
                                with st.expander("Retrieved Regulations"):
                                    reg_list_items = []
                                    for reg in retrieved_regs:
                                        metadata = reg.get('metadata', {})
                                        source_name = metadata.get('file_name') or \
                                                    metadata.get('document_name') or \
                                                    metadata.get('reference_id') or \
                                                    metadata.get('source', 'Unknown Source')
                                        page_num = metadata.get('page_number', 'N/A')
                                        text_preview = reg.get('text', '')[:300]
                                        reg_list_items.append(f"- **[{source_name}, Page {page_num}]**: `{text_preview}...`")
                                    
                                    reg_markdown = "\n".join(reg_list_items)
                                    st.markdown(reg_markdown, unsafe_allow_html=True)
                            
                            # Split analysis into tabs for the different sections
                            analysis_tabs = st.tabs(["Compliance Summary", "Detailed Assessment", 
                                                    "Areas of Uncertainty", "Recommendations"])
                            
                            # Extract sections from analysis_text
                            summary_match = re.search(r"\*\*1\. Compliance Summary:\*\*(.*?)(?=\*\*2\. Prioritized Action Items)", 
                                                    analysis_text, re.DOTALL)
                            action_items_match = re.search(r"\*\*2\. Prioritized Action Items:\*\*(.*?)(?=\*\*3\. Detailed Assessment)", 
                                                        analysis_text, re.DOTALL)
                            detailed_match = re.search(r"\*\*3\. Detailed Assessment:\*\*(.*?)(?=\*\*4\. Areas of Uncertainty)", 
                                                    analysis_text, re.DOTALL)
                            uncertainty_match = re.search(r"\*\*4\. Areas of Uncertainty:\*\*(.*?)(?=\*\*5\. Actionable Recommendations)", 
                                                        analysis_text, re.DOTALL)
                            recommendations_match = re.search(r"\*\*5\. Actionable Recommendations:\*\*(.*)", 
                                                            analysis_text, re.DOTALL)
                            
                            # Put content in each tab
                            with analysis_tabs[0]:  # Compliance Summary
                                if summary_match:
                                    st.markdown("**Compliance Summary:**" + summary_match.group(1))
                                    
                                    if action_items_match:
                                        st.markdown("**Prioritized Action Items:**" + action_items_match.group(1))
                                else:
                                    st.markdown(analysis_text)  # Fallback to full text
                            
                            with analysis_tabs[1]:  # Detailed Assessment
                                if detailed_match:
                                    st.markdown("**Detailed Assessment:**" + detailed_match.group(1))
                                else:
                                    st.markdown(analysis_text)  # Fallback
                            
                            with analysis_tabs[2]:  # Areas of Uncertainty
                                if uncertainty_match:
                                    st.markdown("**Areas of Uncertainty:**" + uncertainty_match.group(1))
                                else:
                                    st.markdown(analysis_text)  # Fallback
                            
                            with analysis_tabs[3]:  # Recommendations
                                if recommendations_match:
                                    st.markdown("**Actionable Recommendations:**" + recommendations_match.group(1))
                                else:
                                    st.markdown(analysis_text)  # Fallback
                
                # Display summary message about skipped sections
                if skipped_sections_count > 0:
                    st.info(f"*Note: {skipped_sections_count} section(s) were omitted from this report as they were fully compliant with no recommendations.*")
                elif total_sections == 0:
                    st.warning("Analysis completed, but no sections were found or analyzed in the document.")