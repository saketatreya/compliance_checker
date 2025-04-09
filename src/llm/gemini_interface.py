import os
import logging
from typing import Dict, List, Optional, Union, Any

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document

# Import the environment loader
from src.utils.env_loader import load_environment_variables

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GeminiInterface:
    """Interface for Google's Gemini LLM.
    
    This class handles the integration with Gemini for generating
    compliance insights based on retrieved regulatory context.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-pro"):
        """Initialize the Gemini interface.
        
        Args:
            api_key: Google API key for Gemini (defaults to GOOGLE_API_KEY env var)
            model_name: Gemini model name to use
        """
        self.model_name = model_name
        
        # Load environment variables from .env file first
        load_environment_variables()
        
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Google API key not provided and GOOGLE_API_KEY environment variable not set")
        
        # Configure Gemini API
        genai.configure(api_key=api_key)
        
        # Initialize LangChain integration
        self.llm = GoogleGenerativeAI(model=model_name, google_api_key=api_key)
        
        # Define prompt templates
        self.compliance_prompt = PromptTemplate(
            input_variables=["context", "document"],
            template="""You are an expert AI assistant specialized in RBI regulations and banking compliance. 
            You will be given excerpts from RBI regulatory documents and a section of a bank's internal policy. 
            Your job is to assess whether the policy section complies with the regulations, point out any compliance issues, 
            and cite the relevant RBI rules.

            When you refer to an RBI regulation, include the document reference or clause number from the provided context.
            If a relevant regulation is not provided in context, say you are not certain, rather than guessing.
            
            Context:
            RBI Regulations:
            {context}
            
            Bank's Policy Section:
            {document}
            
            Identify any compliance issues in the bank's policy section with respect to RBI regulations. 
            Cite the relevant RBI guidelines in your answer. If no issues are found, confirm compliance and cite the relevant guidelines.
            
            Format your response as follows:
            1. Summary: A brief overview of your compliance assessment
            2. Issues: List each compliance issue with citation (if any)
            3. Recommendations: Suggested changes to ensure compliance
            """
        )
        
        # Create LLMChain
        self.compliance_chain = LLMChain(
            llm=self.llm,
            prompt=self.compliance_prompt,
            verbose=True
        )
        
        logger.info(f"Initialized Gemini interface with model: {model_name}")
    
    def format_retrieved_context(self, retrieved_docs: List[Dict]) -> str:
        """Format retrieved documents into a context string.
        
        Args:
            retrieved_docs: List of retrieved documents from Qdrant
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            # Extract metadata
            metadata = doc.get("metadata", {})
            reference_id = metadata.get("reference_id", "Unknown Reference")
            section = metadata.get("section", "")
            clause_id = metadata.get("clause_id", "")
            
            # Format citation
            citation = f"RBI {reference_id}"
            if section:
                citation += f", {section}"
            if clause_id:
                citation += f", Clause {clause_id}"
            
            # Format document text with citation
            doc_text = f"{i}. \"{doc['text']}\" ({citation})"
            context_parts.append(doc_text)
        
        return "\n\n".join(context_parts)
    
    def analyze_compliance(self, document_text: str, retrieved_docs: List[Dict]) -> str:
        """Analyze compliance of a document against retrieved regulatory context.
        
        Args:
            document_text: Text of the document to analyze
            retrieved_docs: List of retrieved regulatory documents
            
        Returns:
            Compliance analysis as text
        """
        logger.info("Analyzing compliance with Gemini")
        
        # Format context from retrieved documents
        context = self.format_retrieved_context(retrieved_docs)
        
        # Run the compliance chain
        result = self.compliance_chain.run({
            "context": context,
            "document": document_text
        })
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Gemini model.
        
        Returns:
            Dictionary containing model information
        """
        try:
            model_info = genai.get_model(self.model_name)
            return {
                "name": model_info.name,
                "display_name": model_info.display_name,
                "description": model_info.description,
                "input_token_limit": model_info.input_token_limit,
                "output_token_limit": model_info.output_token_limit,
                "supported_generation_methods": model_info.supported_generation_methods,
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # Example usage (requires API key)
    try:
        gemini = GeminiInterface()
        
        # Get model info
        model_info = gemini.get_model_info()
        print(f"Model info: {model_info}")
        
        # Example compliance analysis
        document_text = "The total credit exposure to any single MSME client shall not exceed ₹60 crore."
        
        # Mock retrieved documents
        retrieved_docs = [
            {
                "text": "the aggregate exposure of all lending institutions to the MSME borrower should not exceed ₹25 crore as on March 31, 2021.",
                "metadata": {
                    "reference_id": "Circular DOR.STR.REC.12/21.04.048/2021-22",
                    "clause_id": "2(iii)"
                }
            },
            {
                "text": "it has been decided to enhance the above limit from ₹25 crore to ₹50 crore.",
                "metadata": {
                    "reference_id": "Circular DOR.STR.REC.21/21.04.048/2021-22",
                    "clause_id": "3"
                }
            }
        ]
        
        # Analyze compliance
        analysis = gemini.analyze_compliance(document_text, retrieved_docs)
        print("\nCompliance Analysis:")
        print(analysis)
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set the GOOGLE_API_KEY environment variable or provide an API key.")