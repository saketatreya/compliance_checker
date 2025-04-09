import os
import logging
from pathlib import Path
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)

def load_environment_variables(env_file: Optional[str] = None) -> bool:
    """
    Load environment variables from .env file.
    
    Args:
        env_file: Path to the .env file (defaults to project root .env)
        
    Returns:
        True if environment variables were loaded successfully, False otherwise
    """
    try:
        from dotenv import load_dotenv
        
        # If no env_file is specified, look for .env in the project root
        if env_file is None:
            # Get the project root directory (assuming this file is in src/utils)
            project_root = Path(__file__).parent.parent.parent
            env_file = project_root / ".env"
        
        # Load environment variables from .env file
        success = load_dotenv(env_file)
        
        if success:
            logger.info(f"Loaded environment variables from {env_file}")
            # Check if critical variables are loaded
            if "GOOGLE_API_KEY" in os.environ:
                logger.info("GOOGLE_API_KEY environment variable is set")
            else:
                logger.warning("GOOGLE_API_KEY environment variable is not set in .env file")
        else:
            logger.warning(f"Failed to load environment variables from {env_file}")
        
        return success
    except ImportError:
        logger.error("python-dotenv package not installed. Install with: pip install python-dotenv")
        return False
    except Exception as e:
        logger.error(f"Error loading environment variables: {e}")
        return False