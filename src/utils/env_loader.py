import os
import logging
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)

def load_environment_variables(env_path: Optional[str] = None):
    """Loads environment variables from a .env file.

    Args:
        env_path (Optional[str]): Path to the .env file. If None, searches 
                                   in the current directory and parent directories.
    """
    # Find the .env file automatically if no path is specified
    # This searches the current directory and parent directories
    dotenv_path = find_dotenv() if env_path is None else Path(env_path)
    
    if dotenv_path and dotenv_path.exists():
        logger.info(f"Loading environment variables from: {dotenv_path}")
        load_dotenv(dotenv_path=dotenv_path, override=True)
    elif env_path:
        logger.warning(f".env file specified at {env_path} but not found.")
    else:
        logger.info(".env file not found in project structure, relying on system environment variables.")

def find_dotenv(filename='.env', raise_error_if_not_found=False, usecwd=False) -> Optional[Path]:
    """Search in increasingly higher folders for the `filename`.

    Args:
        filename (str): the name of the file to find.
        raise_error_if_not_found (bool): raise an exception if the file is not found.
        usecwd (bool): use the current working directory as the starting directory.

    Returns:
        Optional[Path]: Path to the file if found, else None.
    """
    if usecwd or '__file__' not in globals():
        # should work without __file__, e.g. in REPL or IPython notebook
        path = Path.cwd()
    else:
        path = Path(__file__).resolve().parent

    for _ in range(100):  # Limit search depth
        if (path / filename).is_file():
            return path / filename
        if path.parent == path:  # Root directory
            break
        path = path.parent

    if raise_error_if_not_found:
        raise IOError('File not found')

    return None

# Example of how to ensure keys are loaded when this module is imported
# load_environment_variables() 
# Uncomment above line if you want keys loaded automatically upon import,
# otherwise call load_environment_variables() explicitly in your app entry point.