import os
import json
import time
import uuid
import datetime
import argparse
import re
import logging
from typing import List, Dict, Any, Optional, Union
from pydantic_settings import BaseSettings
import google.generativeai as genai

# --- Configuration Management ---
class Settings(BaseSettings):
    gemini_api_key: str = os.environ.get("GEMINI_API_KEY", "")
    log_level: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()

# --- Logging Setup ---
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Custom Exceptions ---
class GeminiAgentError(Exception):
    """Base exception for Gemini Agent"""
    pass

class APIError(GeminiAgentError):
    """Exception raised during API calls"""
    pass

class FileSystemError(GeminiAgentError):
    """Exception raised during file operations"""
    pass

# --- Core Logic ---
class GeminiCodeAgent:
    """
    Enhanced Code Generation and Optimization System based on Gemini.
    Handles code generation, file management, and structured logging.
    """
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        if not settings.gemini_api_key:
            raise APIError("GEMINI_API_KEY is not set in environment variables.")
        
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(model_name)
        logger.info(f"GeminiCodeAgent initialized with model: {model_name}")

    def _extract_code(self, text: str) -> str:
        """Extracts Python code from Markdown code blocks."""
        pattern = r"```python\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        return text # Fallback if no blocks found

    def generate_code(self, prompt: str) -> str:
        """
        Generates code based on the provided prompt.
        
        Args:
            prompt (str): The task description.
            
        Returns:
            str: The generated code.
        """
        try:
            logger.info("Requesting code generation from Gemini...")
            response = self.model.generate_content(prompt)
            if not response.text:
                raise APIError("Empty response received from Gemini API.")
            
            code = self._extract_code(response.text)
            return code
        except Exception as e:
            logger.error(f"Error during code generation: {str(e)}")
            raise APIError(f"Failed to generate code: {e}")

    def save_to_disk(self, code: str, filename: str) -> str:
        """
        Saves the generated code to a random directory to avoid collisions.
        
        Args:
            code (str): The code content to save.
            filename (str): The name of the file.
            
        Returns:
            str: The absolute path to the saved file.
        """
        try:
            session_id = str(uuid.uuid4())[:8]
            folder_name = f"gen_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{session_id}"
            folder_path = os.path.join(os.getcwd(), folder_name)
            os.makedirs(folder_path, exist_ok=True)
            
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
            
            logger.info(f"Code successfully saved to {file_path}")
            return file_path
        except IOError as e:
            logger.error(f"FileSystem error while saving code: {str(e)}")
            raise FileSystemError(f"Could not save file to disk: {e}")

    def run_task(self, task_name: str, prompt: str) -> Dict[str, Any]:
        """
        Executes a full cycle: Generate -> Save -> Log.
        """
        logger.info(f"Starting task: {task_name}")
        try:
            code = self.generate_code(prompt)
            filename = f"{task_name}.py"
            path = self.save_to_disk(code, filename)
            
            return {
                "status": "success",
                "task": task_name,
                "file_path": path,
                "timestamp": datetime.datetime.now().isoformat()
            }
        except GeminiAgentError as e:
            logger.error(f"Task {task_name} failed: {str(e)}")
            return {"status": "error", "task": task_name, "error": str(e)}

# --- Main Execution ---
if __name__ == "__main__":
    # Example Predefined Tasks
    PREDEFINED_TASKS = {
        "data_analyzer": "Create a Python class DataAnalyzer to read CSV and provide summary statistics.",
        "web_scraper": "Create a Python web scraper using requests and BeautifulSoup with error handling."
    }

    parser = argparse.ArgumentParser(description="Gemini Code Generation Agent")
    parser.add_argument("--task", type=str, choices=list(PREDEFINED_TASKS.keys()), help="Task to execute")
    args = parser.parse_args()

    agent = GeminiCodeAgent()
    
    if args.task:
        result = agent.run_task(args.task, PREDEFINED_TASKS[args.task])
        print(json.dumps(result, indent=2))
    else:
        print("Please specify a task using --task <<tasktask_name>")
