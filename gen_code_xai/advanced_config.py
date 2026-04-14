"""
AutoGen XAI Code Generation and Evaluation System - Advanced Configuration
=======================================
This script demonstrates how to customize and configure the code generation evaluation system.
"""

import os
import json
import time
import requests
import re
import autogen
from dotenv import load_dotenv
import logging
from typing import Dict, List, Any, Optional, Tuple
import datetime
import argparse
from pathlib import Path

# Import base modules from enhanced version
try:
    from enhanced_code_generation_evaluation import (
        XAIAgent,
        extract_files_from_response,
        generate_project_dir_name,
        sanitize_filename,
        logger,
    )
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import from enhanced_code_generation_evaluation: {e}")
    # Define fallback placeholders to avoid crash
    class XAIAgent: pass
    def extract_files_from_response(*args, **kwargs): return {}
    def generate_project_dir_name(*args, **kwargs): return "default_project"
    def sanitize_filename(*args, **kwargs): return "default_file"

# Load environment variables
load_dotenv()

X_API_KEY = os.getenv("X_API_KEY")
if not X_API_KEY:
    logger.warning("X_API_KEY environment variable is not set. API calls may fail.")

class EnhancedCodeGeneratorAgent(XAIAgent):
    """Enhanced code generator agent with language-specific optimizations"""

    def __init__(
        self,
        name: str = "Enhanced Code Generator",
        language: str = "python",
        complexity: str = "medium",
    ):
        """
        Initialize enhanced code generator agent
        """
        super().__init__(name=name)
        self.language = language.lower()
        self.complexity = complexity.lower()
        self._setup_language_prompts()

    def _setup_language_prompts(self):
        """Configure prompts based on language and complexity"""
        prompts = {
            "python": {
                "simple": "Write clean, PEP8 compliant Python code.",
                "medium": "Write professional Python code with type hints and docstrings.",
                "complex": "Write highly optimized, scalable Python code using design patterns."
            },
            "javascript": {
                "simple": "Write clean JavaScript code.",
                "medium": "Write professional JS code using ES6+ standards.",
                "complex": "Write high-performance JS code with TypeScript-like strictness."
            }
        }
        # Default to python if language not found
        lang_prompts = prompts.get(self.language, prompts["python"])
        self.system_prompt_extension = lang_prompts.get(self.complexity, "Write clean code.")

    def generate_code(self, prompt: str) -> str:
        """
        Generate code based on the prompt and agent configuration.
        """
        try:
            full_prompt = f"{self.system_prompt_extension}\n\nTask: {prompt}"
            logger.info(f"Generating {self.language} code with {self.complexity} complexity...")
            # Assuming XAIAgent has a method to call the LLM
            response = self.call_llm(full_prompt) 
            return response
        except Exception as e:
            logger.error(f"Error during code generation: {e}")
            return f"Error generating code: {str(e)}"

    def call_llm(self, prompt: str) -> str:
        """Mock LLM call for demonstration if XAIAgent doesn't implement it"""
        if not X_API_KEY:
            return "Error: X_API_KEY missing"
        # Implementation would go here
        return f"Generated code for: {prompt[:20]}..."

def run_advanced_evaluation(prompt: str, language: str = "python", complexity: str = "medium"):
    """
    Execution pipeline for advanced code generation and evaluation.
    """
    try:
        agent = EnhancedCodeGeneratorAgent(language=language, complexity=complexity)
        code_response = agent.generate_code(prompt)
        
        files = extract_files_from_response(code_response)
        project_name = generate_project_dir_name(prompt)
        
        project_path = Path(project_name)
        project_path.mkdir(parents=True, exist_ok=True)
        
        for filename, content in files.items():
            safe_name = sanitize_filename(filename)
            file_path = project_path / safe_name
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Saved: {file_path}")
            
        return project_name
    except Exception as e:
        logger.exception(f"Critical error in evaluation pipeline: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAI Advanced Code Gen Eval")
    parser.add_argument("--prompt", type=str, default="Create a simple FastAPI app", help="Prompt for code generation")
    parser.add_argument("--lang", type=str, default="python", help="Language")
    parser.add_argument("--complexity", type=str, default="medium", help="Complexity")
    
    args = parser.parse_args()
    
    result = run_advanced_evaluation(args.prompt, args.lang, args.complexity)
    if result:
        print(f"Successfully generated project: {result}")
    else:
        print("Generation failed.")
