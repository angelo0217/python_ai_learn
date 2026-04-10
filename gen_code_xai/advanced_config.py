"""
AutoGen XAI Code Generation and Evaluation System - Advanced Configuration
=======================================
This script demonstrates how to customize and configure the code generation evaluation system.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv
import autogen

# Import base modules from enhanced version
from enhanced_code_generation_evaluation import (
    XAIAgent,
    extract_files_from_response,
    generate_project_dir_name,
    sanitize_filename,
    logger,
)

# Load environment variables
load_dotenv()

X_API_KEY = os.getenv("X_API_KEY")
if not X_API_KEY:
    logger.error("X_API_KEY environment variable is not set")
    raise ValueError("Please set X_API_KEY environment variable")

class EnhancedCodeGeneratorAgent(XAIAgent):
    """
    Enhanced code generator agent with language-specific optimizations.
    Inherits from XAIAgent to provide specialized prompting based on language and complexity.
    """

    def __init__(
        self,
        name: str = "Enhanced Code Generator",
        language: str = "python",
        complexity: str = "medium",
    ):
        """
        Initialize enhanced code generator agent.

        Args:
            name: Agent name.
            language: Target programming language ("python", "javascript", "java", "cpp", "go").
            complexity: Code complexity ("simple", "medium", "complex").
        """
        super().__init__(name=name)
        self.language = language.lower()
        self.complexity = complexity.lower()
        self._setup_language_prompts()

    def _setup_language_prompts(self) -> None:
        """Configure prompts based on the target language and complexity."""
        language_prompts = {
            "python": "Focus on PEP 8 compliance, type hinting, and efficient list comprehensions.",
            "javascript": "Focus on ES6+ syntax, asynchronous patterns (async/await), and modularity.",
            "java": "Focus on strong typing, design patterns, and proper exception handling.",
            "cpp": "Focus on memory management (smart pointers), RAII, and STL efficiency.",
            "go": "Focus on concurrency (goroutines/channels) and explicit error handling.",
        }
        
        complexity_prompts = {
            "simple": "Keep the implementation concise and easy to understand.",
            "medium": "Implement a robust solution with error handling and modular structure.",
            "complex": "Implement a highly scalable, optimized solution with comprehensive design patterns.",
        }

        lang_guidance = language_prompts.get(self.language, "Follow general clean code principles.")
        comp_guidance = complexity_prompts.get(self.complexity, "Maintain a balanced level of complexity.")
        
        self.system_message = (
            f"You are an expert {self.language} developer. {lang_guidance} "
            f"The target complexity is {self.complexity}: {comp_guidance}"
        )

    def generate_code(self, prompt: str) -> str:
        """
        Generate code based on the provided prompt and agent configuration.
        
        Args:
            prompt: The requirements for the code to be generated.
            
        Returns:
            The generated code response.
        """
        full_prompt = f"{self.system_message}\n\nRequirement: {prompt}"
        return self.ask(full_prompt)

def main():
    """Main entry point for demonstrating the advanced configuration."""
    logger.info("Starting Advanced Code Generation Demo...")
    
    try:
        # Example: Generate a complex Python project
        generator = EnhancedCodeGeneratorAgent(language="python", complexity="complex")
        prompt = "Create a high-performance asynchronous web scraper with proxy rotation and rate limiting."
        
        logger.info(f"Generating code for: {prompt}")
        response = generator.generate_code(prompt)
        
        # Process the response to extract files
        files = extract_files_from_response(response)
        project_name = generate_project_dir_name(prompt)
        
        logger.info(f"Project '{project_name}' generated with {len(files)} files.")
        
        for filename, content in files.items():
            safe_name = sanitize_filename(filename)
            logger.info(f"Extracted file: {safe_name}")
            # In a real scenario, you would write these files to disk here.

    except Exception as e:
        logger.exception(f"An error occurred during code generation: {e}")

if __name__ == "__main__":
    main()
