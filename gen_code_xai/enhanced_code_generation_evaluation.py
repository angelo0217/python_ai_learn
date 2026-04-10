"""
Enhanced AutoGen Code Generation and Evaluation System
==================================================
This script implements a structured interaction between a Code Generator and a Code Evaluator
using AutoGen and X.AI's grok model.

The workflow consists of:
1. Agent A (Generator): Produces code based on task descriptions.
2. Agent B (Evaluator): Reviews code, identifies bugs, and suggests optimizations.
3. Iterative Loop: The agents interact for a set number of rounds to refine the output.
"""

import os
import json
import time
import requests
import re
import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import autogen
from autogen import AssistantAgent, UserProxyAgent
from dotenv import load_dotenv

# --- Configuration & Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"autogen_xai_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

load_dotenv()
X_API_KEY = os.getenv("X_API_KEY")
if not X_API_KEY:
    raise ValueError("Please set X_API_KEY environment variable")

@dataclass
class AgentConfig:
    """Configuration for XAI Agents"""
    name: str
    system_message: str
    temperature: float = 0.7
    model: str = "grok-beta"

class XAIAgent:
    """
    Custom XAI agent wrapper to handle direct API calls to X.AI 
    and integrate with AutoGen's messaging patterns.
    """
    def __init__(self, config: AgentConfig):
        self.config = config
        self.api_url = "https://api.x.ai/v1/chat/completions"

    def generate_response(self, prompt: str, history: List[Dict[str, str]] = None) -> str:
        """Generates a response from X.AI API based on prompt and history."""
        messages = [{"role": "system", "content": self.config.system_message}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {X_API_KEY}"
        }

        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return f"Error generating response: {str(e)}"

class CodeGenerationSystem:
    """
    Orchestrates the interaction between the Code Generator and Code Evaluator.
    """
    def __init__(self):
        self.generator_config = AgentConfig(
            name="CodeGenerator",
            system_message="You are an expert Python developer. Generate clean, efficient, and well-documented code."
        )
        self.evaluator_config = AgentConfig(
            name="CodeEvaluator",
            system_message="You are a senior code reviewer. Analyze code for bugs, security vulnerabilities, and performance issues. Provide specific optimization suggestions."
        )
        self.generator = XAIAgent(self.generator_config)
        self.evaluator = XAIAgent(self.evaluator_config)

    def run_evaluation_cycle(self, task_description: str, rounds: int = 2) -> str:
        """
        Executes the iterative generation and evaluation loop.
        """
        logger.info(f"Starting evaluation cycle for task: {task_description}")
        
        current_prompt = task_description
        history = []
        final_code = ""

        for round_num in range(1, rounds + 1):
            logger.info(f"--- Round {round_num} ---")
            
            # 1. Generation Phase
            generated_content = self.generator.generate_response(current_prompt, history)
            logger.info(f"[{self.generator_config.name}] generated code.")
            
            # Extract code block if present
            code_match = re.search(r"```python\n(.*?)\n```", generated_content, re.DOTALL)
            final_code = code_match.group(1) if code_match else generated_content
            
            # Update history
            history.append({"role": "assistant", "content": generated_content})

            # 2. Evaluation Phase
            eval_prompt = f"Evaluate the following code for the task: {task_description}\n\nCode:\n{generated_content}"
            evaluation = self.evaluator.generate_response(eval_prompt, history)
            logger.info(f"[{self.evaluator_config.name}] provided evaluation.")
            
            history.append({"role": "user", "content": evaluation})
            
            # Prepare prompt for next round
            current_prompt = f"Based on the following evaluation, please refine the code:\n\nEvaluation:\n{evaluation}"
            
        return final_code

if __name__ == "__main__":
    # Example Usage
    task = "Create a Python script that fetches the current price of Bitcoin from a public API and saves it to a CSV file every hour."
    system = CodeGenerationSystem()
    result_code = system.run_evaluation_cycle(task)
    
    print("\n" + "="*20 + " FINAL REFINED CODE " + "="*20)
    print(result_code)
    print("="*56)
