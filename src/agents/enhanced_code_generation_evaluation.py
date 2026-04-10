"""
Enhanced AutoGen Code Generation and Evaluation System
==================================================
This script implements two agents for code generation and evaluation interaction using AutoGen:
1. Agent A: Code Generator - Based on X.AI's grok model, generates code from task descriptions
2. Agent B: Code Evaluator - Based on X.AI's grok model, evaluates code and provides optimization suggestions

The agents interact in a more structured way for two rounds.
"""

import os
import json
import time
import requests
import re
import autogen
from autogen import AssistantAgent, UserProxyAgent
from dotenv import load_dotenv
import logging
from typing import Dict, List, Any, Optional, Tuple
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"autogen_xai_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check environment variables
X_API_KEY = os.getenv("X_API_KEY")
if not X_API_KEY:
    raise ValueError("Please set X_API_KEY environment variable")


# Custom X.AI Agent class
class XAIAgent:
    """Custom XAI agent for code generation and evaluation"""

    def __init__(self, name: str, system_message: str, temperature: float = 0.7):
        """
        Initialize XAI agent

        Args:
            name: Agent name
            system_message: System message/prompt
            temperature: Generation temperature (0.0-1.0)
        """
        self.name = name
        self.system_message = system_message
        self.temperature = temperature
        self.model = "grok-3-latest"
        self.api_url = "https://api.x.ai/v1/chat/completions"
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {X_API_KEY}"}

        logger.info(f"Successfully initialized XAI agent: {self.name}")

    def generate_response(self, prompt: str) -> str:
        """
        Generate response

        Args:
            prompt: Input prompt

        Returns:
            Generated response text
        """
        try:
            # Prepare request data
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt},
            ]

            data = {
                "messages": messages,
                "model": self.model,
                "stream": False,
                "temperature": self.temperature,
            }

            # Send request
            response = requests.post(self.api_url, headers=self.headers, data=json.dumps(data))

            # Check response status
            response.raise_for_status()
            response_json = response.json()

            # Extract reply content
            content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")

            return content
        except Exception as e:
            logger.error(f"Error generating XAI response: {str(e)}")
            return f"Error generating response: {str(e)}"


# Custom code generator agent
class CodeGeneratorAgent(XAIAgent):
    """Code generator agent focused on generating code from task descriptions"""

    def __init__(self, name: str = "Code Generator"):
        """Initialize code generator agent"""
        system_message = """你是一位专业的Python开发者，负责根据任务要求编写高质量的代码。
请遵循以下原则:
1. 代码应该高效、清晰且易于理解
2. 提供适当的注释和文档字符串
3. 使用合适的设计模式和最佳实践
4. 避免重复代码，保持代码的模块化
5. 考虑边缘情况和异常处理
6. 确保代码具有良好的可读性和可维护性

如果任务需要多个文件，请为每个文件提供完整的代码实现。对于每个文件，请用以下格式明确标记：

FILE: filename.py
```python
# 文件内容
```

请务必提供完整可运行的代码实现，不要省略任何重要部分。
在回复中，请先简要说明你的实现思路，然后提供完整的代码。"""

        super().__init__(
            name=name,
            system_message=system_message,
            temperature=0.2,  # Use lower temperature for more stable code output
        )


# Custom code evaluator agent
class CodeEvaluatorAgent(XAIAgent):
    """Code evaluator agent focused on evaluating code quality and providing optimization suggestions"""

    def __init__(self, name: str = "Code Evaluator"):
        """Initialize code evaluator agent"""
        system_message = """你是一位经验丰富的代码审查者，负责评估和优化其他开发者的代码。
请遵循以下评估原则:
1. 详细分析代码的性能、可读性和可维护性
2. 找出代码中的潜在问题和改进空间
3. 给出具体的优化建议和可行的解决方案
4. 提供建设性的反馈，指出代码的优点和不足
5. 考虑边缘情况和异常处理是否完整
6. 检查是否遵循Python最佳实践和设计模式

如果代码包含多个文件，请对每个文件进行评估，或者提供整体评估。

请按照以下格式提供评估:

## 代码评估
### 优点
- [列出代码的优点]

### 需要改进
- [列出需要改进的地方]

## 优化建议
1. [具体的优化建议1]
2. [具体的优化建议2]
...

## 总体评分
[1-10分，并简要说明理由]"""

        super().__init__(
            name=name,
            system_message=system_message,
            temperature=0.1,  # Use even lower temperature for more consistent evaluation
        )


def extract_files_from_response(response_text: str) -> Dict[str, str]:
    """
    Extract multiple files from response text

    Args:
        response_text: Text containing file code blocks

    Returns:
        Dictionary mapping filenames to code content
    """
    # First try to extract files with explicit FILE: markers
    file_pattern = r"FILE:\s*([^\n]+?)\s*```(?:python|html|javascript|js|css)?\s*([\s\S]*?)```"
    file_matches = re.findall(file_pattern, response_text)

    files = {}

    if file_matches:
        # Found files with explicit markers
        for filename, content in file_matches:
            # Sanitize filename to remove invalid characters
            safe_filename = sanitize_filename(filename.strip())
            files[safe_filename] = content.strip()
    else:
        # Try to find filename hints in text
        filename_hints = re.findall(
            r"(?:Let\'s create|create|saving|save|file|named|called)\s+`?([a-zA-Z0-9_]+\.[a-z]+)`?",
            response_text,
        )

        # Find all code blocks with different languages
        code_blocks = []
        for lang in ["python", "html", "javascript", "js", "css", ""]:
            pattern = r"```(?:" + lang + r")?\s*([\s\S]*?)```"
            blocks = re.findall(pattern, response_text)
            code_blocks.extend(blocks)

        if code_blocks:
            if len(code_blocks) == 1 and not filename_hints:
                # Only one code block and no filename hints, determine file type
                if "<!DOCTYPE html>" in code_blocks[0] or "<html" in code_blocks[0]:
                    return {"index.html": code_blocks[0].strip()}
                elif (
                    "function" in code_blocks[0]
                    or "const" in code_blocks[0]
                    or "var" in code_blocks[0]
                ):
                    return {"script.js": code_blocks[0].strip()}
                else:
                    return {"main.py": code_blocks[0].strip()}
            elif len(code_blocks) == len(filename_hints):
                # Map each filename hint to a code block
                for i, filename in enumerate(filename_hints):
                    # Sanitize filename
                    safe_filename = sanitize_filename(filename)
                    files[safe_filename] = code_blocks[i].strip()
            else:
                # Detect file types by content
                detected_files = {}
                for i, code in enumerate(code_blocks):
                    if i < len(filename_hints):
                        # Use hint but sanitize
                        safe_filename = sanitize_filename(filename_hints[i])
                        detected_files[safe_filename] = code.strip()
                    else:
                        # Determine file type by content
                        if "<!DOCTYPE html>" in code or "<html" in code:
                            detected_files[f"index.html"] = code.strip()
                        elif "import fastapi" in code or "from fastapi import" in code:
                            detected_files[f"main.py"] = code.strip()
                        elif "function" in code or "const" in code or "var" in code:
                            detected_files[f"script.js"] = code.strip()
                        elif "body" in code and "{" in code:
                            detected_files[f"styles.css"] = code.strip()
                        else:
                            detected_files[f"file_{i+1}.py"] = code.strip()
                files = detected_files
        else:
            # If no code blocks found, return the original text in a default file
            files["main.py"] = response_text

    # If still no files detected, create a single file with all content
    if not files:
        code = extract_code_from_markdown(response_text)
        if "<!DOCTYPE html>" in code or "<html" in code:
            files["index.html"] = code
        elif "import fastapi" in code or "from fastapi import" in code:
            files["main.py"] = code
        else:
            files["main.py"] = code

    # Ensure we resolve any duplicate filenames by adding a suffix
    resolved_files = {}
    for filename, content in files.items():
        base, ext = os.path.splitext(filename)
        counter = 1
        new_filename = filename
        while new_filename in resolved_files:
            new_filename = f"{base}_{counter}{ext}"
            counter += 1
        resolved_files[new_filename] = content

    return resolved_files


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to remove invalid characters

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, "", filename)

    # Replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")

    # Trim excessively long filenames
    if len(sanitized) > 100:
        base, ext = os.path.splitext(sanitized)
        sanitized = base[:95] + ext

    # Use a default name if empty
    if not sanitized or sanitized == ".":
        return "main.py"

    # Ensure we have an extension
    if "." not in sanitized:
        # Add appropriate extension based on content
        sanitized += ".py"

    return sanitized


def extract_code_from_markdown(markdown_text: str) -> str:
    """
    Extract code from markdown formatted text that has ```python ... ``` blocks

    Args:
        markdown_text: Text potentially containing code blocks

    Returns:
        Extracted code without the markdown formatting
    """
    # Look for Python code blocks
    python_code_pattern = r"```(?:python)?\s*([\s\S]*?)```"
    matches = re.findall(python_code_pattern, markdown_text)

    if matches:
        # Return the largest code block (likely the complete implementation)
        return max(matches, key=len).strip()
    else:
        # If no code blocks found, return the original text
        return markdown_text


def get_module_name_from_task(task_description: str) -> str:
    """
    Generate a suitable module name from the task description

    Args:
        task_description: Task description text

    Returns:
        A Python module name (snake_case)
    """
    # Extract the first sentence or first 50 characters
    first_line = task_description.strip().split("\n")[0]
    first_sentence = first_line.split(".")[0]

    # Remove unnecessary words and normalize
    name = first_sentence.lower()
    name = re.sub(r"implement\s+a\s+", "", name)
    name = re.sub(r"create\s+a\s+", "", name)
    name = re.sub(r"develop\s+a\s+", "", name)
    name = re.sub(r"build\s+a\s+", "", name)

    # Convert to snake_case
    name = re.sub(r"[^\w\s]", "", name)  # Remove punctuation
    name = re.sub(r"\s+", "_", name)  # Replace spaces with underscores

    # Ensure name doesn't start with a number
    if re.match(r"^\d", name):
        name = "x_" + name

    # Limit length
    if len(name) > 40:
        name = name[:40]

    return name


def generate_project_dir_name() -> str:
    """
    Generate a project directory name using timestamp

    Returns:
        A directory name using timestamp format
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"project_{timestamp}"


# Define code generation evaluation workflow function
def run_code_generation_evaluation_workflow(task_description: str) -> Dict[str, Any]:
    """
    Run complete code generation and evaluation workflow with two rounds of interaction

    Args:
        task_description: Code generation task description

    Returns:
        Dictionary containing generated content from the entire interaction
    """
    # Create agents
    code_generator = CodeGeneratorAgent()
    code_evaluator = CodeEvaluatorAgent()

    # Dictionary for storing results
    results = {
        "task_description": task_description,
        "initial_code_response": "",
        "initial_files": {},
        "initial_evaluation": "",
        "optimized_code_response": "",
        "optimized_files": {},
        "final_evaluation": "",
        "project_dir": generate_project_dir_name(),
    }

    # First round: code generation
    logger.info("=" * 80)
    logger.info("Starting first round of code generation")
    logger.info("=" * 80)

    code_prompt = f"""Please write high-quality Python code based on the following requirements:

{task_description}

If the task requires multiple files, please create separate files and clearly indicate the filename for each file using the format 'FILE: filename.py' before each code block.

Please provide a complete implementation and ensure the code can run directly."""

    initial_code_response = code_generator.generate_response(code_prompt)
    results["initial_code_response"] = initial_code_response

    # Extract files from the response
    initial_files = extract_files_from_response(initial_code_response)
    results["initial_files"] = initial_files

    logger.info(f"First round code generation completed. Number of files: {len(initial_files)}")
    for filename in initial_files:
        logger.info(f"  - {filename}: {len(initial_files[filename])} characters")

    # First round: code evaluation
    logger.info("=" * 80)
    logger.info("Starting first round of code evaluation")
    logger.info("=" * 80)

    # Prepare the evaluation prompt based on files
    evaluation_prompt = f"""Please evaluate the following Python code based on the requirements:

Task Description:
{task_description}

Code Implementation:
"""

    # Add code blocks for each file
    for filename, content in initial_files.items():
        evaluation_prompt += f"\nFILE: {filename}\n```python\n{content}\n```\n"

    evaluation_prompt += "\nPlease evaluate the code quality in detail, identify strengths and weaknesses, and provide specific optimization suggestions for each file."

    initial_evaluation = code_evaluator.generate_response(evaluation_prompt)
    results["initial_evaluation"] = initial_evaluation
    logger.info(
        f"First round code evaluation completed, length: {len(initial_evaluation)} characters"
    )

    # Second round: code optimization
    logger.info("=" * 80)
    logger.info("Starting second round of code optimization")
    logger.info("=" * 80)

    optimization_prompt = f"""Please optimize your previous code based on the evaluation feedback:

Original Task:
{task_description}

Your original code implementation:
"""

    # Add code blocks for each file
    for filename, content in initial_files.items():
        optimization_prompt += f"\nFILE: {filename}\n```python\n{content}\n```\n"

    optimization_prompt += f"""
Evaluation Feedback:
{initial_evaluation}

Please provide an optimized complete code implementation based on the evaluation. 
Pay special attention to addressing the issues identified in the evaluation.

If the implementation requires multiple files, please clearly indicate each filename using the format 'FILE: filename.py' before each code block."""

    optimized_code_response = code_generator.generate_response(optimization_prompt)
    results["optimized_code_response"] = optimized_code_response

    # Extract optimized files
    optimized_files = extract_files_from_response(optimized_code_response)
    results["optimized_files"] = optimized_files

    logger.info(f"Optimized code generation completed. Number of files: {len(optimized_files)}")
    for filename in optimized_files:
        logger.info(f"  - {filename}: {len(optimized_files[filename])} characters")

    # Final evaluation
    logger.info("=" * 80)
    logger.info("Starting final code evaluation")
    logger.info("=" * 80)

    final_evaluation_prompt = f"""Please compare and evaluate the following two code versions:

Task Description:
{task_description}

Original Code:
"""

    # Add original code blocks
    for filename, content in initial_files.items():
        final_evaluation_prompt += f"\nFILE: {filename}\n```python\n{content}\n```\n"

    final_evaluation_prompt += "\nOptimized Code:\n"

    # Add optimized code blocks
    for filename, content in optimized_files.items():
        final_evaluation_prompt += f"\nFILE: {filename}\n```python\n{content}\n```\n"

    final_evaluation_prompt += """
Please evaluate whether the optimized code addresses the issues identified in the previous evaluation, 
whether the overall code quality has improved, and whether there is still room for further improvement. 
Please provide a detailed final evaluation for each file and an overall assessment."""

    final_evaluation = code_evaluator.generate_response(final_evaluation_prompt)
    results["final_evaluation"] = final_evaluation
    logger.info(f"Final evaluation completed, length: {len(final_evaluation)} characters")

    logger.info("=" * 80)
    logger.info("Code generation and evaluation workflow completed")
    logger.info("=" * 80)

    # Generate module name from task
    module_name = get_module_name_from_task(task_description)
    results["module_name"] = module_name

    return results


def save_results(results: Dict[str, Any], output_dir: str = "results") -> None:
    """
    Save workflow results to appropriate files with numbered iteration folders

    Args:
        results: Dictionary containing workflow results
        output_dir: Output directory for files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get project directory name
    project_dir = results["project_dir"]
    project_path = os.path.join(output_dir, project_dir)

    # Create subdirectory for this specific project
    os.makedirs(project_path, exist_ok=True)

    # Create numbered directories for iterations
    # 1. First iteration - Initial code generation
    initial_dir = os.path.join(project_path, "1_initial_code")
    os.makedirs(initial_dir, exist_ok=True)

    # Save initial files
    for filename, content in results["initial_files"].items():
        file_path = os.path.join(initial_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    # Save initial code review
    review_path = os.path.join(initial_dir, "code_review.md")
    with open(review_path, "w", encoding="utf-8") as f:
        f.write("# Initial Code Review\n\n")
        f.write(results["initial_evaluation"])

    # 2. Second iteration - Optimized code after review
    optimized_dir = os.path.join(project_path, "2_optimized_code")
    os.makedirs(optimized_dir, exist_ok=True)

    # Save optimized files
    for filename, content in results["optimized_files"].items():
        file_path = os.path.join(optimized_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    # Save final code review
    review_path = os.path.join(optimized_dir, "code_review.md")
    with open(review_path, "w", encoding="utf-8") as f:
        f.write("# Final Code Review\n\n")
        f.write(results["final_evaluation"])

    # Also save optimized files in the root directory for easy access
    saved_files = []
    for filename, content in results["optimized_files"].items():
        file_path = os.path.join(project_path, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        saved_files.append(filename)

    # Save comprehensive documentation as Markdown file
    doc_path = os.path.join(project_path, "documentation.md")
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(f"# Generated Project: {project_dir}\n\n")

        # Add task description section
        f.write(f"## Task Description\n\n{results['task_description']}\n\n")

        # Add usage instructions
        f.write(f"## Usage\n\n")
        f.write(f"The code is available in the following files:\n\n")

        for filename in results["optimized_files"].keys():
            f.write(f"- `{filename}`\n")
        f.write("\n")

        # Add iteration information
        f.write(f"## Code Iterations\n\n")
        f.write(f"This project contains the following iteration folders:\n\n")
        f.write(f"- `1_initial_code/`: Initial code generation\n")
        f.write(f"- `2_optimized_code/`: Code after review and optimization\n\n")

        # Add code evaluation
        f.write(f"## Final Code Evaluation\n\n{results['final_evaluation']}\n\n")

        # Add review process section
        f.write(f"## Code Review Process\n\n")
        f.write(f"### Initial Review\n\n{results['initial_evaluation']}\n\n")

        # Add files section with original and optimized versions
        f.write(f"## Original Code (iteration 1)\n\n")
        for filename, content in results["initial_files"].items():
            f.write(f"### {filename}\n\n```python\n{content}\n```\n\n")

        f.write(f"## Optimized Code (iteration 2)\n\n")
        for filename, content in results["optimized_files"].items():
            f.write(f"### {filename}\n\n```python\n{content}\n```\n\n")

    logger.info(f"Results saved to directory: {project_path}")
    print(f"\nFiles generated successfully in directory: {project_path}")
    print("Generated files and iterations:")
    print(f"- 1_initial_code/")
    print(f"- 2_optimized_code/")
    print("- Final code files:")
    for filename in saved_files:
        print(f"  - {filename}")
    print(f"- documentation.md (comprehensive documentation)")


if __name__ == "__main__":
    # Example task - modified to request multiple files
    task = """
    幫我完成，後端使用 python fastapi 寫一個hello world api，前端寫一個簡單的htnl呼叫後端hello world api，需考慮cros問題
    """

    # Run workflow
    results = run_code_generation_evaluation_workflow(task)

    # Save results
    save_results(results)

    print("=" * 50)
    print("Code generation and evaluation workflow completed!")
    print("=" * 50)
