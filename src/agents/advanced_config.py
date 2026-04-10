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

# Check environment variables
X_API_KEY = os.getenv("X_API_KEY")
if not X_API_KEY:
    raise ValueError("Please set X_API_KEY environment variable")


# Enhanced code generator agent with language-specific optimizations
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

        Args:
            name: Agent name
            language: Target programming language ("python", "javascript", "java", "cpp", "go")
            complexity: Code complexity ("simple", "medium", "complex")
        """
        # Language-specific prompts
        language_prompts = {
            "python": """你是一位Python专家，专注于编写清晰、高效且符合PEP 8规范的代码。
请使用现代Python特性（如f-strings、类型注解、dataclasses等）。
优先考虑标准库，必要时再使用第三方库。""",
            "javascript": """你是一位JavaScript/TypeScript专家，专注于编写符合现代ES6+标准的代码。
优先使用函数式编程和不可变数据结构，尽量使用async/await而非回调。
适当时请使用TypeScript类型注解以提高代码质量。""",
            "java": """你是一位Java专家，专注于编写符合最新Java标准的面向对象代码。
遵循SOLID原则，适当使用设计模式，并注重代码的可测试性。
使用Java新特性如Stream API、Optional、var关键字等。""",
            "cpp": """你是一位C++专家，专注于编写高效、安全且现代的C++代码。
遵循C++17/20标准实践，如使用智能指针、避免原始指针、使用auto等。
关注内存安全和性能优化，适当使用STL和并发特性。""",
            "go": """你是一位Go专家，专注于编写符合Go惯用法的简洁高效代码。
遵循Go的设计哲学，如简单性、可读性和组合优于继承。
适当使用Go的并发特性如goroutines和channels，注重错误处理。""",
        }

        # Complexity-adjusted prompts
        complexity_prompts = {
            "simple": "请编写简单、易于理解的代码，避免过度设计和复杂抽象。适合初学者学习使用。",
            "medium": "请平衡代码的简洁性和功能完整性，提供合理的抽象和模块化设计。",
            "complex": "请编写企业级代码，包含完善的错误处理、日志记录、性能优化和扩展性设计。提供完整的文档和单元测试。",
        }

        # Get default prompts (if language not specified)
        lang_prompt = language_prompts.get(language.lower(), language_prompts["python"])
        complex_prompt = complexity_prompts.get(complexity.lower(), complexity_prompts["medium"])

        # Combine system prompts
        system_message = f"""你是一位专业的软件开发者，负责根据任务要求编写高质量的代码。

{lang_prompt}

{complex_prompt}

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

或者对于HTML文件：

FILE: filename.html
```html
<!-- HTML内容 -->
```

请务必提供完整可运行的代码实现，不要省略任何重要部分。
在回复中，请先简要说明你的实现思路，然后提供完整的代码。"""

        # Initialize base class
        super().__init__(
            name=name,
            system_message=system_message,
            temperature=0.2,  # Use lower temperature for more stable code output
        )

        # Save settings for later reference
        self.language = language
        self.complexity = complexity


# Enhanced code evaluator agent
class EnhancedCodeEvaluatorAgent(XAIAgent):
    """Enhanced code evaluator agent focused on evaluating code quality and providing optimization suggestions"""

    def __init__(self, name: str = "Enhanced Code Evaluator"):
        """Initialize enhanced code evaluator agent"""
        system_message = """你是一位经验丰富的代码审查者，负责评估和优化其他开发者的代码。
请遵循以下评估原则:
1. 详细分析代码的性能、可读性和可维护性
2. 找出代码中的潜在问题和改进空间
3. 给出具体的优化建议和可行的解决方案
4. 提供建设性的反馈，指出代码的优点和不足
5. 考虑边缘情况和异常处理是否完整
6. 检查是否遵循编程语言的最佳实践和设计模式

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
            temperature=0.1,  # Use lower temperature for more consistent evaluation
        )


# Enhanced code evaluation workflow
def run_enhanced_workflow(
    task_description: str,
    language: str = "python",
    complexity: str = "medium",
    iterations: int = 2,
    output_dir: str = "results",
) -> Dict[str, Any]:
    """
    Run enhanced code generation and evaluation workflow

    Args:
        task_description: Code generation task description
        language: Target programming language
        complexity: Code complexity
        iterations: Number of optimization iterations
        output_dir: Output directory

    Returns:
        Dictionary containing generated content from the entire interaction
    """
    # Create agents
    code_generator = EnhancedCodeGeneratorAgent(
        name="Code Generator", language=language, complexity=complexity
    )
    code_evaluator = EnhancedCodeEvaluatorAgent(name="Code Evaluator")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Results dictionary
    results = {
        "task_description": task_description,
        "language": language,
        "complexity": complexity,
        "iterations": iterations,
        "iteration_history": [],
    }

    # First round code generation
    logger.info("=" * 80)
    logger.info(
        f"Starting Round 1 code generation (Language: {language}, Complexity: {complexity})"
    )
    logger.info("=" * 80)

    code_prompt = f"""Please write high-quality {language.upper()} code based on the following requirements:

{task_description}

If the task requires multiple files, please create separate files and clearly indicate the filename for each file using the format 'FILE: filename.py' or 'FILE: filename.html' before each code block.

Please provide a complete implementation and ensure the code can run directly."""

    current_code_response = code_generator.generate_response(code_prompt)

    # Extract files from the response
    current_files = extract_files_from_response(current_code_response)

    # Record first round result
    iteration_result = {
        "round": 1,
        "code_response": current_code_response,
        "files": current_files,
        "evaluation": "",
    }

    # Multiple iteration optimization
    for i in range(iterations):
        # Code evaluation
        logger.info("=" * 80)
        logger.info(f"Starting Round {i+1} code evaluation")
        logger.info("=" * 80)

        # Prepare the evaluation prompt based on files
        evaluation_prompt = f"""Please evaluate the following {language.upper()} code based on the requirements:

Task Description:
{task_description}

Code Implementation:
"""

        # Add code blocks for each file
        for filename, content in current_files.items():
            file_ext = os.path.splitext(filename)[1].lower()
            lang = language

            # Determine language for markdown code block
            if file_ext == ".html":
                lang = "html"
            elif file_ext == ".js":
                lang = "javascript"
            elif file_ext == ".css":
                lang = "css"
            elif file_ext == ".java":
                lang = "java"
            elif file_ext == ".cpp" or file_ext == ".h":
                lang = "cpp"
            elif file_ext == ".go":
                lang = "go"

            evaluation_prompt += f"\nFILE: {filename}\n```{lang}\n{content}\n```\n"

        evaluation_prompt += "\nPlease evaluate the code quality in detail, identify strengths and weaknesses, and provide specific optimization suggestions for each file."

        evaluation = code_evaluator.generate_response(evaluation_prompt)
        iteration_result["evaluation"] = evaluation

        # Add to result history
        results["iteration_history"].append(iteration_result.copy())

        # Last round doesn't need optimization
        if i == iterations - 1:
            break

        # Optimize code based on evaluation
        logger.info("=" * 80)
        logger.info(f"Starting Round {i+2} code optimization")
        logger.info("=" * 80)

        optimization_prompt = f"""Please optimize your previous code based on the evaluation feedback:

Original Task:
{task_description}

Your original code implementation:
"""

        # Add code blocks for each file
        for filename, content in current_files.items():
            file_ext = os.path.splitext(filename)[1].lower()
            lang = language

            # Determine language for markdown code block
            if file_ext == ".html":
                lang = "html"
            elif file_ext == ".js":
                lang = "javascript"
            elif file_ext == ".css":
                lang = "css"
            elif file_ext == ".java":
                lang = "java"
            elif file_ext == ".cpp" or file_ext == ".h":
                lang = "cpp"
            elif file_ext == ".go":
                lang = "go"

            optimization_prompt += f"\nFILE: {filename}\n```{lang}\n{content}\n```\n"

        optimization_prompt += f"""
Evaluation Feedback:
{evaluation}

Please provide an optimized complete code implementation based on the evaluation feedback.
Pay special attention to addressing the issues identified in the evaluation and adopting the specific improvement suggestions.

If the implementation requires multiple files, please clearly indicate each filename using the format 'FILE: filename.ext' before each code block.
Please provide the complete code for each file without omitting any parts."""

        optimized_code_response = code_generator.generate_response(optimization_prompt)

        # Extract optimized files
        current_files = extract_files_from_response(optimized_code_response)

        # Prepare next iteration record
        iteration_result = {
            "round": i + 2,
            "code_response": optimized_code_response,
            "files": current_files,
            "evaluation": "",
        }

    # Final evaluation
    logger.info("=" * 80)
    logger.info("Starting final code evaluation")
    logger.info("=" * 80)

    final_evaluation_prompt = f"""Please compare and evaluate the initial and final optimized code versions for the following task:

Task Description:
{task_description}

Initial Code (Round 1):
"""

    # Add initial code blocks
    for filename, content in results["iteration_history"][0]["files"].items():
        file_ext = os.path.splitext(filename)[1].lower()
        lang = language

        # Determine language for markdown code block
        if file_ext == ".html":
            lang = "html"
        elif file_ext == ".js":
            lang = "javascript"
        elif file_ext == ".css":
            lang = "css"
        elif file_ext == ".java":
            lang = "java"
        elif file_ext == ".cpp" or file_ext == ".h":
            lang = "cpp"
        elif file_ext == ".go":
            lang = "go"

        final_evaluation_prompt += f"\nFILE: {filename}\n```{lang}\n{content}\n```\n"

    final_evaluation_prompt += "\nFinal Code (Round {iterations}):\n"

    # Add final code blocks
    for filename, content in current_files.items():
        file_ext = os.path.splitext(filename)[1].lower()
        lang = language

        # Determine language for markdown code block
        if file_ext == ".html":
            lang = "html"
        elif file_ext == ".js":
            lang = "javascript"
        elif file_ext == ".css":
            lang = "css"
        elif file_ext == ".java":
            lang = "java"
        elif file_ext == ".cpp" or file_ext == ".h":
            lang = "cpp"
        elif file_ext == ".go":
            lang = "go"

        final_evaluation_prompt += f"\nFILE: {filename}\n```{lang}\n{content}\n```\n"

    final_evaluation_prompt += """
Please comprehensively evaluate the improvement in code quality, analyze whether the optimization process has addressed key issues,
and whether it conforms to the best practices. Please provide a detailed final evaluation for each file and an overall assessment."""

    final_evaluation = code_evaluator.generate_response(final_evaluation_prompt)
    results["final_evaluation"] = final_evaluation

    logger.info("=" * 80)
    logger.info("Enhanced code generation and evaluation workflow completed")
    logger.info("=" * 80)

    # Generate project directory name
    project_dir = generate_project_dir_name()
    results["project_dir"] = project_dir

    # Create a subdirectory for this specific project
    project_path = os.path.join(output_dir, project_dir)
    os.makedirs(project_path, exist_ok=True)

    # Create numbered directories for each iteration
    saved_files_by_iteration = {}

    for i, iter_result in enumerate(results["iteration_history"]):
        # Create iteration directory with numeric prefix
        iter_dirname = f"{i+1}_iteration_code"
        iter_path = os.path.join(project_path, iter_dirname)
        os.makedirs(iter_path, exist_ok=True)

        # Save files for this iteration
        saved_files = []
        for filename, content in iter_result["files"].items():
            file_path = os.path.join(iter_path, filename)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            saved_files.append(filename)

        saved_files_by_iteration[iter_dirname] = saved_files

        # Save evaluation file for this iteration
        if iter_result["evaluation"]:
            review_path = os.path.join(iter_path, "code_review.md")
            with open(review_path, "w", encoding="utf-8") as f:
                if i == len(results["iteration_history"]) - 1:
                    f.write(f"# Final Code Review (Iteration {i+1})\n\n")
                else:
                    f.write(f"# Code Review (Iteration {i+1})\n\n")
                f.write(iter_result["evaluation"])

    # Also save final files to the root directory for easy access
    final_files = []
    final_iteration = results["iteration_history"][-1]
    for filename, content in final_iteration["files"].items():
        file_path = os.path.join(project_path, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        final_files.append(filename)

    # Save documentation to a Markdown file
    doc_file_name = "documentation.md"
    doc_path = os.path.join(project_path, doc_file_name)

    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(f"# Generated Project ({language.title()})\n\n")
        f.write(f"- **Language**: {language}\n")
        f.write(f"- **Complexity**: {complexity}\n")
        f.write(f"- **Iterations**: {iterations}\n\n")

        # Add task description section
        f.write(f"## Task Description\n\n{task_description}\n\n")

        # Add usage instructions
        f.write(f"## Usage\n\n")
        f.write(f"The code is available in the following files:\n\n")

        for filename in final_files:
            f.write(f"- `{filename}`\n")
        f.write("\n")

        # Add information about iteration folders
        f.write(f"## Code Iterations\n\n")
        f.write(f"This project contains the following iteration folders:\n\n")

        for i in range(len(results["iteration_history"])):
            iter_dirname = f"{i+1}_iteration_code"
            if i == len(results["iteration_history"]) - 1:
                f.write(f"- `{iter_dirname}/`: Final code iteration\n")
            else:
                f.write(f"- `{iter_dirname}/`: Iteration {i+1}\n")
        f.write("\n")

        # Add code evaluation
        f.write(f"## Final Code Evaluation\n\n{final_evaluation}\n\n")

        # Add development history
        f.write(f"## Development History\n\n")
        for i, iter_result in enumerate(results["iteration_history"]):
            f.write(f"### Round {iter_result['round']}\n\n")
            if i < len(results["iteration_history"]) - 1:
                f.write(f"#### Evaluation\n\n{iter_result['evaluation']}\n\n")

        # Add both code versions for reference
        f.write(f"## Initial Files (for reference)\n\n")
        for filename, content in results["iteration_history"][0]["files"].items():
            file_ext = os.path.splitext(filename)[1].lower()
            lang = language

            # Determine language for markdown code block
            if file_ext == ".html":
                lang = "html"
            elif file_ext == ".js":
                lang = "javascript"
            elif file_ext == ".css":
                lang = "css"
            elif file_ext == ".java":
                lang = "java"
            elif file_ext == ".cpp" or file_ext == ".h":
                lang = "cpp"
            elif file_ext == ".go":
                lang = "go"

            f.write(f"### {filename}\n\n```{lang}\n{content}\n```\n\n")

        f.write(f"## Final Files (implemented)\n\n")
        for filename, content in final_iteration["files"].items():
            file_ext = os.path.splitext(filename)[1].lower()
            lang = language

            # Determine language for markdown code block
            if file_ext == ".html":
                lang = "html"
            elif file_ext == ".js":
                lang = "javascript"
            elif file_ext == ".css":
                lang = "css"
            elif file_ext == ".java":
                lang = "java"
            elif file_ext == ".cpp" or file_ext == ".h":
                lang = "cpp"
            elif file_ext == ".go":
                lang = "go"

            f.write(f"### {filename}\n\n```{lang}\n{content}\n```\n\n")

    logger.info(f"Results saved to directory: {project_path}")
    print(f"\nFiles generated successfully in directory: {project_path}")
    print("Generated iteration folders:")
    for i in range(len(results["iteration_history"])):
        iter_dirname = f"{i+1}_iteration_code"
        print(f"- {iter_dirname}/")

    print("Final files in root directory:")
    for filename in final_files:
        print(f"- {filename}")
    print(f"- {doc_file_name}")

    return results


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="AutoGen XAI Code Generation and Evaluation System - Advanced Configuration"
    )

    # Task-related parameters
    parser.add_argument(
        "--task_file", type=str, help="Path to a text file containing the task description"
    )
    parser.add_argument(
        "--task", type=str, help="Task description (provided directly on the command line)"
    )

    # Code generation configuration
    parser.add_argument(
        "--language",
        type=str,
        default="python",
        choices=["python", "javascript", "java", "cpp", "go"],
        help="Target programming language",
    )
    parser.add_argument(
        "--complexity",
        type=str,
        default="medium",
        choices=["simple", "medium", "complex"],
        help="Code complexity level",
    )

    # Workflow configuration
    parser.add_argument(
        "--iterations", type=int, default=2, help="Number of code optimization iterations"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output directory for results"
    )

    # Parse arguments
    args = parser.parse_args()

    # Get task description
    task_description = None
    if args.task_file:
        try:
            with open(args.task_file, "r", encoding="utf-8") as f:
                task_description = f.read()
        except Exception as e:
            logger.error(f"Error reading task file: {str(e)}")
            return
    elif args.task:
        task_description = args.task
    else:
        # Default example task
        task_description = """
        Implement a simple web application with the following features:
        1. A Flask web server with routes for displaying data and handling form submissions
        2. A SQLite database for storing user data
        3. HTML templates for the user interface (use a separate file)
        4. A configuration file for app settings
        5. A utility module with helper functions
        6. Basic user authentication
        """

    # Run workflow
    logger.info(f"Starting workflow with the following configuration:")
    logger.info(f"- Language: {args.language}")
    logger.info(f"- Complexity: {args.complexity}")
    logger.info(f"- Iterations: {args.iterations}")

    results = run_enhanced_workflow(
        task_description=task_description,
        language=args.language,
        complexity=args.complexity,
        iterations=args.iterations,
        output_dir=args.output_dir,
    )

    print(f"\nTask completed! Results saved to {args.output_dir} directory")


if __name__ == "__main__":
    main()
