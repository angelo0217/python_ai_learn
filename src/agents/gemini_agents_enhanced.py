"""
基于 Gemini 2.5 Pro的代码生成与优化系统 - 增强版

这个增强版本能够：
1. 自动在执行目录下创建随机命名的文件夹存储生成的代码
2. 将优化后的代码直接保存为Python文件
3. 支持生成多个代码文件
4. 在JSON记录中保存随机文件夹的名称和所有生成的文件路径
"""

# 安装依赖:
# pip install google-generativeai=="0.7.1"

import os
import json
import time
import uuid
import datetime
import argparse
import re
from typing import List, Dict, Any, Optional, Union
import google.generativeai as genai

# 配置 Gemini API 密钥
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("请设置 GEMINI_API_KEY 环境变量")

genai.configure(api_key=GEMINI_API_KEY)

# 预定义代码任务
PREDEFINED_TASKS = {
    "data_analyzer": """
        请编写一个 Python 程序，实现以下功能：
        1. 创建一个名为 'DataAnalyzer' 的类，该类可以读取 CSV 文件并进行基本的数据分析
        2. 实现以下方法：
           - load_data(filepath): 读取 CSV 文件
           - get_summary_statistics(): 返回数据的基本统计信息（均值、中位数、标准差等）
           - find_correlations(): 计算各列之间的相关性
           - plot_distributions(columns): 绘制所选列的直方图
           - export_results(output_path): 将分析结果导出到 JSON 文件
        3. 添加适当的错误处理和日志记录
        4. 提供一个简单的命令行界面
        5. 包含文档字符串和示例用法
    """,
    "web_scraper": """
        请编写一个 Python 程序，实现以下功能：
        1. 创建一个网页爬虫类，可以爬取指定网站的内容
        2. 实现以下功能：
           - 支持 HTTP 和 HTTPS 协议
           - 处理基本的反爬虫机制（例如，User-Agent 轮换、请求延迟等）
           - 提取网页的结构化数据（标题、正文、图片 URL 等）
           - 支持 CSS 选择器和 XPath 进行数据提取
           - 将爬取的数据保存为 JSON 或 CSV 格式
        3. 实现错误处理和重试机制
        4. 使用异步处理提高效率
        5. 包含完整的文档和使用示例
    """,
    "api_server": """
        请编写一个 Python 程序，实现以下功能：
        1. 使用 FastAPI 创建一个 RESTful API 服务器
        2. 实现以下端点：
           - GET /items: 获取所有项目
           - GET /items/{item_id}: 获取特定项目
           - POST /items: 创建新项目
           - PUT /items/{item_id}: 更新项目
           - DELETE /items/{item_id}: 删除项目
        3. 添加资料验证，使用 Pydantic 模型
        4. 实现基本的身份验证（例如 JWT）
        5. 添加请求记录和错误处理
        6. 包含 Swagger/OpenAPI 文档
        7. 提供单元测试
    """,
}


# 定义 Gemini 模型配置
def get_gemini_config(model_name="gemini-2.5-pro-exp-03-25", temperature=0.3):
    return {
        "model": model_name,
        "temperature": temperature,
        "top_p": 1.0,
        "top_k": 32,
        "max_output_tokens": 8192,
    }


# 定义 Agent A 的系统消息
def get_coder_system_message(task_complexity="standard"):
    if task_complexity == "advanced":
        return """你是一位专业的高级 Python 开发人员（Agent A），专门根据任务需求编写高品质、高效能的代码。
        你的职责是根据给定任务编写完整、可运行、优化的 Python 代码。
        
        每次回应必须是完整的代码实现。如果实现需要多个文件，请明确说明每个文件的名称和内容。
        请按照以下格式提供你的回复：
        
        === file: filename1.py ===
        ```python
        # 文件1的完整代码
        ```
        
        === file: filename2.py ===
        ```python
        # 文件2的完整代码
        ```
        
        ... (其他文件，如果有的话)
        
        每个文件必须有描述性的名称，并包含完整的代码（不要使用省略号或不完整的代码片段）。
        
        你应该考虑以下方面：
        1. 代码效率和性能优化
        2. 内存使用优化
        3. 适当的设计模式
        4. 扩展性和可维护性
        5. 全面的错误处理
        6. 完整的类型提示
        7. 详细的文档
        代码应该遵循 PEP 8 风格指南，包含适当的测试。
        """
    else:
        return """你是一位专业的 Python 开发人员（Agent A），专门根据任务需求编写高品质代码。
        你的职责是根据给定任务编写完整、可运行的 Python 代码。
        
        每次回应必须是完整的代码实现。如果实现需要多个文件，请明确说明每个文件的名称和内容。
        请按照以下格式提供你的回复：
        
        === file: filename1.py ===
        ```python
        # 文件1的完整代码
        ```
        
        === file: filename2.py ===
        ```python
        # 文件2的完整代码
        ```
        
        ... (其他文件，如果有的话)
        
        每个文件必须有描述性的名称，并包含完整的代码（不要使用省略号或不完整的代码片段）。
        
        代码应该遵循 PEP 8 风格指南，包含适当的错误处理、文档字符串和注释。
        """


# 定义 Agent B 的系统消息
def get_reviewer_system_message(review_depth="standard"):
    if review_depth == "detailed":
        return """你是一位经验丰富的高级代码审查专家（Agent B），负责深入评估和提供专业优化建议。
        你的职责是:
        1. 全面评估代码的正确性、效率、可读性、风格和架构
        2. 深入分析代码中的潜在问题、漏洞和安全隐患
        3. 检查代码的时间和空间复杂度
        4. 提供具体的优化建议，包括算法改进、性能优化、代码结构重构等
        5. 建议最佳实践和设计模式
        6. 提供详细的代码改进示例
        
        请使用以下格式提供你的详细评估:
        ```
        ## 总体评价 (1-10分)
        [对代码质量的总体评价和评分]
        
        ## 代码质量评估
        ### 正确性: [评分] - [评价]
        ### 效率: [评分] - [评价]
        ### 可读性: [评分] - [评价]
        ### 可维护性: [评分] - [评价]
        ### 架构设计: [评分] - [评价]
        
        ## 优点
        - [优点1]
        - [优点2]
        ...
        
        ## 需要改进的地方
        - [问题1]: [改进建议1]
        - [问题2]: [改进建议2]
        ...
        
        ## 性能与效率分析
        [时间/空间复杂度分析和效率改进建议]
        
        ## 安全性评估
        [安全问题和改进建议]
        
        ## 优化建议代码示例
        === file: filename1.py ===
        ```python
        # 优化后的文件1代码
        ```
        
        === file: filename2.py ===
        ```python
        # 优化后的文件2代码
        ```
        
        ... (其他文件，如果有的话)
        ```
        
        提供优化建议时，请使用与Agent A相同的文件格式，明确指出每个应该优化的文件。
        """
    else:
        return """你是一位经验丰富的代码审查专家（Agent B），负责评估和提供优化建议。
        你的职责是:
        1. 评估代码的正确性、效率、可读性和风格
        2. 找出代码中的潜在问题或漏洞
        3. 提供具体的优化建议，包括性能改进、代码结构、错误处理等
        4. 使用明确的格式提供反馈，以便 Agent A 可以轻松理解和实施改进
        
        请使用以下格式提供你的评估:
        ```
        ## 总体评价
        [对代码质量的总体评价]
        
        ## 优点
        - [优点1]
        - [优点2]
        ...
        
        ## 需要改进的地方
        - [问题1]: [改进建议1]
        - [问题2]: [改进建议2]
        ...
        
        ## 优化建议代码示例
        === file: filename1.py ===
        ```python
        # 优化后的文件1代码
        ```
        
        === file: filename2.py ===
        ```python
        # 优化后的文件2代码
        ```
        
        ... (其他文件，如果有的话)
        ```
        
        提供优化建议时，请使用与Agent A相同的文件格式，明确指出每个应该优化的文件。
        """


class GeminiAgent:
    """基于 Gemini 的智能代理"""

    def __init__(
        self,
        name: str,
        system_message: str,
        model: str = "gemini-2.5-pro-exp-03-25",
        temperature: float = 0.3,
    ):
        """初始化代理

        Args:
            name: 代理名称
            system_message: 系统提示
            model: Gemini 模型名称
            temperature: 模型温度参数 (0.0-1.0)
        """
        self.name = name
        self.system_message = system_message
        self.model = model
        self.temperature = temperature
        self.chat_history = []

        # 初始化模型
        generation_config = {
            "temperature": temperature,
            "top_p": 1.0,
            "top_k": 32,
            "max_output_tokens": 8192,
        }
        self.gemini_model = genai.GenerativeModel(
            model_name=model, generation_config=generation_config
        )

    def send_message(self, message: str) -> str:
        """向代理发送消息并获取回应

        Args:
            message: 输入消息

        Returns:
            代理的回应
        """
        # 构建完整提示，包含系统消息和历史
        full_prompt = f"{self.system_message}\n\n"

        # 添加历史记录
        for msg in self.chat_history:
            if msg["role"] == "user":
                full_prompt += f"用户: {msg['content']}\n\n"
            else:
                full_prompt += f"助手: {msg['content']}\n\n"

        # 添加当前消息
        full_prompt += f"用户: {message}\n\n助手: "

        try:
            # 调用 Gemini API
            response = self.gemini_model.generate_content(full_prompt)
            response_text = response.text

            # 保存对话历史
            self.chat_history.append({"role": "user", "content": message})
            self.chat_history.append({"role": "assistant", "content": response_text})

            return response_text
        except Exception as e:
            print(f"API 调用出错: {e}")
            return f"发生错误: {e}"


def extract_code_files(text: str) -> dict:
    """从代理回应中提取多个代码文件

    Args:
        text: 包含代码文件的文本

    Returns:
        字典，键为文件名，值为代码内容
    """
    # 匹配 "=== file: filename.py ===" 格式的模式
    pattern = r"===\s*file:\s*([^\s=]+)\s*===\s*```(?:python)?\s*(.*?)```"

    # 使用正则表达式查找所有匹配项，使用 re.DOTALL 标志匹配包括换行符在内的所有字符
    matches = re.findall(pattern, text, re.DOTALL)

    # 创建文件名到代码内容的映射
    files = {}
    for filename, code in matches:
        # 清理文件名和代码内容
        filename = filename.strip()
        code = code.strip()
        files[filename] = code

    # 如果没有找到匹配项，查找单个代码块
    if not files:
        single_code_pattern = r"```(?:python)?\s*(.*?)```"
        code_matches = re.findall(single_code_pattern, text, re.DOTALL)
        if code_matches:
            # 使用默认文件名
            files["main.py"] = code_matches[0].strip()

    return files


def save_code_to_files(files: dict, output_dir: str) -> List[str]:
    """将代码保存到指定目录的文件中

    Args:
        files: 字典，键为文件名，值为代码内容
        output_dir: 输出目录路径

    Returns:
        保存的文件路径列表
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    saved_files = []
    for filename, code in files.items():
        file_path = os.path.join(output_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
        saved_files.append(file_path)
        print(f"已保存文件: {file_path}")

    return saved_files


def create_output_directory() -> str:
    """创建随机命名的输出目录

    Returns:
        创建的目录路径
    """
    # 创建一个基于时间戳和UUID的唯一目录名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = str(uuid.uuid4())[:8]  # 使用UUID的前8个字符作为随机后缀
    dir_name = f"generated_code_{timestamp}_{random_suffix}"

    # 在当前工作目录中创建此目录
    os.makedirs(dir_name, exist_ok=True)

    return os.path.abspath(dir_name)


class CodeGenerationSystem:
    """代码生成与优化系统"""

    def __init__(
        self,
        task=None,
        task_name=None,
        model_name="gemini-2.5-pro-exp-03-25",
        temperature=0.3,
        rounds=2,
        task_complexity="standard",
        review_depth="standard",
        save_history=True,
    ):
        """初始化代码生成系统

        Args:
            task: 自定义任务描述
            task_name: 预定义任务的名称
            model_name: 使用的模型名称
            temperature: 模型温度参数
            rounds: 交互轮数
            task_complexity: 任务复杂度 ("standard" 或 "advanced")
            review_depth: 审查深度 ("standard" 或 "detailed")
            save_history: 是否保存历史记录
        """
        self.model_name = model_name
        self.temperature = temperature
        self.rounds = rounds
        self.save_history = save_history
        self.task_complexity = task_complexity
        self.review_depth = review_depth

        # 确定任务
        if task_name and task_name in PREDEFINED_TASKS:
            self.task = PREDEFINED_TASKS[task_name]
        elif task:
            self.task = task
        else:
            self.task = PREDEFINED_TASKS["data_analyzer"]  # 默认任务

        # 初始化代理
        self.coder_agent = GeminiAgent(
            name="CodeWriter",
            system_message=get_coder_system_message(task_complexity),
            model=model_name,
            temperature=temperature,
        )

        self.reviewer_agent = GeminiAgent(
            name="CodeReviewer",
            system_message=get_reviewer_system_message(review_depth),
            model=model_name,
            temperature=temperature,
        )

        # 创建随机目录用于存储生成的代码
        self.output_directory = create_output_directory()
        print(f"创建输出目录: {self.output_directory}")

        # 用于保存结果
        self.results = {
            "task": self.task,
            "model": model_name,
            "temperature": temperature,
            "rounds": rounds,
            "task_complexity": task_complexity,
            "review_depth": review_depth,
            "output_directory": self.output_directory,
            "iterations": [],
        }

    def run(self):
        """执行代码生成和审查的互动循环"""
        print(f"开始代码生成和优化流程 ({self.rounds} 轮)...")

        # 第一步：代码编写者生成初始代码
        print("第 0 步: 生成初始代码...")
        initial_code_response = self.coder_agent.send_message(self.task)

        # 提取并保存初始代码
        initial_files = extract_code_files(initial_code_response)
        if not initial_files:
            print("警告: 无法从初始回应中提取代码文件")
            initial_files = {"main.py": initial_code_response}  # 保存原始回应作为备份

        # 保存初始代码到初始版本子目录
        initial_code_dir = os.path.join(self.output_directory, "version_0_initial")
        initial_file_paths = save_code_to_files(initial_files, initial_code_dir)

        # 记录初始代码信息
        self.results["initial_code"] = {
            "response": initial_code_response,
            "files": initial_files,
            "directory": initial_code_dir,
            "file_paths": initial_file_paths,
        }

        # 进行多轮交互
        for round_num in range(1, self.rounds + 1):
            print(f"\n--- 第 {round_num} 轮交互 ---")

            # 构建审查消息
            if round_num == 1:
                file_text = ""
                for filename, code in initial_files.items():
                    file_text += f"\n=== file: {filename} ===\n```python\n{code}\n```\n"
                review_message = f"请评估以下代码并提供优化建议：\n{file_text}"
            else:
                prev_optimized_files = self.results["iterations"][-1]["optimized_code"]["files"]
                file_text = ""
                for filename, code in prev_optimized_files.items():
                    file_text += f"\n=== file: {filename} ===\n```python\n{code}\n```\n"
                review_message = f"请评估以下代码并提供优化建议：\n{file_text}"

            # 发送给代码审查者进行评估
            print(f"步骤 {round_num}.1: 代码审查...")
            review_response = self.reviewer_agent.send_message(review_message)

            # 代码编写者根据评估进行优化
            print(f"步骤 {round_num}.2: 代码优化...")
            optimize_message = (
                f"根据以下审查反馈优化你的代码 (第 {round_num} 轮)：\n\n{review_response}"
            )
            optimized_code_response = self.coder_agent.send_message(optimize_message)

            # 提取并保存优化后的代码
            optimized_files = extract_code_files(optimized_code_response)
            if not optimized_files:
                print(f"警告: 无法从第 {round_num} 轮优化回应中提取代码文件")
                optimized_files = {"main.py": optimized_code_response}  # 保存原始回应作为备份

            # 创建版本目录并保存代码
            optimized_code_dir = os.path.join(
                self.output_directory, f"version_{round_num}_optimized"
            )
            optimized_file_paths = save_code_to_files(optimized_files, optimized_code_dir)

            # 保存本轮结果
            self.results["iterations"].append(
                {
                    "round": round_num,
                    "review": {"response": review_response},
                    "optimized_code": {
                        "response": optimized_code_response,
                        "files": optimized_files,
                        "directory": optimized_code_dir,
                        "file_paths": optimized_file_paths,
                    },
                }
            )

            print(f"第 {round_num} 轮完成，代码已保存至: {optimized_code_dir}")

        # 最后审查
        print("\n--- 最终评估 ---")
        final_files = self.results["iterations"][-1]["optimized_code"]["files"]
        file_text = ""
        for filename, code in final_files.items():
            file_text += f"\n=== file: {filename} ===\n```python\n{code}\n```\n"
        final_review_message = f"请对最终优化后的代码进行评估：\n{file_text}"

        final_review_response = self.reviewer_agent.send_message(final_review_message)
        self.results["final_review"] = final_review_response

        # 创建最终版本目录，复制最后一轮优化的文件
        final_code_dir = os.path.join(self.output_directory, "final_version")
        final_file_paths = save_code_to_files(final_files, final_code_dir)
        self.results["final_code"] = {
            "files": final_files,
            "directory": final_code_dir,
            "file_paths": final_file_paths,
        }

        print("\n--- 代码生成和优化流程完成 ---")
        print(f"所有生成的代码已保存到目录: {self.output_directory}")
        print(f"最终版本代码位于: {final_code_dir}")

        # 保存结果记录到输出目录
        if self.save_history:
            self._save_history()

        return self.results

    def _save_history(self):
        """保存互动历史记录到输出目录"""
        history_file = os.path.join(self.output_directory, "generation_history.json")
        with open(history_file, "w", encoding="utf-8") as f:
            # 创建一个简化版本的结果，去除大型文本内容
            simplified_results = self._simplify_results_for_json()
            json.dump(simplified_results, f, ensure_ascii=False, indent=2)

        # 创建一个详细的记录文件
        detailed_history_file = os.path.join(
            self.output_directory, "detailed_generation_history.json"
        )
        with open(detailed_history_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        print(f"历史记录已保存到: {history_file}")
        print(f"详细历史记录已保存到: {detailed_history_file}")

    def _simplify_results_for_json(self):
        """创建一个简化版本的结果，适合保存为JSON"""
        simplified = {
            "task": (
                self.task[:200] + "..." if len(self.task) > 200 else self.task
            ),  # 截断长任务描述
            "model": self.model_name,
            "temperature": self.temperature,
            "rounds": self.rounds,
            "task_complexity": self.task_complexity,
            "review_depth": self.review_depth,
            "output_directory": self.output_directory,
            "file_summary": {
                "initial_files": list(self.results.get("initial_code", {}).get("files", {}).keys()),
                "final_files": list(self.results.get("final_code", {}).get("files", {}).keys()),
                "iterations": [],
            },
        }

        # 添加迭代信息
        for iteration in self.results.get("iterations", []):
            simplified["file_summary"]["iterations"].append(
                {
                    "round": iteration.get("round"),
                    "optimized_files": list(
                        iteration.get("optimized_code", {}).get("files", {}).keys()
                    ),
                }
            )

        return simplified


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="基于 Gemini 的代码生成与优化系统")

    # 基本参数
    parser.add_argument("--task", type=str, help="自定义任务描述")
    parser.add_argument(
        "--task_name", type=str, choices=list(PREDEFINED_TASKS.keys()), help="预定义任务名称"
    )

    # 模型参数
    parser.add_argument(
        "--model", type=str, default="gemini-2.5-pro-exp-03-25", help="使用的模型名称"
    )
    parser.add_argument("--temperature", type=float, default=0.3, help="模型温度参数 (0.0-1.0)")

    # 系统参数
    parser.add_argument("--rounds", type=int, default=2, help="交互轮数")
    parser.add_argument(
        "--task_complexity",
        type=str,
        choices=["standard", "advanced"],
        default="standard",
        help="任务复杂度",
    )
    parser.add_argument(
        "--review_depth",
        type=str,
        choices=["standard", "detailed"],
        default="standard",
        help="审查深度",
    )
    parser.add_argument(
        "--no_save", action="store_false", dest="save_history", help="不保存历史记录"
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 创建代码生成系统
    system = CodeGenerationSystem(
        task=args.task,
        task_name=args.task_name,
        model_name=args.model,
        temperature=args.temperature,
        rounds=args.rounds,
        task_complexity=args.task_complexity,
        review_depth=args.review_depth,
        save_history=args.save_history,
    )

    # 运行系统
    results = system.run()

    print("\n代码生成和优化流程完成！")
    print(f"所有文件已保存到目录: {results['output_directory']}")

    return results


if __name__ == "__main__":
    main()
