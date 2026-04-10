import asyncio
import argparse
import os
import json
import datetime
import sys
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

X_API_KEY = os.getenv("X_API_KEY")


def get_xai_client() -> OpenAIChatCompletionClient:  # type: ignore
    "Mimic OpenAI API using Local LLM Server."
    return OpenAIChatCompletionClient(
        model="grok-3-beta",
        api_key=X_API_KEY,
        base_url="https://api.x.ai/v1",
        model_capabilities={
            "json_output": True,
            "vision": True,
            "function_calling": True,
            "structured_output": True,
        },
    )


def create_output_folder(folder_name="code_review_results"):
    """
    創建用于存儲輸出結果的文件夾

    Args:
        folder_name: 輸出文件夾名稱

    Returns:
        Path對象，指向創建的文件夾
    """
    # 獲取當前日期時間，用於創建唯一的文件夾名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 創建輸出路徑
    output_dir = Path(folder_name) / timestamp

    # 確保文件夾存在
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def save_results_to_file(task_result, output_dir, task):
    """
    將結果保存到文件，並按照專案結構組織

    Args:
        task_result: 任務結果
        output_dir: 輸出目錄
        task: 原始任務描述
    """
    # 創建一個包含所有訊息的列表，用於保存
    conversation = []

    # 將用戶任務添加到對話中
    conversation.append({"role": "user", "content": task})

    # 添加代理之間的對話
    for msg in task_result.messages:
        conversation.append(
            {
                "role": msg.source,
                "content": msg.content,
                "tokens": (
                    {
                        "prompt": getattr(msg.models_usage, "prompt_tokens", 0),
                        "completion": getattr(msg.models_usage, "completion_tokens", 0),
                        "total": getattr(msg.models_usage, "prompt_tokens", 0)
                        + getattr(msg.models_usage, "completion_tokens", 0),
                    }
                    if msg.models_usage
                    else {}
                ),
            }
        )

    # 創建結果摘要
    result_summary = {
        "task": task,
        "stop_reason": task_result.stop_reason,
        "conversation": conversation,
    }

    # 保存JSON格式的完整對話
    with open(output_dir / "full_conversation.json", "w", encoding="utf-8") as f:
        json.dump(result_summary, f, ensure_ascii=False, indent=2)

    # 保存文本格式的對話，便於閱讀
    with open(output_dir / "conversation.txt", "w", encoding="utf-8") as f:
        f.write(f"任務描述：\n{task}\n\n")
        f.write("=" * 60 + "\n")

        for msg in task_result.messages:
            if msg.source == "user":
                f.write("📋 需求描述：\n")
            elif msg.source == "programmer":
                f.write("👨‍💻 開發工程師提交：\n")
            elif msg.source == "code_reviewer":
                f.write("🔍 代碼審查反饋：\n")

            f.write("-" * 40 + "\n")
            f.write(f"{msg.content}\n\n")

            if msg.models_usage:
                f.write(f"Token統計：\n")
                f.write(f"· 提示tokens: {msg.models_usage.prompt_tokens}\n")
                f.write(f"· 生成tokens: {msg.models_usage.completion_tokens}\n")
                f.write(
                    f"· 總計tokens: {msg.models_usage.prompt_tokens + msg.models_usage.completion_tokens}\n\n"
                )

        f.write("=" * 60 + "\n")
        f.write(f"評審結果：{task_result.stop_reason}\n")

    try:
        # 嘗試從開發工程師的最後一條消息中提取代碼
        code_files = extract_code_from_messages(task_result.messages)

        # 創建項目結構目錄
        project_dirs = set()
        for filepath in code_files.keys():
            # 標準化文件路徑，處理可能的反斜槓和移除反引號
            filepath = filepath.replace("\\", "/").replace("`", "")

            # 確保路徑開頭沒有斜槓
            if filepath.startswith("/"):
                filepath = filepath[1:]

            # 拆分路徑並建立目錄結構
            parts = filepath.split("/")
            current_path = output_dir
            for i in range(len(parts) - 1):  # 不包括文件名
                current_path = current_path / parts[i]
                project_dirs.add(current_path)

        # 確保基本目錄結構存在
        base_dirs = [
            output_dir / "backend",
            output_dir / "frontend",
            output_dir / "frontend" / "static",
            output_dir / "frontend" / "static" / "css",
            output_dir / "frontend" / "static" / "js",
        ]
        for dir_path in base_dirs:
            project_dirs.add(dir_path)

        # 確保所有需要的目錄都存在，先創建較短的路徑
        for dir_path in sorted(project_dirs, key=lambda x: len(str(x))):
            dir_path.mkdir(exist_ok=True, parents=True)

        # 保存所有提取的代碼
        saved_files = 0
        for filepath, code in code_files.items():
            # 標準化文件路徑並移除反引號
            filepath = filepath.replace("\\", "/").replace("`", "")
            if filepath.startswith("/"):
                filepath = filepath[1:]

            # 建立完整路徑並保存文件
            full_path = output_dir / filepath

            try:
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(code)
                saved_files += 1
            except Exception as e:
                print(f"保存文件 {filepath} 時出錯: {str(e)}")
                # 嘗試將文件保存到備用位置
                fallback_path = output_dir / f"unknown_{filepath.replace('/', '_')}"
                with open(fallback_path, "w", encoding="utf-8") as f:
                    f.write(code)
                saved_files += 1

        # 創建一個README.md文件，說明專案結構和運行方法
        output_dir_name = str(output_dir.name).replace("`", "")  # 移除反引號
        readme_content = f"""# 代碼審查自動生成專案

## 專案描述
{task}

## 專案結構
```
{output_dir_name}/
├── backend/            # 後端程式碼
├── frontend/           # 前端程式碼
├── conversation.txt    # 審查對話記錄
└── full_conversation.json # 完整對話記錄(JSON格式)
```

## 如何運行
1. 啟動後端服務：
   ```bash
   cd {output_dir_name}/backend
   # 安裝依賴
   pip install -r requirements.txt
   # 運行服務
   uvicorn main:app --reload
   ```

2. 啟動前端頁面：
   ```bash
   cd {output_dir_name}/frontend
   # 可以使用簡單的HTTP伺服器
   python -m http.server 8080
   ```

3. 在瀏覽器中訪問 http://localhost:8080
"""

        with open(output_dir / "README.md", "w", encoding="utf-8") as f:
            f.write(readme_content)

        # 如果沒有requirements.txt，自動生成一個基本的
        if not any(file.endswith("requirements.txt") for file in code_files.keys()):
            with open(output_dir / "backend" / "requirements.txt", "w", encoding="utf-8") as f:
                f.write("fastapi>=0.68.0\nuvicorn>=0.15.0\npython-dotenv>=0.19.0\n")

        return saved_files
    except Exception as e:
        print(f"處理代碼文件時出錯: {str(e)}")
        # 發生錯誤時，嘗試保存至少一個基本的示例文件
        try:
            # 確保基本目錄存在
            backend_dir = output_dir / "backend"
            frontend_dir = output_dir / "frontend"

            # 移除目錄名中可能存在的反引號
            backend_dir = Path(str(backend_dir).replace("`", ""))
            frontend_dir = Path(str(frontend_dir).replace("`", ""))

            backend_dir.mkdir(exist_ok=True)
            frontend_dir.mkdir(exist_ok=True)

            # 創建基本的FastAPI後端
            backend_code = """from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/hello")
def get_hello():
    return {"message": "Hello, World!"}
"""
            with open(backend_dir / "main.py", "w", encoding="utf-8") as f:
                f.write(backend_code)

            # 創建基本的前端HTML
            frontend_code = """<!DOCTYPE html>
<html>
<head>
    <title>Hello World API Demo</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            margin-top: 50px;
        }
        button {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
        }
        #result {
            margin-top: 20px;
            font-size: 24px;
        }
    </style>
</head>
<body>
    <h1>Hello World API Demo</h1>
    <button onclick="fetchHello()">Get Hello Message</button>
    <div id="result"></div>
    
    <script>
        async function fetchHello() {
            try {
                const response = await fetch('http://localhost:8000/hello');
                const data = await response.json();
                document.getElementById('result').textContent = data.message;
            } catch (error) {
                document.getElementById('result').textContent = 'Error: ' + error.message;
            }
        }
    </script>
</body>
</html>
"""
            with open(frontend_dir / "index.html", "w", encoding="utf-8") as f:
                f.write(frontend_code)

            # 創建要求文件
            with open(backend_dir / "requirements.txt", "w", encoding="utf-8") as f:
                f.write("fastapi>=0.68.0\nuvicorn>=0.15.0\n")

            print(f"已創建基本的備用文件，因為在處理原始代碼時發生錯誤: {str(e)}")
            return 3  # 返回創建的備用文件數量
        except Exception as backup_error:
            print(f"創建備用文件時出錯: {str(backup_error)}")
            return 0


def get_file_extension(language):
    """
    根據語言標識符返回適當的文件擴展名

    Args:
        language: 語言標識符

    Returns:
        文件擴展名，包括點號
    """
    if not language:
        return ".txt"

    # 清理可能包含的反引號
    language = language.replace("`", "").lower().strip()

    # 檢查特殊情況
    if "requirements" in language or "req" in language:
        return ".txt"
    elif "markdown" in language or "md" in language:
        return ".md"
    elif "python" in language or language == "py":
        return ".py"
    elif "javascript" in language or language == "js":
        return ".js"
    elif "css" in language or "style" in language:
        return ".css"
    elif "html" in language or language == "htm":
        return ".html"

    # 標準映射
    extension_map = {
        "python": ".py",
        "py": ".py",
        "javascript": ".js",
        "js": ".js",
        "typescript": ".ts",
        "ts": ".ts",
        "java": ".java",
        "c": ".c",
        "cpp": ".cpp",
        "c++": ".cpp",
        "csharp": ".cs",
        "cs": ".cs",
        "php": ".php",
        "ruby": ".rb",
        "rb": ".rb",
        "go": ".go",
        "rust": ".rs",
        "swift": ".swift",
        "kotlin": ".kt",
        "scala": ".scala",
        "html": ".html",
        "css": ".css",
        "json": ".json",
        "xml": ".xml",
        "yaml": ".yaml",
        "yml": ".yml",
        "markdown": ".md",
        "md": ".md",
        "sql": ".sql",
        "bash": ".sh",
        "sh": ".sh",
        "powershell": ".ps1",
        "ps1": ".ps1",
        "r": ".r",
        "dart": ".dart",
        "elixir": ".ex",
        "perl": ".pl",
    }

    # 嘗試精確匹配
    if language in extension_map:
        return extension_map[language]

    # 嘗試部分匹配
    for key, ext in extension_map.items():
        if key in language:
            return ext

    # 預設為文本檔
    return ".txt"


def extract_code_from_messages(messages):
    """
    從消息列表中提取代碼，按照專案結構組織文件

    Args:
        messages: 消息列表

    Returns:
        包含文件名和代碼內容的字典，按照專案結構組織
    """
    code_files = {}

    # 檢查消息中的程式碼塊
    for msg in messages:
        if msg.source == "programmer":
            content = msg.content

            # 嘗試尋找文件路徑標識
            current_file_path = None

            # 尋找所有代碼塊 ```language ... ```
            lines = content.split("\n")
            i = 0
            while i < len(lines):
                line = lines[i]

                # 檢查是否是文件路徑標記 (例如 ### backend/main.py 或 "文件：backend/main.py")
                if (line.startswith("#") and ("/" in line or "." in line)) or (
                    ("文件" in line or "檔案" in line or "file" in line.lower())
                    and (":" in line or "：" in line)
                ):
                    if line.startswith("#"):
                        path_candidate = line.strip("#").strip()
                    else:
                        parts = line.replace("：", ":").split(":", 1)
                        if len(parts) > 1:
                            path_candidate = (
                                parts[1].strip().strip('"').strip("'").strip("`").strip()
                            )

                    # 確認這是一個合理的文件路徑
                    if "/" in path_candidate or "." in path_candidate:
                        current_file_path = path_candidate

                # 檢查代碼塊開始
                if line.startswith("```"):
                    # 提取語言，並標準化處理
                    language_raw = line[3:].strip()

                    # 去除可能的多餘符號
                    language = language_raw.replace("`", "").strip().lower()

                    # 處理一些特殊情況
                    if language == "":
                        language = "unknown"
                    elif language == "python3":
                        language = "python"
                    elif language in ["js", "javascript"]:
                        language = "javascript"
                    elif language in ["html5", "markup"]:
                        language = "html"

                    # 收集代碼塊
                    code_block = []
                    i += 1  # 跳過 ``` 這一行

                    # 繼續收集直到代碼塊結束
                    while i < len(lines) and not lines[i].strip().startswith("```"):
                        code_block.append(lines[i])
                        i += 1

                    # 如果找到完整代碼塊
                    if i < len(lines) and code_block:
                        code_content = "\n".join(code_block)

                        # 判斷文件類型和路徑
                        if current_file_path and (
                            "/" in current_file_path
                            or "\\" in current_file_path
                            or "." in current_file_path
                        ):
                            # 使用已識別的路徑，但確保路徑和內容類型匹配
                            file_path = current_file_path

                            # 檢查擴展名是否與內容類型匹配
                            if file_path.endswith(".py") and not is_backend:
                                # 這可能是誤標識的前端代碼
                                if "<html" in code_content.lower():
                                    file_path = "frontend/index.html"
                                elif (
                                    "{" in code_content
                                    and ":" in code_content
                                    and ";" in code_content
                                ):
                                    file_path = "frontend/static/css/style.css"
                                elif "function" in code_content.lower() or "fetch(" in code_content:
                                    file_path = "frontend/static/js/app.js"
                            elif file_path.endswith((".html", ".htm")) and not is_frontend:
                                # 這可能是誤標識的後端代碼或純文本
                                if is_backend:
                                    file_path = "backend/main.py"

                            # 如果文件路徑中沒有明確的目錄結構，根據類型添加
                            if "/" not in file_path and "\\" not in file_path:
                                if file_path.endswith((".py", ".txt")):
                                    file_path = f"backend/{file_path}"
                                elif file_path.endswith((".html", ".css", ".js")):
                                    file_path = f"frontend/{file_path}"
                                    # 再次進一步細分
                                    if file_path.endswith(".css"):
                                        file_path = (
                                            f"frontend/static/css/{file_path.split('/')[-1]}"
                                        )
                                    elif file_path.endswith(".js"):
                                        file_path = f"frontend/static/js/{file_path.split('/')[-1]}"
                        else:
                            # 自動猜測文件類型和路徑
                            extension = get_file_extension(language)

                            # 根據內容和語言識別文件類型
                            is_backend = (
                                (
                                    language
                                    and ("python" in language.lower() or "py" in language.lower())
                                )
                                or "fastapi" in code_content.lower()
                                or "flask" in code_content.lower()
                                or "django" in code_content.lower()
                                or "import " in code_content.lower()
                                or "def " in code_content.lower()
                                or "@app" in code_content
                                or ("class " in code_content and ":" in code_content)
                            )

                            is_frontend = (
                                (
                                    language
                                    and (
                                        "html" in language.lower()
                                        or "css" in language.lower()
                                        or "js" in language.lower()
                                        or "javascript" in language.lower()
                                    )
                                )
                                or "<html" in code_content.lower()
                                or "<body" in code_content.lower()
                                or "document." in code_content.lower()
                                or "fetch(" in code_content.lower()
                                or "function " in code_content.lower()
                                or "const " in code_content.lower()
                                or "let " in code_content.lower()
                                or "addEventListener" in code_content.lower()
                            )

                            is_requirements = (
                                "requirements.txt" in (current_file_path or "")
                                or (code_content.count("\n") < 10 and "==" in code_content)
                                or (code_content.count("\n") < 10 and ">=" in code_content)
                                or (language and "requirements" in language.lower())
                            )

                            # 根據類型決定文件路徑
                            if is_requirements:
                                file_path = "backend/requirements.txt"
                            elif is_backend:
                                if extension == ".py":
                                    if "test" in code_content.lower():
                                        file_path = "backend/tests/test_api.py"
                                    elif (
                                        "config" in code_content.lower()
                                        or "settings" in code_content.lower()
                                    ):
                                        file_path = "backend/config.py"
                                    else:
                                        file_path = "backend/main.py"
                                else:
                                    file_path = f"backend/unknown{extension}"
                            elif is_frontend:
                                if "<html" in code_content.lower():
                                    file_path = "frontend/index.html"
                                elif (
                                    "{" in code_content
                                    and ":" in code_content
                                    and ";" in code_content
                                    and not "function" in code_content.lower()
                                ):
                                    file_path = "frontend/static/css/style.css"
                                elif (
                                    "function" in code_content.lower()
                                    or "const " in code_content
                                    or "fetch(" in code_content
                                ):
                                    file_path = "frontend/static/js/app.js"
                                elif extension == ".html":
                                    file_path = "frontend/index.html"
                                elif extension == ".css":
                                    file_path = "frontend/static/css/style.css"
                                elif extension == ".js":
                                    file_path = "frontend/static/js/app.js"
                                else:
                                    file_path = f"frontend/unknown{extension}"
                            else:
                                # 對於無法判斷的內容，根據擴展名猜測
                                if extension in [".py", ".rb", ".java", ".scala"]:
                                    file_path = f"backend/unknown{extension}"
                                else:
                                    file_path = f"frontend/unknown{extension}"

                        # 清除檔案名稱中可能的特殊字符和標記
                        # 1. 移除任何前綴如 "文件："、"File:" 等
                        if "文件：" in file_path:
                            file_path = file_path.replace("文件：", "")
                        if "文件:" in file_path:
                            file_path = file_path.replace("文件:", "")
                        if "檔案：" in file_path:
                            file_path = file_path.replace("檔案：", "")
                        if "檔案:" in file_path:
                            file_path = file_path.replace("檔案:", "")
                        if "file:" in file_path.lower():
                            parts = file_path.split(":", 1)
                            if len(parts) > 1:
                                file_path = parts[1]

                        # 2. 移除反引號
                        file_path = file_path.replace("`", "")

                        # 3. 處理數字編號開頭（如 "1. 配置文件"）
                        if any(
                            file_path.startswith(f"{i}. ") or file_path.startswith(f"{i}.")
                            for i in range(1, 10)
                        ):
                            # 找出第一個非數字、非點、非空格的字符的位置
                            for j, char in enumerate(file_path):
                                if not (char.isdigit() or char in "., "):
                                    file_path = file_path[j:]
                                    break

                        # 4. 處理可能的空格、引號和其他不適合文件名的字符
                        file_path = file_path.strip().strip('"').strip("'").strip()

                        code_files[file_path] = code_content
                i += 1  # 繼續處理下一行
                continue

                i += 1  # 如果不是特殊行，正常遞增

    # 確保核心文件存在
    if not any(path.endswith("main.py") for path in code_files.keys()):
        for path, content in list(code_files.items()):
            if path.endswith(".py") and ("fastapi" in content.lower() or "@app" in content):
                code_files["backend/main.py"] = content
                break

    if not any(path.endswith("index.html") for path in code_files.keys()):
        for path, content in list(code_files.items()):
            if path.endswith(".html"):
                code_files["frontend/index.html"] = content
                break

    return code_files


def print_formatted_result(task_result):
    """
    格式化輸出結果到控制台

    Args:
        task_result: 任務結果對象
    """
    print("\n" + "=" * 60)
    print("代碼評審過程".center(60))
    print("=" * 60 + "\n")

    for msg in task_result.messages:
        if msg.source == "user":
            print("📋 需求描述：")
        elif msg.source == "programmer":
            print("👨‍💻 開發工程師提交：")
        elif msg.source == "code_reviewer":
            print("🔍 代碼審查反饋：")

        print("-" * 40)
        print(f"{msg.content}\n")

        if msg.models_usage:
            print(f"Token統計：")
            print(f"· 提示tokens: {msg.models_usage.prompt_tokens}")
            print(f"· 生成tokens: {msg.models_usage.completion_tokens}")
            print(
                f"· 總計tokens: {msg.models_usage.prompt_tokens + msg.models_usage.completion_tokens}\n"
            )

    print("=" * 60)
    print("評審結果：".center(60))
    print("=" * 60)
    print(f"\n{task_result.stop_reason}\n")


async def run_code_review(task):
    """
    運行代碼審查流程

    Args:
        task: 代碼審查任務

    Returns:
        任務結果
    """
    # 檢查API密鑰
    if not X_API_KEY:
        print("錯誤：環境變量 X_API_KEY 未設置。請設置有效的 X.AI API 密鑰。")
        sys.exit(1)

    # 創建 X.AI 模型客戶端
    model_client = get_xai_client()

    # 創建Python開發工程師
    programmer_agent = AssistantAgent(
        "programmer",
        model_client=model_client,
        system_message="""你是一個專業的Python開發工程師。
請基於需求編寫清晰、可維護且符合PEP8規範的Python代碼。
代碼要包含:
- 清晰的注釋和文檔字符串
- 適當的錯誤處理
- 代碼性能優化
- 單元測試
""",
    )

    # 創建代碼審計專家
    code_reviewer_agent = AssistantAgent(
        "code_reviewer",
        model_client=model_client,
        system_message="""你是一位資深的代碼審查專家。請對代碼進行全面的評審,包括:
- 代碼規範性和可讀性
- 設計模式的使用
- 性能和效率
- 安全性考慮
- 測試覆蓋率
- 潛在問題
當代碼符合要求時,回復'同意通過'。""",
    )

    # 定義終止條件:當評論員同意時停止任務
    text_termination = TextMentionTermination("同意通過")

    # 創建一個包含主要智能助手和評論員的團隊
    team = RoundRobinGroupChat(
        [programmer_agent, code_reviewer_agent], termination_condition=text_termination
    )

    # 在脚本中运行时使用 `asyncio.run(...)`
    return await team.run(task=task)


async def main():
    # 解析命令行參數
    parser = argparse.ArgumentParser(description="自動代碼審查系統")
    parser.add_argument("--task", type=str, help="代碼審查任務描述，如果不提供，將使用默認任務")
    parser.add_argument(
        "--output", type=str, default="code_review_results", help="輸出結果的文件夾名稱"
    )
    args = parser.parse_args()

    # 使用默認任務或命令行提供的任務
    task = args.task
    if not task:
        task = """
        請實現一個文件處理類 FileProcessor,要求:
        1. 支持讀取、寫入和追加文本文件
        2. 包含基本的文件統計功能(行數、字符數、單詞數)
        3. 支持文件加密/解密功能
        4. 實現異常處理
        5. 編寫完整的單元測試
        """

    print(f"代碼審查任務：{task}")
    print("開始執行代碼審查...")

    # 運行代碼審查
    result = await run_code_review(task)

    # 創建輸出文件夾
    output_dir = create_output_folder(args.output)
    print(f"創建輸出文件夾：{output_dir}")

    # 保存結果
    saved_files = save_results_to_file(result, output_dir, task)

    # 打印格式化結果
    print_formatted_result(result)

    # 顯示結果保存信息
    print(f"\n結果已保存到文件夾：{output_dir}")
    print(f"保存了 {saved_files} 個代碼文件和對話記錄")


if __name__ == "__main__":
    asyncio.run(main())
