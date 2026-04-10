# 基于 Gemini 2.5 Pro 的代码生成与优化系统 - 增强版

这个项目实现了一个基于 Google Gemini 2.5 Pro 的高级代码生成与优化系统。系统包含两个主要智能代理：

- **Agent A (代码编写者)**：根据任务需求生成完整、可运行的 Python 代码
- **Agent B (代码审查者)**：评估代码并提供具体的优化建议

这两个代理通过轮询方式进行多轮交互，共同完成高质量代码的生成。

## 增强功能

这个增强版本提供了以下主要功能：

1. **自动创建随机命名的输出目录**：在执行目录下自动创建唯一的随机命名文件夹
2. **多文件代码生成支持**：能够识别并保存多个代码文件，而不仅仅是单一脚本
3. **版本控制**：为每一轮优化创建独立的版本目录
4. **完整历史记录**：生成详细的 JSON 历史记录，包含随机文件夹名称和所有生成的文件路径
5. **最终版本**：自动创建一个独立的 `final_version` 目录，包含最终优化的代码

## 系统特点

- 直接使用 Google Generative AI SDK，无需依赖其他框架
- 使用 Google 的 Gemini 2.5 Pro (gemini-2.5-pro-exp-03-25) 模型
- 支持多轮代码生成和优化交互
- 提供结构化的反馈和评估格式
- 预定义多种代码生成任务
- 支持自定义任务复杂度和评估深度

## 安装依赖

只需安装 Google Generative AI 库：

```bash
pip install google-generativeai=="0.7.1"
```

## 环境变量设置

在运行脚本前，请确保设置了 Gemini API 密钥：

```bash
# Linux/Mac
export GEMINI_API_KEY='your-api-key-here'

# Windows (CMD)
set GEMINI_API_KEY=your-api-key-here

# Windows (PowerShell)
$env:GEMINI_API_KEY='your-api-key-here'
```

## 使用方法

### 基本使用

```bash
python gemini_agents_enhanced.py
```

### 高级选项

```bash
# 使用预定义任务
python gemini_agents_enhanced.py --task_name data_analyzer

# 自定义任务
python gemini_agents_enhanced.py --task "请编写一个图像处理程序，支持基本的滤镜和转换..."

# 自定义轮数和复杂度
python gemini_agents_enhanced.py --rounds 3 --task_complexity advanced --review_depth detailed
```

## 预定义任务

系统包含以下预定义任务：

1. **data_analyzer**: 数据分析工具，提供统计、相关性分析和可视化
2. **web_scraper**: 网页爬虫，支持结构化数据提取和保存
3. **api_server**: FastAPI 服务器实现，包含 CRUD 操作和身份验证

## 输出目录结构

执行后，系统会创建如下结构的目录：

```
generated_code_YYYYMMDD_HHMMSS_randomID/
├── version_0_initial/          # 初始生成的代码
│   ├── file1.py
│   └── file2.py
├── version_1_optimized/        # 第1轮优化后的代码
│   ├── file1.py
│   └── file2.py
├── version_2_optimized/        # 第2轮优化后的代码
│   ├── file1.py
│   └── file2.py
├── final_version/              # 最终版本的代码
│   ├── file1.py
│   └── file2.py
├── generation_history.json     # 简化版历史记录
└── detailed_generation_history.json  # 详细历史记录
```

## JSON 历史记录

系统会生成两种 JSON 历史记录文件：

1. **generation_history.json**: 简化版本，包含主要结构和文件名信息
2. **detailed_generation_history.json**: 详细版本，包含完整对话历史和代码内容

JSON 结构示例：

```json
{
  "task": "任务描述...",
  "model": "gemini-2.5-pro-exp-03-25",
  "temperature": 0.3,
  "rounds": 2,
  "task_complexity": "standard",
  "review_depth": "standard",
  "output_directory": "/absolute/path/to/generated_code_20250426_123456_abcd1234",
  "file_summary": {
    "initial_files": ["file1.py", "file2.py"],
    "final_files": ["file1.py", "file2.py"],
    "iterations": [
      {
        "round": 1,
        "optimized_files": ["file1.py", "file2.py"]
      },
      {
        "round": 2,
        "optimized_files": ["file1.py", "file2.py"]
      }
    ]
  }
}
```

## 命令行参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--task` | 自定义任务描述 | 无 |
| `--task_name` | 预定义任务名称 (data_analyzer, web_scraper, api_server) | 无 (默认使用 data_analyzer) |
| `--model` | Gemini 模型名称 | gemini-2.5-pro-exp-03-25 |
| `--temperature` | 模型温度参数 (0.0-1.0) | 0.3 |
| `--rounds` | 交互轮数 | 2 |
| `--task_complexity` | 任务复杂度 (standard, advanced) | standard |
| `--review_depth` | 审查深度 (standard, detailed) | standard |
| `--no_save` | 不保存历史记录 | 默认保存 |

## 自定义系统

### 自定义任务

您可以通过传递自定义任务描述来生成特定代码：

```bash
python gemini_agents_enhanced.py --task "您的详细任务描述..."
```

### 调整任务复杂度

任务复杂度影响 Agent A 的系统提示，从而生成不同质量的代码：

```bash
# 标准复杂度
python gemini_agents_enhanced.py --task_complexity standard

# 高级复杂度 (包含更多优化、设计模式和类型提示)
python gemini_agents_enhanced.py --task_complexity advanced
```

### 调整评估深度

评估深度影响 Agent B 的系统提示，决定代码审查的详细程度：

```bash
# 标准评估
python gemini_agents_enhanced.py --review_depth standard

# 详细评估 (包含更深入的性能和安全分析)
python gemini_agents_enhanced.py --review_depth detailed
```

## 注意事项

- 确保 Gemini API 密钥有足够的配额
- 根据任务复杂度，可能需要调整模型参数（如 temperature、max_output_tokens 等）
- 对于大型代码生成任务，可能需要增加交互轮数
- 系统会自动保存所有生成的文件，无需手动复制代码
