# AutoGen 代码生成评估系统

这个项目使用 AutoGen 框架实现了一个代码生成与评估系统，由两个智能体协作完成：

1. **Agent A（代码生成器）**：根据任务描述自动生成高质量的代码
2. **Agent B（代码评估器）**：评估代码质量，提供优化建议

两个智能体通过多轮交互完成代码优化，形成一个完整的代码生成-评估-优化流程。

## 安装依赖

首先，请安装所需依赖：

```bash
pip install -r requirements.txt
```

或者直接安装以下依赖：

```bash
pip install python-dotenv>=1.0.0 ag2>=0.2.18 google-generativeai>=0.3.1 matplotlib>=3.8.0 pandas>=2.1.1 numpy>=1.26.0
```

## 环境配置

1. 复制`.env.example`文件为`.env`
2. 编辑`.env`文件，添加您的API密钥：

```
X_API_KEY=your_api_key_here
```

## 文件说明

本项目包含以下主要文件：

- `enhanced_code_generation_evaluation.py`：增强版代码生成评估系统，提供多文件支持
- `advanced_config.py`：支持高级配置和自定义的版本，多语言、多文件支持
- `examples.py`：展示不同任务场景的示例
- `requirements.txt`：依赖列表
- `.env.example`：环境变量示例文件

## 两个主要文件的差异

本系统提供了两个不同复杂度和功能层次的实现版本：

### `enhanced_code_generation_evaluation.py` vs `advanced_config.py`

#### 功能特性比较

| 特性 | enhanced_code_generation_evaluation | advanced_config |
|------|-------------------------------------|-----------------|
| 基本代码生成 | ✓ | ✓ |
| 代码评估 | ✓ | ✓ |
| 代码优化 | ✓ | ✓ |
| 详细日志 | 增强 | 增强 |
| 多语言支持 | 仅Python | 多种语言 |
| 复杂度设置 | 固定 | 可配置 |
| 命令行参数 | 无 | ✓ |
| 迭代次数控制 | 固定2轮 | 可配置 |
| 文件输出 | 增强 | 高度定制 |
| 多文件支持 | 支持 | 增强支持 |
| 迭代文件夹 | ✓ | ✓ |

#### 使用场景

- **enhanced_code_generation_evaluation.py**：
  - 快速测试多文件生成功能
  - Python代码生成的简单任务
  - 不需要额外配置的情况

- **advanced_config.py**：
  - 需要生成多种语言代码的场景
  - 需要通过命令行控制参数的情况
  - 需要不同代码复杂度设置的用户
  - 需要更多迭代轮次的复杂任务

#### 如何选择

- 如果需要快速生成Python多文件项目：使用`enhanced_code_generation_evaluation.py`
- 如果需要高度定制化和多语言支持：使用`advanced_config.py`

## 使用方法

### 增强版代码生成评估系统

运行增强版代码生成评估系统：

```bash
python enhanced_code_generation_evaluation.py
```

### 高级配置

高级配置支持多种编程语言、不同复杂度级别和多轮迭代：

```bash
python advanced_config.py --language python --complexity medium --iterations 2
```

可用参数：

- `--language`：编程语言（python, javascript, java, cpp, go）
- `--complexity`：代码复杂度（simple, medium, complex）
- `--iterations`：优化迭代次数
- `--task`：直接在命令行提供任务描述
- `--task_file`：从文件读取任务描述
- `--output_dir`：结果输出目录

示例：

```bash
# 使用JavaScript生成简单复杂度的代码，进行3轮迭代
python advanced_config.py --language javascript --complexity simple --iterations 3

# 从文件读取任务描述
python advanced_config.py --task_file tasks/my_task.txt --language python

# 直接提供任务描述
python advanced_config.py --task "实现一个简单的天气预报应用，可以获取和显示特定城市的未来5天天气预报"
```

## 多文件项目支持

系统支持生成包含多个文件的项目：

1. 使用增强版代码生成评估系统（`enhanced_code_generation_evaluation.py`）或高级配置版本（`advanced_config.py`）
2. 在任务描述中明确需要多个文件的结构
3. 所有生成的文件会保存在一个以时间戳命名的子目录中

您可以使用以下两种方式之一创建多文件项目：

### 使用 enhanced_code_generation_evaluation.py

```bash
python enhanced_code_generation_evaluation.py
```

### 使用 advanced_config.py (推荐)

```bash
python advanced_config.py --task "实现一个简单的Web应用程序，包含前端HTML和后端FastAPI服务器，需处理CORS问题"
```

advanced_config.py 提供了更灵活的选项：
- 支持更多编程语言（不仅是Python）
- 提供复杂度选项（simple, medium, complex）
- 允许自定义迭代次数

示例任务描述（多文件项目）：

```
实现一个简单的Web应用程序，包含：
1. 一个Flask服务器文件
2. 一个SQLite数据库访问模块
3. HTML模板文件
4. 配置文件
5. 实用工具模块
```

或者更具体的FastAPI示例：

```
后端使用Python FastAPI实现一个Hello World API，前端使用HTML/JavaScript调用此API，并妥善处理CORS跨域问题
```

## 输出结果

系统的输出结果包括：

1. 控制台日志：显示整个工作流程的进度
2. 项目目录：按时间戳创建子目录，格式为 `project_YYYYMMDD_HHMMSS`
3. 迭代文件夹：每个迭代版本的代码和评估保存在单独的编号文件夹中（如 `1_initial_code`、`2_optimized_code` 等）
4. 源代码文件：所有生成的源代码文件，最终版本同时保存在根目录和最后一个迭代文件夹中
5. 文档文件：包含项目说明、代码评估和参考的Markdown文件

### 迭代文件夹结构

系统现在为每次代码迭代和评估创建单独的编号文件夹：

```
results/
  └── project_20250428_123456/            # 按时间戳命名的项目目录
      ├── 1_initial_code/                 # 第一次迭代 - 初始代码
      │   ├── main.py                     # 初始版本代码
      │   ├── index.html                  # 初始版本HTML（如有）
      │   └── code_review.md              # 初始代码评估结果
      ├── 2_optimized_code/               # 第二次迭代 - 优化后代码
      │   ├── main.py                     # 优化后代码
      │   ├── index.html                  # 优化后HTML（如有）
      │   └── code_review.md              # 最终代码评估结果
      ├── main.py                         # 最终版本代码（与最后迭代相同）
      ├── index.html                      # 最终版本HTML（与最后迭代相同）
      └── documentation.md                # 完整项目文档（包含所有迭代信息）
```

这种结构使您可以：
- 清晰地追踪代码的演化过程
- 比较不同迭代版本的代码变化
- 查看每个迭代的评估和反馈

## 常见问题

1. **API密钥问题**：确保您有有效的API密钥，并正确设置在`.env`文件中
2. **依赖问题**：确保安装了正确版本的依赖，特别是ag2
3. **多文件识别问题**：如果系统未正确识别多个文件，请在任务描述中明确指出需要多个独立文件

## 性能优化

- 调整`temperature`参数可以控制代码生成的创造性和稳定性
- 增加迭代次数可获得更优质的代码，但会增加API调用次数
- 对于复杂任务，建议使用`complexity=complex`参数获得更完善的实现
- 对于多文件项目，考虑添加明确的文件结构描述以获得更好的结果

---

希望这个指南能帮助您充分利用AutoGen代码生成评估系统。如有任何问题，请参考代码中的注释或查看相关文档。
