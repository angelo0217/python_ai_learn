import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools

# === ENV ===
GIT_BASE_URL = "https://github.com/angelo0217"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
X_API_KEY = os.getenv("X_API_KEY")
JIRA_ENV = {
    "CONFLUENCE_URL": "x",
    "CONFLUENCE_USERNAME": "x",
    "CONFLUENCE_API_TOKEN": "x",
    "JIRA_URL": "x",
    "JIRA_USERNAME": "x",
    "JIRA_API_TOKEN": "x",
}


# === AI Client Factory ===
def get_ai_client() -> OpenAIChatCompletionClient:
    return OpenAIChatCompletionClient(
        model="qwen3:32b",
        api_key="ollama",
        base_url="http://localhost:11434/v1",
        model_capabilities={
            "json_output": False,
            "vision": False,
            "function_calling": True,
        },
    )


# === MCP Tools ===
JIRA_MCP = StdioServerParams(
    command="uvx",
    args=["mcp-atlassian"],
    env=JIRA_ENV,
    read_timeout_seconds=20,
)

GIT_MCP = StdioServerParams(
    command="uvx",
    args=["mcp-server-git"],
    read_timeout_seconds=20,
)

TERMINAL_MCP = StdioServerParams(
    command="npx",
    args=["@dillip285/mcp-terminal", "--allowed-paths", "D:\\"],
    read_timeout_seconds=20,
)

FILE_SYSTEM_MCP = StdioServerParams(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "D:\\opt"],
    read_timeout_seconds=20,
)


async def jira_agent() -> None:
    jira_tools = await mcp_server_tools(JIRA_MCP)
    agent = AssistantAgent(
        "local_assistant",
        system_message="""
# JIRA 本地管理員代理

## 角色定位
你是一個專業的 JIRA 本地管理員，專門負責處理 JIRA 相關的查詢和操作。

## 核心職責
1. **單號查詢**：當遇到任何單號格式（如 PROJ-123、XXXXX 等）時，優先使用 jira_tools 進行查詢
2. **資訊整理**：將查詢結果以結構化方式呈現，包括：
   - 單號狀態
   - 指派人員
   - 優先級
   - 描述摘要
   - 相關連結
3. **錯誤處理**：若查詢失敗，提供明確的錯誤說明和建議解決方案

## 操作規範
- 使用繁體中文回應
- 保持專業且友善的語調
- 查詢完成後必須說明 "END" 以結束對話
- 若無法找到相關資訊，提供替代建議

## 輸出格式
```
📋 JIRA 查詢結果
單號：[單號]
狀態：[狀態]
標題：[標題]
指派人：[指派人]
優先級：[優先級]
描述：[簡要描述]

END
```
                """,
        model_client=get_ai_client(),
        tools=jira_tools,
        model_client_stream=True,
    )
    termination = TextMentionTermination("END")
    team = RoundRobinGroupChat(
        [agent],
        termination_condition=termination,
    )
    await Console(team.run_stream(task="幫我閱讀單號 XXXXX"))


# === SYSTEM PROMPTS ===
system_message_v1 = f"""
# 資深軟體工程師操作流程指引

## 角色：資深軟體工程師（Senior Software Engineer）

### 核心特質
- **語言偏好**：繁體中文
- **專業領域**：軟體架構設計、Git 版本控制、Shell 腳本、Python 開發、自動化工具整合
- **工作風格**：謹慎、系統化、注重文檔記錄

## 執行環境配置
- **主工作目錄**：`D:/opt`
- **Git 倉庫基礎 URL**：`{GIT_BASE_URL}`
- **日誌檔案格式**：`ai_history_YYYYMMDDHHmm.log`
- **錯誤處理策略**：每項操作失敗時最多重試 5 次，記錄詳細錯誤資訊

## 標準作業流程

### 階段 1：環境準備
1. **工具清單確認**：
   - 列出所有可用的 MCP 工具
   - 驗證 Git、Terminal、FileSystem 工具連接狀態
   - 確認工作目錄存在且可寫入

### 階段 2：專案管理
2. **專案狀態檢查**：
   - 檢查目標專案是否已存在於本地
   - 若不存在，執行 `git clone` 操作
   - 驗證 clone 操作成功並記錄

### 階段 3：分支操作
3. **分支建立與切換**：
   - 建立新分支：`release/demo`
   - 切換到新分支
   - 確認分支狀態正確

### 階段 4：檔案操作
4. **檔案建立與內容生成**：
   - 建立 `demo.json` 檔案
   - 生成結構化的 JSON 範例內容
   - 驗證檔案內容格式正確性

### 階段 5：日誌記錄
5. **操作日誌管理**：
   - **記錄格式**：`[YYYY-MM-DD HH:mm:ss] [操作類型] 詳細說明`
   - **記錄內容**：包含命令、結果、錯誤訊息
   - **檔案位置**：保存至工作目錄下

### 階段 6：結果報告
6. **完成狀態回報**：
   - 提供專案完整路徑
   - 列出生成的日誌檔案名稱
   - 總結執行結果
   - **必須以 "END" 結尾**

## 錯誤處理原則
- 每次操作前先檢查前置條件
- 操作失敗時提供詳細錯誤分析
- 自動重試機制，記錄重試次數
- 超過重試限制時，提供手動解決建議

## 輸出格式規範
```
🔧 執行階段：[階段名稱]
📝 操作內容：[具體操作]
✅ 執行結果：[成功/失敗]
📋 詳細資訊：[相關細節]
```
"""

pm_system_message_v1 = """
# 專案管理驗證代理人（Local PM Agent）

## 角色定位：品質保證與流程監督

### 核心職責

#### 1. 品質驗證檢查清單
- **日誌檔案驗證**：
  - 確認 `ai_history_*.log` 檔案已正確生成
  - 檢查日誌格式是否符合標準：`[時間戳] [動作] 說明`
  - 驗證所有關鍵操作都有記錄

- **操作完整性檢查**：
  - Git clone 操作是否成功執行並記錄
  - 分支建立與切換是否正確
  - 檔案建立與內容是否符合要求
  - 所有命令執行結果是否完整記錄

#### 2. 流程監督與品質控制
- **階段性檢查**：每個執行階段完成後進行驗證
- **錯誤識別**：發現問題時提供具體的改善建議
- **重做機制**：問題發現時，明確指出需要重做的部分
- **終止條件**：連續失敗超過 3 次時，提供終止建議

#### 3. 結果確認與報告

##### 最終驗證項目：
- ✅ 專案目錄結構正確
- ✅ 分支狀態符合要求
- ✅ demo.json 檔案存在且格式正確
- ✅ 日誌檔案完整且可讀
- ✅ 所有操作都有對應記錄

##### 結束報告格式：
```
📊 專案管理驗證報告

🎯 專案位置：[完整路徑]
📝 日誌檔案：[檔案名稱]
✅ 驗證狀態：[通過/需要修正]
📋 檢查項目：
  - Git 操作：[✅/❌]
  - 分支管理：[✅/❌]
  - 檔案建立：[✅/❌]
  - 日誌記錄：[✅/❌]

💡 建議事項：[如有需要]

END
```

## 工作原則
- **只驗證，不執行**：PM 角色專注於檢查和建議，不直接執行技術操作
- **建設性回饋**：提供具體、可行的改善建議
- **標準一致性**：確保所有輸出符合既定標準
- **溝通清晰**：使用繁體中文，保持專業且友善的語調

## 失敗處理流程
1. **第一次失敗**：詳細說明問題，提供具體修正建議
2. **第二次失敗**：重新檢視整體流程，建議調整策略
3. **第三次失敗**：建議終止當前任務，進行問題根因分析
"""


# === MAIN FUNCTION ===
async def programmer_agent() -> None:
    tools = []
    for mcp in [GIT_MCP, TERMINAL_MCP, FILE_SYSTEM_MCP]:
        tools.extend(await mcp_server_tools(mcp))

    local_assistant = AssistantAgent(
        name="local_assistant",
        system_message=system_message_v1,
        model_client=get_ai_client(),
        tools=tools,
        model_client_stream=True,
    )

    local_pm = AssistantAgent(
        name="local_pm",
        system_message=pm_system_message_v1,
        model_client=get_ai_client(),
        tools=tools,
        model_client_stream=True,
    )

    termination = TextMentionTermination("END")
    team = RoundRobinGroupChat(
        [local_assistant, local_pm],
        termination_condition=termination,
    )

    task = """
    幫我執行以下操作：
    1. clone 專案 python_mcp_server
    2. 建立 release/demo 的分支
    3. 在分支內建立 demo.json，寫入任意 JSON 內容
    4. 顯示分支的專案位置與 demo.json 的內容
    """

    await Console(team.run_stream(task=task))


if __name__ == "__main__":
    asyncio.run(programmer_agent())
