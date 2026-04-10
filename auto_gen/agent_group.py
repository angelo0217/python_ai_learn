from typing import Dict, Any, List, Optional
import autogen
from core.config import settings
from core.logger import logger
from core.exceptions import APIError

class AgentFactory:
    """
    Factory class to create AutoGen agents with standardized configurations.
    """
    
    @staticmethod
    def get_llm_config(model_name: str = "grok-3-beta", base_url: Optional[str] = None) -> Dict[str, Any]:
        \"\"\"
        Generates a standardized LLM configuration.
        \"\"\"
        logger.info(f"Generating LLM config for model: {model_name}")
        return {
            "config_list": [
                {
                    "model": model_name,
                    "api_key": settings.X_API_KEY,
                    "base_url": base_url or "https://api.x.ai/v1",
                }
            ],
            "cache_seed": 42,
        }

    @classmethod
    def create_assistant(cls, name: str, system_message: str, model_name: str = "grok-3-beta") -> autogen.AssistantAgent:
        \"\"\"
        Creates an AssistantAgent with the specified configuration.
        \"\"\"
        try:
            config = cls.get_llm_config(model_name=model_name)
            logger.info(f"Creating agent: {name}")
            return autogen.AssistantAgent(
                name=name,
                system_message=system_message,
                llm_config=config,
            )
        except Exception as e:
            logger.error(f"Failed to create agent {name}: {str(e)}")
            raise APIError(f"Agent creation failed: {str(e)}")

def run_coordinator_workflow(task: str) -> None:
    \"\"\"
    Executes the coordinator-analyst-writer workflow.
    \"\"\"
    # System Messages
    COORD_MSG = \"\"\"你是團隊的協調者 (Coordinator)。
你的任務是：
1. 接收一個主任務。
2. 分析任務，並將其分解為適合 AgentB (數據分析專家) 和 AgentC (報告撰寫專家) 的子任務。
3. 清晰地向 AgentB 和 AgentC 分配子任務，並要求他們完成後向你報告。
4. 收集 AgentB 和 AgentC 的輸出。
5. 整合他們的輸出，形成一個完整、連貫的最終答案或報告。
6. 在你提供最終整合的答案後，以 \"TERMINATE\" 結束對話。
\"\"\"
    ANALYST_MSG = \"\"\"你是數據分析專家 (DataAnalyst_AgentB)。
你會從 Coordinator_AgentA 那裡接收數據分析相關的任務。
請執行分析，並將清晰的分析結果回報給 Coordinator_AgentA。
\"\"\"
    WRITER_MSG = \"\"\"你是報告撰寫專家 (ReportWriter_AgentC)。
你會從 Coordinator_AgentA 那裡接收撰寫報告或文本內容的任務。
請撰寫要求的內容，並將其回報給 Coordinator_AgentA。
\"\"\"

    # Create Agents
    agent_a = AgentFactory.create_assistant("Coordinator_AgentA", COORD_MSG)
    agent_b = AgentFactory.create_assistant("DataAnalyst_AgentB", ANALYST_MSG)
    agent_c = AgentFactory.create_assistant("ReportWriter_AgentC", WRITER_MSG)

    user_proxy = autogen.UserProxyAgent(
        name="User",
        system_message="A human user",
        code_execution_config=False,
        human_input_mode="NEVER",
    )

    groupchat = autogen.GroupChat(
        agents=[user_proxy, agent_a, agent_b, agent_c], 
        messages=[], 
        max_round=12
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=AgentFactory.get_llm_config())

    logger.info(f"Starting workflow for task: {task}")
    user_proxy.initiate_chat(manager, message=task)

if __name__ == "__main__":
    test_task = "分析 2024 年全球 AI 趨勢並撰寫一份簡短報告"
    run_coordinator_workflow(test_task)
