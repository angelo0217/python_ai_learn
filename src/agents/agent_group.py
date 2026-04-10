import os

import autogen

X_API_KEY = os.getenv("X_API_KEY")
llm_config = {
    "config_list": {
        "model": "grok-3-beta",
        "api_key": X_API_KEY,
        "base_url": "https://api.x.ai/v1",
    },
    "cache_seed": 42,
}  # cache_seed 用於可重複性

# 1. 定義代理
agent_a_system_message = """你是團隊的協調者 (Coordinator)。
你的任務是：
1. 接收一個主任務。
2. 分析任務，並將其分解為適合 AgentB (數據分析專家) 和 AgentC (報告撰寫專家) 的子任務。
3. 清晰地向 AgentB 和 AgentC 分配子任務，並要求他們完成後向你報告。
4. 收集 AgentB 和 AgentC 的輸出。
5. 整合他們的輸出，形成一個完整、連貫的最終答案或報告。
6. 在你提供最終整合的答案後，以 "TERMINATE" 結束對話。
確保在最終總結前，你已經收到了 AgentB 和 AgentC 的明確回應。
"""
agent_a = autogen.AssistantAgent(
    name="Coordinator_AgentA",
    system_message=agent_a_system_message,
    llm_config=llm_config,
)

agent_b_system_message = """你是數據分析專家 (DataAnalyst_AgentB)。
你會從 Coordinator_AgentA 那裡接收數據分析相關的任務。
請執行分析，並將清晰的分析結果回報給 Coordinator_AgentA。
你的回覆應該直接針對 Coordinator_AgentA。
"""
agent_b = autogen.AssistantAgent(
    name="DataAnalyst_AgentB",
    system_message=agent_b_system_message,
    llm_config=llm_config,
)

agent_c_system_message = """你是報告撰寫專家 (ReportWriter_AgentC)。
你會從 Coordinator_AgentA 那裡接收撰寫報告或文本內容的任務，可能基於 DataAnalyst_AgentB 的分析結果。
請撰寫要求的內容，並將其回報給 Coordinator_AgentA。
你的回覆應該直接針對 Coordinator_AgentA。
"""
agent_c = autogen.AssistantAgent(
    name="ReportWriter_AgentC",
    system_message=agent_c_system_message,
    llm_config=llm_config,
)

# 用戶代理，用於發起請求
user_proxy = autogen.UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",  # 或 "TERMINATE" 或 "ALWAYS"
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config=False,  # 這個場景不需要代碼執行
    # system_message="你將發起一個任務，並等待 Coordinator_AgentA 的最終回覆。" # 可選
)

# 2. 設置群聊
groupchat = autogen.GroupChat(
    agents=[user_proxy, agent_a, agent_b, agent_c],  # user_proxy 也加入，以便發起和結束
    messages=[],
    max_round=15,  # 最大對話輪次
    speaker_selection_method="auto",  # 關鍵！
    # admin_name="Coordinator_AgentA" # 可以考慮設置，讓A在選擇時有更高優先級或作為預設
    send_introductions=True,  # 發送介紹性消息，讓 LLM 了解每個 agent 的角色
)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
)

# 3. 啟動對話
initial_task = "請分析提供的銷售數據（假設數據已給出或 agent B 知道如何獲取），找出增長最快的產品類別，並撰寫一份簡短的市場趨勢報告摘要。"

# 讓 UserProxyAgent 發起對話，並將任務傳遞給 GroupChatManager
user_proxy.initiate_chat(
    manager,
    message=initial_task,
)

# 如果 UserProxyAgent 的 human_input_mode="NEVER"，它會直接將任務發給 manager
# manager.initiate_chat(user_proxy, message=initial_task) # 另一種啟動方式，取決於誰先說話
