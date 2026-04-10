import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams, SseServerParams, mcp_server_tools

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
            "structured_output": True,  # Added missing field
        },
    )


STORE_MCP_PARAMS = StdioServerParams(
    command="python",
    args=["store_count_mcp.py"],  # Fixed typo in filename
    read_timeout_seconds=10,
)

DB_MCP_PARAMS = SseServerParams(url="http://localhost:8082/sse", headers={})

# server_params = StdioServerParams(
#     command="mcp-proxy",
#     args=["http://localhost:8082/sse"],
#     read_timeout_seconds=10,
# )


async def work_bench_demo():
    # 目前似乎只支援一組mcp，還在研究
    async with McpWorkbench(DB_MCP_PARAMS) as mcp:
        tools = await mcp.list_tools()
        tool_names = [tool["name"] for tool in tools]
        print(tool_names)

        agent = AssistantAgent(
            "local_assistant",
            system_message="""
            你是一個本地管理員，專注於處裡本地資訊，當結束查詢時請說 END
            """,
            model_client=get_xai_client(),
            workbench=mcp,
            model_client_stream=True,
        )
        termination = TextMentionTermination("END")
        team = RoundRobinGroupChat(
            [agent],
            termination_condition=termination,
        )
        await Console(team.run_stream(task="有哪些 db表能用"))


async def mcp_demo():
    model_client = get_xai_client()
    try:
        store_tools = await mcp_server_tools(STORE_MCP_PARAMS)
        db_tools = await mcp_server_tools(DB_MCP_PARAMS)

        tools = store_tools + db_tools
        agent = AssistantAgent(
            "local_assistant",
            system_message="""
            你是一個本地管理員，專注於處裡本地資訊，當結束查詢時請說 END
            """,
            model_client=model_client,
            tools=tools,
            model_client_stream=True,
        )

        termination = TextMentionTermination("END")
        team = RoundRobinGroupChat(
            [agent],
            termination_condition=termination,
        )
        # await Console(team.run_stream(task="目前有哪些表格能使用"))
        await Console(
            team.run_stream(task="STORE1 有幾人, 又有哪些db能用，再給我一個sqlalchemy的範例")
        )
    except Exception as ex:
        print(ex)
    finally:
        # 关闭模型客户端资源
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(work_bench_demo())
