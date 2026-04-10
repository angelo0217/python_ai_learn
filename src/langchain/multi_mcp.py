# mcp_demo.py
import asyncio
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# Set up basic logging for debugging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0)


async def main(task):
    # Initialize the client outside the async with block
    client = MultiServerMCPClient(
        {
            "store_info": {
                "command": "python",
                # 確保更新到 math_server.py 的完整絕對路徑
                "args": ["mcp_demo/store_count_mcp.py"],
                "transport": "stdio",
            },
            "sqlite": {
                # 確保你從 weather_server.py 開始
                "url": "http://localhost:8082/sse",
                "transport": "sse",
            },
        }
    )

    # Get tools after initializing the client
    tools = await client.get_tools()
    agent = create_react_agent(llm, tools)

    math_response = await agent.ainvoke({"messages": task})

    for message in math_response["messages"]:
        print(f"Math Message type: {type(message).__name__}")
        print(f"Math Message content: {message.content}")
        if hasattr(message, "tool_calls") and message.tool_calls:
            print(f"Math Tool calls: {message.tool_calls}")
        if hasattr(message, "name") and message.name:
            print(f"Math Tool name: {message.name}")
        if hasattr(message, "tool_call_id") and message.tool_call_id:
            print(f"Math Tool call id: {message.tool_call_id}")
        print("-" * 50)
    print("*" * 50)


if __name__ == "__main__":
    asyncio.run(
        main(
            """
        STORE1 有哪些人，目前db又有哪些表格
    """
        )
    )
