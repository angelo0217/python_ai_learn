# mcp_demo.py
import asyncio
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.client import MultiServerMCPClient

# Set up basic logging for debugging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0)

STORE_MCP_PARAMS = StdioServerParameters(
    command="python",
    args=["mcp_demo/store_count_mcp.py"],
)


async def main():
    logger.info("Starting main application...")
    async with stdio_client(STORE_MCP_PARAMS) as (read, write):
        logger.info("Stdio client connected. Initializing MCP session...")
        async with ClientSession(read, write) as session:
            # 初始化連接
            await session.initialize()
            logger.info("MCP session initialized.")

            # 獲取工具
            logger.info("Loading MCP tools...")
            tools = await load_mcp_tools(session)
            logger.info(f"Tools loaded: {[t.name for t in tools]}")
            if not tools:
                logger.error("No tools loaded from MCP server. Check mcp_demo/store_count_mcp.py.")
                return

            # 創建並運行代理
            logger.info("Creating ReAct agent...")
            agent = create_react_agent(llm, tools)
            logger.info("Invoking agent with query: 'STORE1 有哪些人'")
            agent_response = await agent.ainvoke({"messages": "STORE1 有哪些人"})

            # 輸出所有消息
            logger.info("Agent response received. Printing all messages:")
            for message in agent_response["messages"]:
                logger.info(f"Message type: {type(message).__name__}")
                logger.info(f"Message content: {message.content}")
                if hasattr(message, "tool_calls") and message.tool_calls:
                    logger.info(f"Tool calls: {message.tool_calls}")
                if hasattr(message, "name") and message.name:
                    logger.info(f"Tool name: {message.name}")
                if hasattr(message, "tool_call_id") and message.tool_call_id:
                    logger.info(f"Tool call id: {message.tool_call_id}")
                logger.info("-" * 50)
    logger.info("Main application finished.")


if __name__ == "__main__":
    asyncio.run(main())
