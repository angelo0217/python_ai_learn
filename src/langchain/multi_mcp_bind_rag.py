# mcp_demo.py
import os
import asyncio
import logging
from langchain_core.tools import Tool
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_community.vectorstores import Redis as RedisVectorStore

# Set up basic logging for debugging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0)

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
index_name = os.getenv("REDIS_INDEX_NAME", "story_rag_index")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Connect to the existing Redis vector store
try:
    vectorstore = RedisVectorStore(embedding=embeddings, redis_url=redis_url, index_name=index_name)
    logger.info("Connected to Redis vector store.")
except Exception as e:
    logger.error(f"Failed to connect to Redis vector store: {e}")
    # Handle the error appropriately, e.g., exit or disable RAG functionality
    vectorstore = None  # Set to None if connection fails

retriever = None
qa_chain = None

if vectorstore:
    # Create retriever
    retriever = vectorstore.as_retriever()
    logger.info("Created retriever.")

    # Create RAG Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )
    logger.info("Created RAG Chain.")
# --- End of Vector DB and RAG Chain Setup ---


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
    mcp_tools = await client.get_tools()
    custom_tools = []
    if qa_chain:

        def run_rag_query(query: str) -> str:
            """
            Queries the vector database using the RAG chain to retrieve relevant information.
            Input should be a question or query.
            """
            try:
                result = qa_chain.invoke({"query": query})
                # You might want to format the output for the agent,
                # perhaps including source documents.
                if result and "result" in result:
                    source_docs = result.get("source_documents", [])
                    source_info = (
                        "\nSources:\n"
                        + "\n".join([doc.metadata.get("source", "Unknown") for doc in source_docs])
                        if source_docs
                        else ""
                    )
                    return f"Answer: {result['result']}{source_info}"
                return "No relevant information found."
            except Exception as e:
                logger.error(f"Error during RAG query: {e}")
                return f"An error occurred while querying the knowledge base: {e}"

        rag_tool = Tool(
            name="knowledge_base_query",
            func=run_rag_query,
            description="Useful for answering questions about specific documents or information stored in the vector database. Input should be a clear, concise question.",
        )
        custom_tools.append(rag_tool)
        logger.info("Added knowledge_base_query tool to agent.")
    else:
        logger.warning(
            "RAG chain not initialized, 'knowledge_base_query' tool will not be available."
        )

    # Combine tools from MCP client and custom RAG tool
    all_tools = mcp_tools + custom_tools
    agent = create_react_agent(llm, all_tools)

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
        STORE1 有哪些人，目前db又有哪些表格，提到冰冷的建築段落主要在說什麼
    """
        )
    )
