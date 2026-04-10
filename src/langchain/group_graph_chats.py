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
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator

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
    vectorstore = None

retriever = None
qa_chain = None

if vectorstore:
    retriever = vectorstore.as_retriever()
    logger.info("Created retriever.")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )
    logger.info("Created RAG Chain.")


# --- Define Agent State for LangGraph ---
class AgentState(TypedDict):
    """
    Represents the state of our graph.

    - `input`: The original user query.
    - `store_output`: The result from the STORE agent.
    - `sqlite_output`: The result from the SQLite agent.
    - `rag_output`: The result from the RAG agent.
    - `final_answer`: The combined final answer.
    """

    input: str
    store_output: Annotated[str, operator.add]
    sqlite_output: Annotated[str, operator.add]
    rag_output: Annotated[str, operator.add]
    final_answer: Annotated[str, operator.add]


async def main(task):
    tools_client1 = MultiServerMCPClient(
        {
            "store_info": {
                "command": "python",
                "args": ["mcp_demo/store_count_mcp.py"],  # 確保路徑正確
                "transport": "stdio",
            },
        }
    )
    tools_client2 = MultiServerMCPClient(
        {
            "sqlite": {
                "url": "http://localhost:8082/sse",
                "transport": "sse",
            },
        }
    )

    store_tools = await tools_client1.get_tools()
    sqlite_tools = await tools_client2.get_tools()

    rag_tool = None
    if qa_chain:

        def run_rag_query(query: str) -> str:
            """
            Queries the vector database using the RAG chain to retrieve relevant information.
            Input should be a question or query.
            """
            try:
                result = qa_chain.invoke({"query": query})
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
        logger.info("Added knowledge_base_query tool.")
    else:
        logger.warning(
            "RAG chain not initialized, 'knowledge_base_query' tool will not be available."
        )

    # --- Agent Definitions ---

    # 1. STORE Agent
    store_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一個專門處理 STORE 相關查詢的專家助理。使用你可用的工具來回答關於商店資訊、數量以及任何與 'store_info' 相關的問題。只回答與 STORE 相關的問題，若無相關問題，則輸出 'STORE_SKIP'。",
            ),
            ("human", "{messages}"),
        ]
    )
    store_agent = create_react_agent(llm, store_tools, prompt=store_prompt)

    # 2. SQLite Agent
    sqlite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一個專門處理 SQLite 資料庫查詢的專家助理。使用你可用的工具來與 SQLite 資料庫互動並檢索資訊，特別是關於資料庫表格及其內容。只回答與 SQLite 相關的問題，若無相關問題，則輸出 'SQLITE_SKIP'。",
            ),
            ("human", "{messages}"),
        ]
    )
    sqlite_agent = create_react_agent(llm, sqlite_tools, prompt=sqlite_prompt)

    # 3. General RAG Agent (for remaining questions + vector database)
    rag_tools = [rag_tool] if rag_tool else []
    general_rag_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一個樂於助人的通用助理。如果用戶的問題與我們知識庫中儲存的特定文件或資訊相關，請使用 'knowledge_base_query' 工具。對於其他一般問題，請直接回答。盡量從向量資料庫中尋找答案，若無相關則輸出 'RAG_SKIP'。",
            ),
            ("human", "{messages}"),
        ]
    )
    general_rag_agent = create_react_agent(llm, rag_tools, prompt=general_rag_prompt)

    # --- Node Definitions for LangGraph ---
    async def call_store_agent(state: AgentState):
        logger.info("Calling STORE Agent...")
        logger.info(f"Input to STORE Agent: {state['input']}")
        response = await store_agent.ainvoke({"messages": state["input"]})
        # Extract content from the last message
        content = response["messages"][-1].content if response["messages"] else "STORE_SKIP"
        logger.info(f"STORE Agent raw response: {content}")
        # Simple check if the agent decided to skip or found nothing relevant
        if (
            "STORE_SKIP" in content or "Tool calls: []" in content
        ):  # Added check for empty tool calls
            logger.info("STORE Agent decided to skip or found no relevant tool calls.")
            return {"store_output": ""}  # Return empty to signify skip
        return {"store_output": f"\nSTORE Agent Response: {content}"}

    async def call_sqlite_agent(state: AgentState):
        logger.info("Calling SQLite Agent...")
        response = await sqlite_agent.ainvoke({"messages": state["input"]})
        content = response["messages"][-1].content if response["messages"] else "SQLITE_SKIP"
        logger.info(f"SQLite Agent raw response: {content}")
        if (
            "SQLITE_SKIP" in content or "Tool calls: []" in content
        ):  # Added check for empty tool calls
            logger.info("SQLite Agent decided to skip or found no relevant tool calls.")
            return {"sqlite_output": ""}
        return {"sqlite_output": f"\nSQLITE Agent Response: {content}"}

    async def call_rag_agent(state: AgentState):
        logger.info("Calling RAG Agent...")
        response = await general_rag_agent.ainvoke({"messages": state["input"]})
        content = response["messages"][-1].content if response["messages"] else "RAG_SKIP"
        logger.info(f"RAG Agent raw response: {content}")
        if (
            "RAG_SKIP" in content
            or "No relevant information found." in content
            or "Tool calls: []" in content
        ):  # Added check for empty tool calls
            logger.info("RAG Agent decided to skip or found no relevant tool calls.")
            return {"rag_output": ""}
        return {"rag_output": f"\nRAG Agent Response: {content}"}

    # mcp_demo.py (只修改 format_final_answer 函數)

    def format_final_answer(state: AgentState):
        logger.info("Formatting final answer with enhanced logic...")
        combined_output = []

        # 追蹤已回答的問題類別
        answered_categories = {"store": False, "sqlite_tables": False, "rag_general": False}

        # 1. 處理 STORE Agent 的輸出
        if state.get("store_output") and "STORE_SKIP" not in state["store_output"]:
            # 檢查 STORE Agent 的輸出是否包含實際內容（例如，數字或具體信息）
            # 這裡可以根據實際輸出內容做更精確的判斷
            if "使用者" in state["store_output"] or "管理員" in state["store_output"]:
                combined_output.append(state["store_output"].strip())
                answered_categories["store"] = True
                logger.info("STORE category answered.")

        # 2. 處理 SQLite Agent 的輸出
        if state.get("sqlite_output") and "SQLITE_SKIP" not in state["sqlite_output"]:
            # 檢查 SQLite Agent 的輸出是否包含實際表格信息
            if (
                "表格" in state["sqlite_output"]
                or "欄位" in state["sqlite_output"]
                or "table" in state["sqlite_output"]
            ):
                combined_output.append(state["sqlite_output"].strip())
                answered_categories["sqlite_tables"] = True
                logger.info("SQLITE tables category answered.")

        # 3. 處理 RAG Agent 的輸出
        # 只有當 STORE 或 SQLite 相關的問題沒有被明確回答時，RAG 才嘗試回答相關部分
        if (
            state.get("rag_output")
            and "RAG_SKIP" not in state["rag_output"]
            and "No relevant information found." not in state["rag_output"]
        ):
            rag_answer = state["rag_output"].strip()

            # 嘗試從 RAG 輸出中提取各部分答案
            # 這裡需要更精細的邏輯來判斷 RAG 答案的內容
            # 由於你的 RAG 輸出了多個點的答案，需要拆解
            rag_parts = rag_answer.split("\n")
            processed_rag_parts = []

            for part in rag_parts:
                # 判斷 RAG 輸出的每個部分是否與已回答的類別重疊
                if "STORE1" in part and answered_categories["store"]:
                    logger.info(f"Skipping RAG part (STORE) due to prior answer: {part}")
                    continue  # 已被 STORE Agent 回答，跳過 RAG 的這部分

                if ("表格" in part or "資料庫" in part) and answered_categories["sqlite_tables"]:
                    logger.info(f"Skipping RAG part (SQLITE tables) due to prior answer: {part}")
                    continue  # 已被 SQLite Agent 回答，跳過 RAG 的這部分

                # 如果沒有重疊，或者不是 Store/SQLite 問題，則保留 RAG 的答案
                processed_rag_parts.append(part)
                answered_categories["rag_general"] = True  # 標記 RAG 處理了通用問題

            if processed_rag_parts:
                # 重新組裝 RAG 的有效部分
                # 添加一個標題來區分 RAG 的回答
                combined_output.append("\n知識庫查詢結果:")
                combined_output.extend(processed_rag_parts)

        # 如果所有代理人都沒有提供有效答案
        if not combined_output:
            return {"final_answer": "抱歉，我未能找到與您查詢相關的有效資訊。"}

        # 最終組裝答案
        final_text = "\n\n".join(combined_output)
        return {"final_answer": final_text}

    # --- Build the LangGraph Workflow ---
    workflow = StateGraph(AgentState)

    # Add nodes for each agent
    workflow.add_node("store_node", call_store_agent)
    workflow.add_node("sqlite_node", call_sqlite_agent)
    workflow.add_node("rag_node", call_rag_agent)
    workflow.add_node("final_format_node", format_final_answer)

    # Set the entry point
    workflow.set_entry_point("store_node")

    # Define the sequence
    workflow.add_edge("store_node", "sqlite_node")
    workflow.add_edge("sqlite_node", "rag_node")
    workflow.add_edge("rag_node", "final_format_node")
    workflow.add_edge("final_format_node", END)

    # Compile the graph
    app = workflow.compile()

    # Run the graph with the initial task
    final_state = await app.ainvoke({"input": task})

    print("\n" + "=" * 70)
    print("FINAL COMBINED ANSWER:")
    print(final_state["final_answer"])
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(
        main(
            """
           1、TORE1 有哪些人
           2、查詢db，張三的email是什麼?
           3、提到冰冷的建築段落主要在說什麼
        """
        )
    )
