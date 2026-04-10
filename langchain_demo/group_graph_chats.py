import os
import asyncio
import logging
from typing import TypedDict, Annotated, List, Optional
import operator

from langchain_core.tools import Tool
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_community.vectorstores import Redis as RedisVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Config:
    LLM_MODEL = "gemini-2.5-flash-preview-04-17"
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_INDEX_NAME = os.getenv("REDIS_INDEX_NAME", "story_rag_index")
    EMBEDDING_MODEL = "nomic-embed-text"

class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    next_step: Optional[str]

class RAGManager:
    """Handles Redis Vector Store and Retrieval operations."""
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model=Config.EMBEDDING_MODEL)
        self.vectorstore = self._connect_redis()
        self.retriever = self.vectorstore.as_retriever() if self.vectorstore else None

    def _connect_redis(self):
        try:
            store = RedisVectorStore(
                embedding=self.embeddings, 
                redis_url=Config.REDIS_URL, 
                index_name=Config.REDIS_INDEX_NAME
            )
            logger.info("Connected to Redis vector store.")
            return store
        except Exception as e:
            logger.error(f"Failed to connect to Redis vector store: {e}")
            return None

    def get_relevant_docs(self, query: str):
        if not self.retriever:
            return "No retriever available."
        return self.retriever.get_relevant_documents(query)

async def main():
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model=Config.LLM_MODEL, temperature=0)
    
    # Initialize RAG
    rag_manager = RAGManager()
    
    # Define Tools
    rag_tool = Tool(
        name="story_rag",
        func=rag_manager.get_relevant_docs,
        description="Useful for retrieving information from the story vector database."
    )
    
    # Define Agent
    agent = create_react_agent(llm, tools=[rag_tool])
    
    # Example usage
    inputs = {"messages": [("user", "What is the main plot of the story?")]}
    async for event in agent.astream(inputs):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
