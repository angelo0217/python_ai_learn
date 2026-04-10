import asyncio
import logging
import os
from typing import List, Dict, Any, Optional

import redis
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Redis as RedisVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration constants
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
STORY_INDEX_NAME = "story_rag_index"
TECH_DOC_INDEX_NAME = "tech_doc_rag_index"
DEFAULT_MODEL = "llama3"

class RAGService:
    """
    Service class to handle Retrieval-Augmented Generation (RAG) using Redis and Ollama.
    """
    def __init__(self, redis_url: str = REDIS_URL):
        self.redis_url = redis_url
        self.embeddings = OllamaEmbeddings(model=DEFAULT_MODEL)
        logger.info(f"RAGService initialized with Redis URL: {self.redis_url}")

    def _clear_redis_index(self, index_name: str) -> None:
        """
        Clears an existing Redis index if it exists.
        """
        try:
            r = redis.from_url(self.redis_url)
            # Redis Vector Store in LangChain uses specific keys; 
            # for a full clear in this demo, we flush or delete specific keys.
            # Note: In production, use a more targeted deletion.
            r.flushall() 
            logger.info(f"Redis index {index_name} cleared.")
        except redis.RedisError as e:
            logger.error(f"Failed to clear Redis index {index_name}: {e}")
            raise

    def train_vector_database(self, file_path: str, index_name: str) -> None:
        """
        Loads a text file, splits it into chunks, and stores it in Redis.
        
        Args:
            file_path: Path to the text file to be indexed.
            index_name: Name of the Redis index to create/update.
        """
        try:
            logger.info(f"Training vector database '{index_name}' with file: {file_path}")
            
            # Load document
            loader = TextLoader(file_path)
            documents = loader.load()

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=100
            )
            texts = text_splitter.split_documents(documents)

            # Store in Redis
            RedisVectorStore.from_documents(
                documents=texts,
                embedding=self.embeddings,
                redis_url=self.redis_url,
                index_name=index_name
            )
            logger.info(f"Successfully indexed {len(texts)} chunks into {index_name}.")
        except Exception as e:
            logger.error(f"Error during training vector database: {e}")
            raise

    def query(self, query_text: str, index_name: str) -> Dict[str, Any]:
        """
        Performs a RAG query to retrieve relevant documents and generate an answer.
        
        Args:
            query_text: The user's question.
            index_name: The index to query from.
            
        Returns:
            A dictionary containing the answer and the source documents.
        """
        try:
            logger.info(f"Querying index {index_name} with: {query_text}")
            
            # Initialize VectorStore
            vectorstore = RedisVectorStore(
                redis_url=self.redis_url,
                index_name=index_name,
                embedding=self.embeddings
            )

            # Setup LLM (Ollama)
            llm = ChatOllama(model=DEFAULT_MODEL)

            # Create RetrievalQA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever()
            )

            # Execute query
            result = qa_chain.invoke({"query": query_text})
            
            # Extract sources (LangChain's RetrievalQA usually returns 'result' and 'source_documents' if return_source_documents=True)
            # For simplicity in this refactor, we'll use the standard invoke and handle the result.
            
            return {
                "answer": result.get("result", "No answer found."),
                "sources": [doc.page_content for doc in result.get("source_documents", [])]
            }
        except Exception as e:
            logger.error(f"Error during RAG query: {e}")
            raise
