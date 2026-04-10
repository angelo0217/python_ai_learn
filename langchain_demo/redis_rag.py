import asyncio
import os
import redis
import logging
from typing import List, Dict, Any, Optional
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Redis as RedisVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import settings
from core.logger import logger
from core.exceptions import DatabaseError, ModelError

# Define different index names
STORY_INDEX_NAME = "story_rag_index"
TECH_DOC_INDEX_NAME = "tech_doc_rag_index"

class RAGService:
    """
    Service for managing RAG (Retrieval-Augmented Generation) operations using Redis as a vector store.
    """
    def __init__(self, redis_url: str = settings.REDIS_URL):
        self.redis_url = redis_url
        self.embeddings = OllamaEmbeddings(model="llama3")
        self.llm = ChatOllama(model="llama3")
        logger.info(f"RAGService initialized with Redis URL: {self.redis_url}")

    def _get_redis_client(self):
        try:
            return redis.from_url(self.redis_url)
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise DatabaseError(f"Redis connection failed: {e}")

    def _clear_redis_index(self, index_name: str) -> None:
        """
        Clears an existing Redis index if it exists.
        """
        try:
            r = self._get_redis_client()
            # RedisVectorStore uses a specific key pattern; we clear based on index name
            # Note: In a production environment, use a more precise deletion method
            r.flushdb() 
            logger.info(f"Redis index {index_name} cleared.")
        except Exception as e:
            logger.warning(f"Could not clear Redis index {index_name}: {e}")

    def train_vector_database(self, file_path: str, index_name: str) -> None:
        """
        Loads a text file, splits it into chunks, and stores embeddings in Redis.
        """
        try:
            logger.info(f"Training vector database {index_name} with file {file_path}")
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            loader = TextLoader(file_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_documents(documents)

            self._clear_redis_index(index_name)
            
            RedisVectorStore.from_documents(
                documents=texts,
                embedding=self.embeddings,
                redis_url=self.redis_url,
                index_name=index_name
            )
            logger.info(f"Successfully trained index {index_name}")
        except Exception as e:
            logger.error(f"Error during training vector database {index_name}: {e}")
            raise ModelError(f"Failed to train RAG index: {e}")

    def query(self, query_text: str, index_name: str) -> Dict[str, Any]:
        """
        Queries the vector store and generates a response using the LLM.
        """
        try:
            logger.info(f"Querying index {index_name} with text: {query_text}")
            
            vector_store = RedisVectorStore(
                index_name=index_name,
                embedding=self.embeddings,
                redis_url=self.redis_url
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever()
            )
            
            result = qa_chain.invoke({"query": query_text})
            
            # Extract sources if available
            sources = []
            if "source_documents" in result:
                sources = [doc.page_content for doc in result["source_documents"]]
            
            return {
                "answer": result.get("result", "No answer found"),
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Error querying index {index_name}: {e}")
            raise ModelError(f"RAG query failed: {e}")
