import os
from typing import List, Optional
from langchain_community.vectorstores import Redis as RedisVectorStore
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from src.core.config import settings

class RAGService:
    """
    RAG (Retrieval-Augmented Generation) Service for managing 
    document indexing and querying using Redis and Ollama.
    """
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.embeddings = OllamaEmbeddings(model=settings.OLLAMA_EMBEDDING_MODEL)
        self.vector_store = RedisVectorStore.from_existing_index(
            index_name=self.index_name,
            embedding=self.embeddings,
            redis_url=settings.REDIS_URL
        )
        self.llm = ChatOllama(model=settings.OLLAMA_MODEL)

    def query(self, question: str) -> str:
        """
        Queries the vector store and generates a response using the LLM.
        """
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )
        return qa_chain.run(question)

    def add_documents(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        """
        Adds documents to the Redis vector store.
        """
        self.vector_store.add_texts(texts=texts, metadatas=metadatas)

    def clear_index(self):
        """
        Clears the current Redis index.
        """
        import redis
        r = redis.from_url(settings.REDIS_URL)
        r.delete(self.index_name)
