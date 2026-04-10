import os
import logging
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_demo.redis_rag import RAGService, REDIS_URL

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG API Service", description="API for training and querying Redis-based RAG")

# Initialize RAG Service
rag_service = RAGService(redis_url=REDIS_URL)

# --- Request Models ---

class QueryRequest(BaseModel):
    query: str
    index_name: str

class TrainRequest(BaseModel):
    file_path: str
    index_name: str

class MultiQueryRequest(BaseModel):
    query: str
    index_names: List[str]

# --- Endpoints ---

@app.post("/train")
async def train_endpoint(request: TrainRequest):
    """
    Trains a vector database index using a provided text file.
    """
    try:
        if not os.path.exists(request.file_path):
            logger.error(f"File not found: {request.file_path}")
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")

        rag_service.train_vector_database(request.file_path, request.index_name)
        return {
            "status": "success",
            "message": f"Index '{request.index_name}' trained successfully.",
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(f"Unexpected error during training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """
    Queries a specific vector index to get an AI-generated answer based on retrieved context.
    """
    try:
        result = rag_service.query(request.query, request.index_name)
        return {
            "query": request.query,
            "index": request.index_name,
            "answer": result.get("answer"),
            "sources": result.get("sources", [])
        }
    except Exception as e:
        logger.exception(f"Error during query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/multi-query")
async def multi_query_endpoint(request: MultiQueryRequest):
    """
    Queries multiple vector indices and aggregates the answers.
    """
    try:
        aggregated_results = []
        for index_name in request.index_names:
            res = rag_service.query(request.query, index_name)
            aggregated_results.append({
                "index": index_name,
                "answer": res.get("answer"),
                "sources": res.get("sources", [])
            })
        
        return {
            "query": request.query,
            "results": aggregated_results
        }
    except Exception as e:
        logger.exception(f"Error during multi-query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
