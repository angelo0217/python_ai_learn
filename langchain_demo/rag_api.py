import os
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.config import settings
from core.logger import logger
from core.exceptions import BaseAppException, DatabaseError, ModelError
from langchain_demo.redis_rag import RAGService

# Initialize FastAPI app
app = FastAPI(title="RAG API Service")

# Dependency Injection for RAGService
def get_rag_service() -> RAGService:
    return RAGService(redis_url=settings.REDIS_URL)

# Custom Exception Handler for BaseAppException
@app.exception_handler(BaseAppException)
async def app_exception_handler(request: Request, exc: BaseAppException):
    logger.error(f"AppException occurred: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.message},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception occurred: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "An unexpected internal server error occurred."},
    )

# Pydantic models for request bodies
class QueryRequest(BaseModel):
    query: str
    index_name: str

class TrainRequest(BaseModel):
    file_path: str
    index_name: str

class MultiQueryRequest(BaseModel):
    query: str
    index_names: List[str]

@app.post("/train")
async def train_endpoint(request: TrainRequest, rag_service: RAGService = Depends(get_rag_service)):
    """
    Trains the vector database with a provided text file.
    """
    try:
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")

        rag_service.train_vector_database(request.file_path, request.index_name)
        return {
            "status": "success",
            "message": f"Index '{request.index_name}' trained successfully.",
        }
    except BaseAppException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in /train: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_endpoint(request: QueryRequest, rag_service: RAGService = Depends(get_rag_service)):
    """
    Queries the RAG system for a specific index.
    """
    try:
        result = rag_service.query(request.query, request.index_name)
        return {
            "status": "success",
            "data": result
        }
    except BaseAppException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in /query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/multi-query")
async def multi_query_endpoint(request: MultiQueryRequest, rag_service: RAGService = Depends(get_rag_service)):
    """
    Queries multiple RAG indices and aggregates results.
    """
    try:
        results = {}
        for index_name in request.index_names:
            results[index_name] = rag_service.query(request.query, index_name)
        
        return {
            "status": "success",
            "data": results
        }
    except BaseAppException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in /multi-query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
