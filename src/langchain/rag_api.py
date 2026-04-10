import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_demo.redis_rag import RAGService, REDIS_URL

# ... (paste the RAGService class and helper functions here) ...

app = FastAPI()
rag_service = RAGService(redis_url=REDIS_URL)


# Pydantic model for request body
class QueryRequest(BaseModel):
    query: str
    index_name: str


class TrainRequest(BaseModel):
    file_path: str
    index_name: str


class MultiQueryRequest(BaseModel):
    query: str
    index_names: list[str]


@app.post("/train")
async def train_endpoint(request: TrainRequest):
    try:
        # In a real API, you might upload files or specify paths accessible by the server
        # For demonstration, ensure the file_path exists on the server.
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")

        rag_service.train_vector_database(request.file_path, request.index_name)
        return {
            "status": "success",
            "message": f"Index '{request.index_name}' trained successfully.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        response = rag_service.query(request.query, request.index_name)
        # Format the source documents for a cleaner API response
        formatted_sources = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in response.get("source_documents", [])
        ]
        return {"answer": response["result"], "sources": formatted_sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/multi_query")
async def multi_query_endpoint(request: MultiQueryRequest):
    try:
        response = rag_service.multi_index_query(request.query, request.index_names)
        formatted_sources = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in response.get("source_documents", [])
        ]
        return {"answer": response["result"], "sources": formatted_sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# To run this FastAPI app:
# uvicorn langchain_demo.rag_api:app --reload
