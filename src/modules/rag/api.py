from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List
from src.modules.rag.service import RAGService
from src.core.config import settings

app = FastAPI(title="RAG API Service")

class QueryRequest(BaseModel):
    question: str
    index_name: str = "default_index"

class DocumentRequest(BaseModel):
    texts: List[str]
    index_name: str = "default_index"

def get_rag_service(index_name: str = "default_index") -> RAGService:
    """
    Dependency provider for RAGService.
    """
    return RAGService(index_name=index_name)

@app.post("/query")
async def ask_question(request: QueryRequest, service: RAGService = Depends(lambda: RAGService(request.index_name))):
    try:
        answer = service.query(request.question)
        return {"question": request.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/index")
async def add_docs(request: DocumentRequest, service: RAGService = Depends(lambda: RAGService(request.index_name))):
    try:
        service.add_documents(request.texts)
        return {"status": "success", "message": f"Added {len(request.texts)} documents to {request.index_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
