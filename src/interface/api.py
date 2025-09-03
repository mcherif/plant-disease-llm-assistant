from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Any, Dict, List

from src.llm.rag_pipeline import RAGPipeline, RetrievalConfig

app = FastAPI(title="Plant Disease RAG API")

# Simple in-process pipeline
_cfg = RetrievalConfig(index_dir="models/index/kb-faiss-bge", top_k=3, device="cpu")
_rag = RAGPipeline(_cfg)

class RagRequest(BaseModel):
    question: str
    plant: Optional[str] = None

class RagResponse(BaseModel):
    answer: str
    retrieved: List[Dict[str, Any]]

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/rag", response_model=RagResponse)
def rag(req: RagRequest):
    res = _rag.answer(req.question, plant=req.plant)
    return RagResponse(answer=res.get("answer", ""), retrieved=res.get("retrieved", []))
