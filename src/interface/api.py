"""
FastAPI backend for Plant Disease RAG Assistant.

This module exposes a REST API for retrieval-augmented generation (RAG) over the plant disease knowledge base.
- /health: Returns API and pipeline status, config, and metadata.
- /rag: Accepts a user query and optional filters, runs retrieval and LLM answer generation, and returns sources, retrieved chunks, and latency.

Use this API for programmatic access, integration with other apps, or as a backend for the Streamlit UI.
To run locally: uvicorn src.interface.api:app --reload --port 8000
"""
import os
import time
from typing import Optional, List, Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.llm.rag_pipeline import RAGPipeline, RetrievalConfig
import torch

# --- Config (env overrides) ---


def _auto_device():
    env = os.getenv("RETRIEVAL_DEVICE")
    if env:
        return env
    return "cuda" if torch.cuda.is_available() else "cpu"


INDEX_DIR = os.getenv("INDEX_DIR", "models/index/kb-faiss-bge")
RETRIEVAL_DEVICE = _auto_device()
DEFAULT_TOP_K = int(os.getenv("TOP_K", "3"))

# Lazy global pipeline (built once)
_rag_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        cfg = RetrievalConfig(index_dir=INDEX_DIR,
                              top_k=DEFAULT_TOP_K, device=RETRIEVAL_DEVICE)
        _rag_pipeline = RAGPipeline(cfg)
        print(f"[api] Loaded pipeline index_dir={INDEX_DIR} "
              f"docs={len(_rag_pipeline.meta)} device={RETRIEVAL_DEVICE}")
    return _rag_pipeline


# --- Schemas ---
class RagRequest(BaseModel):
    query: str = Field(..., description="User question")
    plant: Optional[str] = Field(None, description="Plant name filter")
    disease: Optional[str] = Field(None, description="Disease name filter")
    top_k: Optional[int] = Field(
        None, ge=0, le=12, description="Override top_k")
    fusion: Optional[str] = Field(
        None, description="Fusion strategy (none|sum|rrf)")
    alpha: Optional[float] = Field(
        None, ge=0, le=1, description="Alpha for weighted sum fusion")
    temperature: float = Field(0.0, ge=0, le=1, description="RAG temperature")
    timeout: Optional[float] = Field(
        None, ge=1, le=120, description="RAG call timeout seconds")


class Source(BaseModel):
    id: int
    title: str
    url: str = ""


class Retrieved(BaseModel):
    score: float
    meta: Dict[str, Any]


class RagResponse(BaseModel):
    answer: str
    sources: List[Source]
    retrieved: List[Retrieved]
    latency_ms: int
    top_k: int


# --- App ---
app = FastAPI(title="Plant Disease RAG API", version="0.1.0-skeleton")


@app.get("/health")
def health():
    p = get_pipeline()
    meta_count = getattr(p, "meta", None)
    if isinstance(meta_count, list):
        meta_count = len(meta_count)
    return {
        "status": "ok",
        "index_dir": INDEX_DIR,
        "device": RETRIEVAL_DEVICE,
        "default_top_k": DEFAULT_TOP_K,
        "meta_docs": meta_count,
        "model_env": os.getenv("OPENAI_MODEL"),
    }


def _clean_title(t: str) -> str:
    if not t:
        return ""
    # TECH-DEBT: This is a naive mojibake cleanup (double-encoded UTF-8 artifacts).
    # Proper fix: ensure ingestion reads UTF-8 once; optionally apply a robust
    # normalization (e.g., ftfy.fix_text) during KB build, not at serving time.
    return (
        t.replace("â€“", "–")
         .replace("â€”", "—")
         .replace("â", "-")
         .strip()
    )


@app.post("/rag", response_model=RagResponse)
def rag_endpoint(req: RagRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")
    pipeline = get_pipeline()

    start = time.time()
    try:
        res = pipeline.answer(
            query=req.query.strip(),
            plant=(req.plant or None),
            disease=(req.disease or None),
            top_k=req.top_k,
            fusion=req.fusion,
            alpha=req.alpha,
            temperature=req.temperature,
            timeout=req.timeout,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"RAG failure: {e}") from e
    latency_ms = int((time.time() - start) * 1000)

    # Shape / validate
    answer = res.get("answer", "")
    sources_raw = res.get("sources", []) or []
    retrieved_raw = res.get("retrieved", []) or []

    sources = [
        Source(
            id=int(s.get("id", i + 1)),
            title=_clean_title(str(s.get("title") or f"Source {i+1}")),
            url=str(s.get("url") or ""),
        )
        for i, s in enumerate(sources_raw)
    ]
    # Preserve a trimmed snippet for debugging
    retrieved = []
    for r in retrieved_raw:
        meta = r.get("meta") or {}
        snippet = (meta.get("text") or "")[:180].strip()
        if snippet:
            meta = {k: v for k, v in meta.items() if k != "text"}
            meta["snippet"] = snippet
        retrieved.append(
            Retrieved(score=float(r.get("score", 0.0)), meta=meta))

    return RagResponse(
        answer=answer,
        sources=sources,
        retrieved=retrieved,
        latency_ms=latency_ms,
        top_k=len(retrieved),
    )


# Optional local run: uvicorn src.interface.api:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.interface.api:app",
                host="0.0.0.0", port=8000, reload=True)
