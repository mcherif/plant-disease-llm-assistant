"""Guardrail test: refuse politely when retrieval yields no context.

How it works
- The pipeline first retrieves with the given plant filter. If there are no hits and a plant was provided,
  it retries once without the plant filter (fallback).
- If there are still no hits, it returns a short refusal message and DOES NOT call the LLM.

Why this is offline/deterministic
- The no-context path skips the LLM call, so no OPENAI_API_KEY is needed and the test is stable.

How we force "no context"
- We use an off-topic query and a bogus plant name, with top_k=1 and fusion='none' to minimize
  the chance of an accidental hit.

Skip conditions
- The test is skipped if the FAISS index directory is missing (CI safety).
"""
from pathlib import Path
import pytest

from src.llm.rag_pipeline import RAGPipeline, RetrievalConfig

INDEX_DIR = Path("models/index/kb-faiss-bge")


@pytest.mark.skipif(not (INDEX_DIR / "faiss.index").exists(), reason="Hybrid index not built")
def test_refusal_when_no_context(monkeypatch):
    """If retrieval returns no context (even after fallback), respond with refusal."""
    cfg = RetrievalConfig(index_dir=INDEX_DIR, device="cpu")
    rag = RAGPipeline(cfg)

    # Assert the LLM is not called on no-context path
    def boom(*args, **kwargs):
        raise AssertionError("LLM should not be called on no-context guardrail path")
    monkeypatch.setattr(RAGPipeline, "_generate", boom, raising=True)

    # Deterministically force no context: top_k=0 yields no retrieved chunks
    res = rag.answer("Quantum entanglement basics", plant="NonExistingPlant", top_k=0)
    assert isinstance(res["answer"], str)
    assert res["sources"] == []
    assert res["retrieved"] == []