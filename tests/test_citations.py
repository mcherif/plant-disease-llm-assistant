from pathlib import Path
import re
import pytest

from src.llm.rag_pipeline import RAGPipeline, RetrievalConfig

INDEX_DIR = Path("models/index/kb-faiss-bge")


@pytest.mark.skipif(not (INDEX_DIR / "faiss.index").exists(), reason="Hybrid index not built")
def test_enforce_citations_appends_when_missing(monkeypatch):
    """When the LLM returns no [n], the pipeline should append a valid [1] based on sources."""
    cfg = RetrievalConfig(index_dir=INDEX_DIR, device="cpu", top_k=1)
    rag = RAGPipeline(cfg)

    # Force the LLM output to have no citations
    monkeypatch.setattr(RAGPipeline, "_generate", lambda *a, **k: "Answer without citations.")

    res = rag.answer("tomato yellow leaf curl symptoms", plant="Tomato")
    assert isinstance(res["answer"], str) and len(res["answer"]) > 0
    assert len(res["sources"]) >= 1

    # Validate at least one [n] within sources range exists
    nums = [int(n) for n in re.findall(r"\[(\d+)\]", res["answer"])]
    assert nums, "Expected at least one citation [n] to be added"
    max_id = len(res["sources"])
    assert all(1 <= n <= max_id for n in nums), "All citations must reference available sources"