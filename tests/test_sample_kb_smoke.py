import pytest
from pathlib import Path
from src.llm.rag_pipeline import RetrievalConfig, RAGPipeline

SAMPLE_INDEX = Path("models/index/sample-faiss-bge")

@pytest.mark.skipif(not SAMPLE_INDEX.exists(), reason="Sample index not built")
def test_sample_kb_powdery_mildew_unique_source():
    cfg = RetrievalConfig(index_dir=str(SAMPLE_INDEX), device="cpu", top_k=3)
    rag = RAGPipeline(cfg)
    res = rag.answer("Powdery mildew management?", plant="Peach")
    titles = [s["title"] for s in res["sources"]]
    assert titles, "No sources returned"
    assert len(set(titles)) == len(titles), "Duplicate source titles found"

def test_sample_kb_tomato_early_blight():
    cfg = RetrievalConfig(index_dir=str(SAMPLE_INDEX), device="cpu", top_k=3)
    rag = RAGPipeline(cfg)
    res = rag.answer("Early blight symptoms?", plant="Tomato")
    assert any("Early Blight" in s["title"] for s in res["sources"])