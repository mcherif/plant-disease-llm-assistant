import subprocess
import sys
from pathlib import Path
import pytest
from src.llm.rag_pipeline import RetrievalConfig, RAGPipeline

SAMPLE_INDEX = Path("models/index/sample-faiss-bge")

@pytest.fixture(scope="session", autouse=True)
def ensure_sample_index():
    # Build if faiss.index missing
    if not SAMPLE_INDEX.joinpath("faiss.index").exists():
        subprocess.check_call([sys.executable, "-m", "src.ingestion.build_sample_kb"])

@pytest.mark.skipif(not SAMPLE_INDEX.exists(), reason="Sample index directory missing")
def test_sample_kb_tomato_early_blight(ensure_sample_index):
    cfg = RetrievalConfig(index_dir=str(SAMPLE_INDEX), device="cpu", top_k=3)
    rag = RAGPipeline(cfg)
    res = rag.answer("Early blight symptoms?", plant="Tomato")
    assert any("Early Blight" in s["title"] for s in res["sources"])

@pytest.mark.skipif(not SAMPLE_INDEX.exists(), reason="Sample index directory missing")
def test_sample_kb_powdery_mildew_unique_source(ensure_sample_index):
    cfg = RetrievalConfig(index_dir=str(SAMPLE_INDEX), device="cpu", top_k=3)
    rag = RAGPipeline(cfg)
    res = rag.answer("Powdery mildew management?", plant="Peach")
    titles = [s["title"] for s in res["sources"]]
    assert titles
    assert len(set(titles)) == len(titles)