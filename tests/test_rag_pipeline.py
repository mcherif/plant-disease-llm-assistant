"""Tests for the RAG pipeline.

Contains:
- Unit test with a mocked LLM (_generate) to keep the test offline, deterministic,
  and fast. This verifies retrieval, prompt composition, and wiring without calling
  external APIs.
- Optional integration smoke test that hits the real OpenAI API when OPENAI_API_KEY
  is set, to catch breaking changes in the backend.
"""
from pathlib import Path
import os
import logging

import pytest

from src.llm.rag_pipeline import RAGPipeline, RetrievalConfig

INDEX_DIR = Path("models/index/kb-faiss-bge")


@pytest.mark.skipif(not (INDEX_DIR / "faiss.index").exists(), reason="Hybrid index not built")
def test_rag_answer_with_mock_llm(monkeypatch):
    """Unit test: mock the LLM so we can test end-to-end without network or cost.

    Why mock?
    - Avoid dependence on external API availability/latency/cost.
    - Make assertions deterministic and reproducible in CI.

    What it checks:
    - Pipeline runs end-to-end with retrieval and prompt composition.
    - The composed prompt includes numbered context and the original question.
    - Output schema matches expectations.
    """
    cfg = RetrievalConfig(index_dir=INDEX_DIR, device="cpu", fusion="sum", alpha=0.7, top_k=3)
    rag = RAGPipeline(cfg)

    captured = {"prompt": None}

    # Replace the real _generate with a fake that records the prompt and returns a fixed answer.
    def fake_generate(self, prompt: str, model=None) -> str:
        captured["prompt"] = prompt
        return "Mock answer [1]."

    monkeypatch.setattr(RAGPipeline, "_generate", fake_generate, raising=True)

    res = rag.answer("tomato yellow leaf curl symptoms", plant="Tomato")

    assert isinstance(res["answer"], str) and res["answer"].startswith("Mock answer")
    assert 1 <= len(res["sources"]) <= 3
    assert len(res["retrieved"]) == len(res["sources"])

    # Prompt sanity checks
    assert captured["prompt"] is not None
    p = captured["prompt"].lower()
    assert "\n[1] " in p
    assert "question:" in p
    assert "tomato yellow leaf curl symptoms" in p


@pytest.mark.skipif(not (INDEX_DIR / "faiss.index").exists(), reason="Hybrid index not built")
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_rag_openai_integration_smoke():
    """Integration smoke test: exercise the real OpenAI backend when a key is present.

    Notes:
    - Runs only when OPENAI_API_KEY is set.
    - Not strictly deterministic, so keep assertions loose and focus on basic behavior.
    """
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    logging.info("OpenAI integration enabled. Using model=%s; index=%s", model, INDEX_DIR)
    cfg = RetrievalConfig(index_dir=INDEX_DIR, device="cpu", fusion="sum", alpha=0.7, top_k=2)
    rag = RAGPipeline(cfg)
    res = rag.answer("tomato yellow leaf curl symptoms", plant="Tomato")
    logging.info("Received answer chars=%d; sources=%d", len(res["answer"]), len(res["sources"]))
    logging.debug("Answer preview: %s", res["answer"][:300].replace("\n", " ") + "...")
    assert isinstance(res["answer"], str) and len(res["answer"].strip()) > 0
    assert len(res["sources"]) > 0
