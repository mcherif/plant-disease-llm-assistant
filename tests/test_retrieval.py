"""Smoke tests for the hybrid retrieval pipeline (FAISS + sentence embeddings).

This test:
- Loads the built FAISS index and aligned metadata.
- Encodes a query using the model specified in the index config (adds 'query: ' if needed).
- Verifies the top-5 include a Tomato result and a Yellow Leaf Curl disease mention.

Note:
- Skipped automatically if models/index/kb-faiss-bge does not exist (CI safety).
"""
# Tests for retrieval pipeline

import json
from pathlib import Path

import numpy as np
import pytest
import faiss  # type: ignore
from sentence_transformers import SentenceTransformer


def _load_meta_cfg(index_dir: Path):
    """Load meta.jsonl and config.json from the given index directory.

    Returns:
        Tuple[List[dict], dict]: (meta rows aligned with vectors, index config).
    """
    meta = []
    with (index_dir / "meta.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                meta.append(json.loads(line))
    cfg = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
    return meta, cfg


def _embed(q: str, model_name: str, device: str = "cpu") -> np.ndarray:
    """Encode a query into a normalized embedding for cosine/IP search."""
    m = SentenceTransformer(model_name, device=device)
    v = m.encode([q], convert_to_numpy=True, normalize_embeddings=False).astype("float32")
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return v


@pytest.mark.skipif(not Path("models/index/kb-faiss-bge").exists(), reason="Hybrid index not built")
def test_tylcv_retrieval_top5():
    """Top-5 retrieval contains Tomato + Yellow Leaf Curl signals (sanity check)."""
    index_dir = Path("models/index/kb-faiss-bge")

    # Load FAISS index and metadata/config
    index = faiss.read_index(str(index_dir / "faiss.index"))
    meta, cfg = _load_meta_cfg(index_dir)
    assert index.ntotal == len(meta) > 0

    # Prepare query prefix based on how docs were indexed (e.g., BGE needs 'query: ')
    model = cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
    q_prefix = "query: " if str(cfg.get("doc_prefix", "")).strip().lower().startswith("passage:") else ""
    q = "tomato yellow leaf curl symptoms"
    v = _embed(f"{q_prefix}{q}", model_name=model, device="cpu")

    # Vector search
    scores, idxs = index.search(v, 5)
    idxs = idxs[0].tolist()
    top = [meta[i] for i in idxs]

    # Assertions: plant and disease signals appear in top-5
    assert any("tomato" == str(r.get("plant", "")).lower() for r in top)
    assert any("yellow leaf curl" in str(r.get("disease", "")).lower() for r in top)
