"""
Tests for ingestion utilities:
- MinHash/LSH dedup keeps a single representative among identical chunks.
- Chunking respects token bounds and overlap and produces non-empty content.
"""
import pytest
import re
from src.ingestion.build_kb import deduplicate_chunks, chunk_text_sentence_aware, count_tokens


def test_deduplicate_chunks_keeps_one():
    pytest.importorskip(
        "datasketch", reason="requires datasketch for MinHash/LSH")
    rows = [
        {"doc_id": "a", "split_idx": 0, "text": "Fungus Fungus Oomycete"},
        {"doc_id": "b", "split_idx": 0, "text": "Fungus Fungus Oomycete"},
    ]
    out = deduplicate_chunks(rows, method="minhash",
                             threshold=0.8, verbose=False)
    assert len(out) == 1


def test_chunking_respects_bounds():
    text = "A. " * 600  # ~600 tokens
    chunks = chunk_text_sentence_aware(text, max_tokens=200, overlap_tokens=50)
    assert len(chunks) >= 3
    for c in chunks:
        # Allow a small overhead due to sentence packing
        assert 1 <= count_tokens(c) <= 220


def test_chunks_have_content():
    text = "Sentence one. Sentence two. Sentence three."
    chunks = chunk_text_sentence_aware(text, max_tokens=8, overlap_tokens=2)
    assert all(re.search(r"\w", c) for c in chunks)
