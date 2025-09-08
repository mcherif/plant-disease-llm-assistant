"""Structure and sanity tests for the RAG QA dataset (data/eval/rag_qa.jsonl).

This test ensures:
- The dataset file exists and contains a reasonable number of rows.
- Each row has a sufficiently long 'question'.
- Optional 'intent' values are from an allowed set.
- No duplicate (question, plant) pairs.

Regenerate the dataset with:
    python -m src.eval.make_rag_qa
or via Make:
    make rag_qa
"""
from pathlib import Path
import json

DATA = Path("data/eval/rag_qa.jsonl")

def test_rag_qa_dataset_structure():
    assert DATA.exists(), "data/eval/rag_qa.jsonl missing"
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert 50 <= len(rows) <= 300
    seen = set()
    intents = {"symptoms","diagnosis","treatment","prevention","ipm","sanitation","chemicals","general"}
    for r in rows:
        assert isinstance(r.get("question"), str) and len(r["question"]) >= 10
        if "intent" in r:
            assert r["intent"] in intents
        key = (r["question"].lower(), (r.get("plant") or "").lower())
        assert key not in seen, "duplicate question+plant"
        seen.add(key)
