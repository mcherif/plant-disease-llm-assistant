from pathlib import Path
import json

DATA = Path("data/eval/rag_qa.jsonl")

def test_rag_qa_dataset_structure():
    assert DATA.exists(), "data/eval/rag_qa.jsonl missing"
    rows = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]
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