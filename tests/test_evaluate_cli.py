import json
import sys
import subprocess
from pathlib import Path

import pytest


INDEX_DIR = Path("models/index/kb-faiss-bge")


@pytest.mark.skipif(not (INDEX_DIR / "faiss.index").exists(), reason="Hybrid index not built")
def test_evaluate_cli_writes_metrics(tmp_path: Path):
    # Create a tiny labels file
    labels = [
        {"query": "tomato yellow leaf curl symptoms", "plant": "Tomato"},
        {"query": "citrus greening symptoms", "plant": "Orange"},
    ]
    labels_path = tmp_path / "labels.jsonl"
    with labels_path.open("w", encoding="utf-8") as f:
        for row in labels:
            f.write(json.dumps(row) + "\n")

    out_dir = tmp_path / "eval_out"

    # Run the evaluator as a CLI module
    cmd = [
        sys.executable, "-m", "src.retrieval.evaluate",
        "--index-dir", str(INDEX_DIR),
        "--labels", str(labels_path),
        "--device", "cpu",
        "--fusion", "sum",
        "--alpha", "0.7",
        "--top-ks", "5",
        "--out", str(out_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert res.returncode == 0
    assert (out_dir / "retrieval_eval.json").exists()

    # Basic shape check
    data = json.loads((out_dir / "retrieval_eval.json").read_text(encoding="utf-8"))
    assert "aggregate" in data and "per_query" in data
    assert isinstance(data["per_query"], list)