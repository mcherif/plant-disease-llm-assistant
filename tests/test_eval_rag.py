"""CLI smoke test for the RAG evaluator.

What it checks
- The evaluate_rag CLI runs and exits with code 0.
- If OPENAI_API_KEY is set, it should produce rag_eval.json/csv artifacts.
- If OPENAI_API_KEY is not set, it should print a skip message and produce no artifacts.

Why it’s stable
- Uses --skip-if-no-key to avoid failing in CI without credentials.
- Limits examples to a tiny subset for speed.
- Runs the script as a module (-m) so package imports (src.*) resolve.

Skip conditions
- Skips if the labels dataset is missing.
"""
from pathlib import Path
import os
import subprocess
import sys

DATASET = Path("data/kb/labels.jsonl")
MODULE = "src.eval.evaluate_rag"  # run as module so src.* imports resolve


def test_evaluate_rag_cli_smoke(tmp_path):
    if not DATASET.exists():
        import pytest
        pytest.skip("labels.jsonl missing")

    out = tmp_path / "rag_eval"
    cmd = [
        sys.executable,
        "-m",
        MODULE,
        "--dataset",
        str(DATASET),
        "--out",
        str(out),
        "--n",
        "2",
        "--skip-if-no-key",
    ]
    # Ensure working dir is project root so relative paths resolve
    repo_root = Path(__file__).resolve().parents[1]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
    assert r.returncode == 0

    has_key = "OPENAI_API_KEY" in os.environ and bool(
        os.environ["OPENAI_API_KEY"])
    json_path = out / "rag_eval.json"
    csv_path = out / "rag_eval.csv"

    if has_key:
        assert json_path.exists()
        assert csv_path.exists()
    else:
        # Skipped run: artifacts need not exist; check skip message surfaced
        assert "skipping evaluation" in (r.stderr or "").lower()
