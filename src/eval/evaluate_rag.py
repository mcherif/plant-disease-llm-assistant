"""Evaluate RAG answers with an LLM-as-judge (faithfulness and relevance).

This CLI runs the RAG pipeline over a dataset of prompts and asks an LLM to
score each answer using only the retrieved context. It saves per-query and
aggregate metrics as JSON and CSV under artifacts/rag_eval/.

Usage (Windows):
  python -m src.eval.evaluate_rag --dataset data\kb\labels.jsonl --out artifacts\rag_eval --n 50 --skip-if-no-key

Inputs
- --dataset: JSONL with fields: question (or query), plant (optional).
- --index:   Retrieval index directory (default: models/index/kb-faiss-bge).
- --top-k:   Number of chunks used as context (default: 3).

Outputs
- artifacts/rag_eval/rag_eval.json: { meta, aggregate, per_query[] }.
- artifacts/rag_eval/rag_eval.csv:  per-query rows (question, plant, scores, reasoning, answer).

Environment
- OPENAI_API_KEY (required unless --skip-if-no-key is set).
- OPENAI_MODEL (optional; judge model override; default: gpt-4o-mini).

Exit codes
- 0: success (or skipped when --skip-if-no-key and no key set).
- 2: setup error (missing key without skip, or SDK unavailable).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.llm.rag_pipeline import RAGPipeline, RetrievalConfig

try:
    from openai import OpenAI  # OpenAI Python SDK v1.x
except Exception:
    OpenAI = None  # type: ignore


@dataclass
class JudgeScores:
    faithfulness: float  # 0.0–1.0
    relevance: float     # 0.0–1.0
    reasoning: str       # brief explanation


def _load_jsonl(p: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _judge_prompt(question: str, answer: str, contexts: List[str]) -> str:
    # Keep prompt compact; contexts truncated earlier
    ctx_joined = "\n\n".join(f"- {c}" for c in contexts)
    return (
        "You are an evaluator. Score the assistant's answer using ONLY the provided context.\n"
        "Return a strictly valid JSON object with keys: faithfulness (0..1), relevance (0..1), reasoning (short string).\n\n"
        f"Question:\n{question}\n\n"
        f"Assistant answer:\n{answer}\n\n"
        f"Context snippets (evidence, may be partial):\n{ctx_joined}\n\n"
        "Instructions:\n"
        "- Faithfulness: 1 if claims are supported by context; 0 if contradicted/unsupported; allow partial credit.\n"
        "- Relevance: 1 if answer addresses the question; 0 if off-topic; allow partial credit.\n"
        "- Be strict: if unsure, lower faithfulness.\n"
        "Output JSON only."
    )


def _truncate(s: str, max_chars: int) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_chars]


def _judge(client: Any, model: str, question: str, answer: str, contexts: List[str]) -> JudgeScores:
    prompt = _judge_prompt(question, answer, contexts)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise, strict evaluator returning valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=300,
    )
    content = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(content)
        return JudgeScores(
            faithfulness=float(data["faithfulness"]),
            relevance=float(data["relevance"]),
            reasoning=str(data.get("reasoning", ""))[:500],
        )
    except Exception:
        # Fallback parse for minor formatting mistakes
        m = re.search(r"\{.*\}", content, flags=re.S)
        if not m:
            raise RuntimeError(f"Judge did not return JSON: {content}")
        data = json.loads(m.group(0))
        return JudgeScores(
            faithfulness=float(data["faithfulness"]),
            relevance=float(data["relevance"]),
            reasoning=str(data.get("reasoning", ""))[:500],
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate RAG answers with LLM-as-judge.")
    ap.add_argument("--dataset", type=Path, required=True,
                    help="Path to JSONL with fields: question, plant (optional).")
    ap.add_argument("--index", type=Path, default=Path("models/index/kb-faiss-bge"),
                    help="Retrieval index directory.")
    ap.add_argument("--out", type=Path, default=Path("artifacts/rag_eval"),
                    help="Output directory for artifacts.")
    ap.add_argument("--n", type=int, default=50,
                    help="Max examples to evaluate (cap).")
    ap.add_argument("--judge-model", type=str, default=os.getenv(
        "OPENAI_MODEL", "gpt-4o-mini"), help="Judge model name.")
    ap.add_argument("--top-k", type=int, default=3,
                    help="Top-k chunks for RAG.")
    ap.add_argument("--ctx-chars", type=int, default=1200,
                    help="Max total characters of context shown to judge.")
    ap.add_argument("--skip-if-no-key", action="store_true",
                    help="Exit 0 if OPENAI_API_KEY is missing.")
    ap.add_argument("--timeout", type=float, default=30.0,
                    help="Per OpenAI call timeout (seconds).")
    ap.add_argument("--progress-every", type=int, default=5,
                    help="Print progress every N examples.")
    args = ap.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        if args.skip_if_no_key:
            print("OPENAI_API_KEY not set; skipping evaluation.", file=sys.stderr)
            return 0
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        return 2
    if OpenAI is None:
        print("ERROR: openai SDK not available.", file=sys.stderr)
        return 2

    rows = _load_jsonl(args.dataset)[: args.n]
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = RetrievalConfig(index_dir=args.index, top_k=args.top_k, device="cpu")
    rag = RAGPipeline(cfg)
    client = OpenAI(api_key=api_key, timeout=args.timeout)

    per_rows: List[Dict[str, Any]] = []
    agg = {"faithfulness": 0.0, "relevance": 0.0}
    count = 0

    t0 = time.perf_counter()
    total_n = len(rows)
    for i, ex in enumerate(rows, start=1):
        q = ex.get("question") or ex.get("query") or ""
        plant = ex.get("plant")
        if not q:
            continue

        res = rag.answer(q, plant=plant)
        answer = res["answer"]
        retrieved = res.get("retrieved", [])
        # Build compact context for the judge
        ctx_texts: List[str] = []
        total = 0
        for item in retrieved:
            t = str(item["meta"].get("text", ""))
            tt = _truncate(t, max(100, args.ctx_chars // max(1, args.top_k)))
            ctx_texts.append(tt)
            total += len(tt)
            if total >= args.ctx_chars:
                break

        try:
            scores = _judge(client, args.judge_model, q, answer, ctx_texts)
        except Exception as e:
            scores = JudgeScores(
                faithfulness=0.0, relevance=0.0, reasoning=f"judge_error: {e}")

        per_rows.append({
            "question": q,
            "plant": plant or "",
            "answer": answer,
            "faithfulness": round(scores.faithfulness, 3),
            "relevance": round(scores.relevance, 3),
            "reasoning": scores.reasoning,
        })
        agg["faithfulness"] += scores.faithfulness
        agg["relevance"] += scores.relevance
        count += 1

        if i % max(1, args.progress_every) == 0:
            elapsed = time.perf_counter() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (total_n - i) / rate if rate > 0 else float("inf")
            print(f"[{i}/{total_n}] avg F={agg['faithfulness']/i:.2f} R={agg['relevance']/i:.2f} | "
                  f"{elapsed:.1f}s elapsed, ~{eta:.1f}s ETA", file=sys.stderr, flush=True)

    if count:
        agg["faithfulness"] = round(agg["faithfulness"] / count, 3)
        agg["relevance"] = round(agg["relevance"] / count, 3)

    # Save CSV
    csv_path = args.out / "rag_eval.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
                           "question", "plant", "faithfulness", "relevance", "reasoning", "answer"])
        w.writeheader()
        for r in per_rows:
            w.writerow(r)

    # Save JSON
    out_json = {
        "meta": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "index_dir": str(args.index).replace("\\", "/"),
            "top_k": args.top_k,
            "ctx_chars": args.ctx_chars,
            "judge_model": args.judge_model,
            "dataset": str(args.dataset).replace("\\", "/"),
            "version": "1",
        },
        "aggregate": {"faithfulness": agg["faithfulness"], "relevance": agg["relevance"]},
        "per_query": per_rows,
    }
    (args.out / "rag_eval.json").write_text(json.dumps(out_json,
                                                       ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {csv_path} and {args.out/'rag_eval.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
