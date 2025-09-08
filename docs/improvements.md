# Improvements Roadmap

Themes and tasks derived from recent work. Includes rationale and test ideas.

1) Citation enforcement
- Validate output only cites [n] within 1..len(sources).
- Fixer: append missing citation or trigger one-shot regenerate with a stricter instruction.
- Tests: mock LLM without citations; ensure fixer corrects or regenerates.

2) Robust LLM calls
- Add retries with exponential backoff and per-request timeout.
- Make temperature and max_tokens configurable (env/args).
- Tests: monkeypatch client to fail first → succeed on retry.

3) Retrieval quality
- Add MMR reranking (config: rerank=mmr, lambda=0.5).
- Tests: verify selected contexts differ vs. baseline and stay within top_k.

4) Developer UX
- make rag_local: run a few demo queries; pretty-print answers and sources.
- CLI wrapper: src/cli/rag_demo.py reading queries from a file.

5) Generation evaluation
- Script src/eval/evaluate_rag.py: LLM-as-judge for faithfulness/relevance on ~50–100 QA.
- Artifacts: CSV/plots under artifacts/rag_eval/.
- Pytest marker to skip in CI unless OPENAI_API_KEY is set.

6) Telemetry/observability
- Log latency, token counts, prompt/answer sizes; write to artifacts/logs/.
- Optional: basic trace IDs; prepare for Langfuse/OpenTelemetry later.

7) Optional backends
- HF/local backend (transformers) for offline runs; config switch backend=openai|hf.
- Caching and model selection guidance for small CPU-friendly models.

8) UI follow-up
- Streamlit: connect to pipeline; audience selector (farmer/gardener); clickable sources.
- Basic error banners for guardrails/refusals.

Acceptance notes
- Each task comes with unit tests or smoke tests, and produces artifacts where applicable.
- Update execution_plan.md as items land (Done/In Progress).

## Current evaluation snapshot (Milestone 4 kick-off)
- Dataset: data/kb/labels.jsonl
- Index: models/index/kb-faiss-bge
- Judge model: gpt-4o-mini
- RAG params: top_k=3, ctx_chars=1200
- Artifacts: artifacts/rag_eval/rag_eval.json and rag_eval.csv
- Aggregate (n=2):
  - faithfulness: 1.0
  - relevance: 1.0
Notes:
- Tiny sample; expand to 50–100 queries for meaningful averages.
- Headings like “Action Steps” are induced by the answer template; edit src/llm/prompts/answer.txt to change style.

## Next tasks (Milestone 4 — Evaluation & Monitoring)
- Offline evaluation
  - [ ] Expand dataset to 50–100 mixed queries (symptoms, treatment, prevention; multiple plants).
  - [ ] Add edge cases (ambiguous queries, low-context topics, and guardrail/refusal cases).
  - [ ] Run ablations (lexical-only vs vector-only vs hybrid) and compare averages.
  - [ ] Save artifacts per run under artifacts/rag_eval with metadata; add simple plots/summary.
- Judge robustness
  - [ ] Harden JSON parsing; validate schema and clamp scores to [0,1].
  - [ ] Optional: add deterministic string-match sanity checks alongside the judge.
- Telemetry
  - [ ] Log latency, token counts, prompt/answer sizes to artifacts/logs/runs.csv.
  - [ ] Add simple trace IDs to correlate retrieval, generation, and judge entries.
- Reporting
  - [ ] Small plotting script or notebook to visualize score distributions and outliers.
  - [ ] README section on interpreting results and common failure modes.
- Feedback loop
  - [ ] UI thumbs‑up/down + comment → data/feedback/*.jsonl
  - [ ] Aggregation script → artifacts/feedback_report.csv

## Completed
- [x] Guardrails/refusals: no-context early return with plant-filter fallback (tests/test_guardrails.py)
- [x] Citation enforcement: append [1] when missing; ensure [n] within range (tests/test_citations.py)
- [x] OpenAI call robustness: retry/backoff + configurable temperature/timeout
