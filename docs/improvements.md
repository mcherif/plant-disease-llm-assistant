# Improvements Roadmap

Themes and tasks derived from recent work. Includes rationale and test ideas.

1) Guardrails and refusals
- Insufficient context: if no hits or scores below threshold, return a brief refusal with guidance.
- Tests: simulate empty/low-score retrieval; assert refusal message schema.

2) Citation enforcement
- Validate output only cites [n] within 1..len(sources).
- Fixer: append missing citation or trigger one-shot regenerate with a stricter instruction.
- Tests: mock LLM without citations; ensure fixer corrects or regenerates.

3) Robust LLM calls
- Add retries with exponential backoff and per-request timeout.
- Make temperature and max_tokens configurable (env/args).
- Tests: monkeypatch client to fail first → succeed on retry.

4) Retrieval quality
- Add MMR reranking (config: rerank=mmr, lambda=0.5).
- Tests: verify selected contexts differ vs. baseline and stay within top_k.

5) Developer UX
- make rag_local: run a few demo queries; pretty-print answers and sources.
- CLI wrapper: src/cli/rag_demo.py reading queries from a file.

6) Generation evaluation
- Script src/eval/evaluate_rag.py: LLM-as-judge for faithfulness/relevance on ~50–100 QA.
- Artifacts: CSV/plots under artifacts/rag_eval/.
- Pytest marker to skip in CI unless OPENAI_API_KEY is set.

7) Telemetry/observability
- Log latency, token counts, prompt/answer sizes; write to artifacts/logs/.
- Optional: basic trace IDs; prepare for Langfuse/OpenTelemetry later.

8) Optional backends
- HF/local backend (transformers) for offline runs; config switch backend=openai|hf.
- Caching and model selection guidance for small CPU-friendly models.

9) UI follow-up
- Streamlit: connect to pipeline; audience selector (farmer/gardener); clickable sources.
- Basic error banners for guardrails/refusals.

Acceptance notes
- Each task comes with unit tests or smoke tests, and produces artifacts where applicable.
- Update execution_plan.md as items land (Done/In Progress).