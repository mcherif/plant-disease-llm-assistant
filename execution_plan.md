# Execution Plan — Plant-Disease-LLM Assistant

**Milestone 0 is DONE ✅.** This plan tracks remaining work with clear acceptance checks.  
Legend: `- [ ]` = TODO · `- [x]` = DONE · `- [~]` = IN PROGRESS

---

## Milestone 0 — Foundation & Scaffolding (DONE)
Goal: Set the repo so everything is reproducible and testable.

- [x] Project scaffold ready: `src/`, `data/`, `docs/`, `tests/`, `models/`, `artifacts/`
- [x] Classifier project cloned/migrated into this repo (reuse components where helpful)
- [x] Dev flow: `docker compose up --build` starts local app (Streamlit UI on :8501)
- [x] Pinned dependencies + `.env.example`
- [x] Basic tests: `pytest -q` runs at least one placeholder test
- [x] Makefile basics (`run`, `test`, etc.)
- [x] README Quick Start + initial architecture diagram/notes
- [x] Data wrangling kick-off: scraper started for PlantVillage & Wikipedia

**Acceptance checks**
- [x] Fresh clone → `docker compose up --build` boots the UI locally
- [x] `pytest -q` succeeds on the skeleton tests

---

## Milestone 1 — Knowledge Base (KB) Corpus — Done
Goal: Build a clean, deduplicated, well‑tagged KB from PlantVillage + Wikipedia.
Status: Done (first version). Ongoing: normalization polish and more tests.

How to build (current state):
- PV + Wikipedia:
  - python -m src.ingestion.build_kb --sources plantvillage,wikipedia --out data\kb --min_tokens 50 --max_tokens 1000 --overlap 100 --dedup minhash --dedup-threshold 0.9 --wiki-lang en --wiki-interval 0.5 --verbose
- Outputs:
  - data/kb/chunks/*.md (chunk files)
  - data/kb/manifest.parquet (or CSV fallback)

- src/ingestion/refresh_kb_descriptions.py (PV refresher: summary/symptoms/cause)
- src/ingestion/validate_kb_urls.py (URL validator/fixer)
- scripts/plantvillage_scraper.py, scripts/verify_scoped_parsing.py, scripts/parse_html_debug.py (dev tools)

- [~] Ingestion entrypoint
  - [x] PlantVillage refresher: src/ingestion/refresh_kb_descriptions.py
  - [x] Unified builder: src/ingestion/build_kb.py (scaffold with CLI and chunking)
  - [x] CLI: --sources plantvillage,wikipedia --out data/kb --min_tokens 50 --max_tokens 1000 --overlap 100
  - [x] Respect robots.txt, polite rate limiting, retry/backoff
  - [~] Normalize to markdown/text; strip boilerplate/nav
  - [x] Metadata: doc_id, url, title, plant, disease, section, lang, crawl_date
- [x] Chunking (512–1,000 tokens), sentence‑aware splits + overlap
- [x] Deduplication (MinHash/LSH) across pages and chunks
- [x] Manifest: data/kb/manifest.parquet with  
      doc_id, url, title, plant, disease, split_idx, text, n_tokens, lang, crawl_date
- [x] Make target: make kb (runs build_kb, writes manifest + chunks)
- [x] Data card: docs/data_card.md (sources, licenses, cleaning steps, limitations)
- [~] Unit tests: tests/test_ingestion.py (chunk lengths, metadata presence, dedup sanity)

**Acceptance checks**
- [x] One command builds the KB end‑to‑end
- [x] Spot‑check 10 random chunks → clean text & correct tags

---

## Milestone 2 — Retrieval (Hybrid Search)
Goal: High‑recall retrieval fusing lexical + vector.

- [~] Lexical search
  - [x] Option A: lightweight BM25 (rank_bm25 over candidates)
  - [ ] Option B: Elasticsearch/OpenSearch (containerized)
- [x] Vector search
  - [x] Embedding model via config
  - [x] Build FAISS index → persist under models/index/*
  - [x] Script: src/retrieval/build_index.py to embed + index manifest chunks
- [x] Hybrid fusion
  - [x] Reciprocal Rank Fusion (RRF) or weighted score sum
  - [x] Configurable top_k and fusion weights
- [x] Filters
  - [x] Plant/disease metadata filters
  - [x] Deterministic seeds not needed (pure retrieval)
- [~] Evaluation scaffolding
  - [x] Mini labeled set: data/kb/labels.jsonl
  - [x] Script: src/retrieval/evaluate.py → Recall@k, nDCG@k
  - [ ] Tests: tests/test_retrieval.py
- [x] Make targets
  - [x] make index (embed + build index)
  - [x] make eval_retrieval (run evaluate.py)

**Acceptance checks**
- [x] Hybrid retrieval measurable & repeatable; metrics saved to `artifacts/`  <!-- artifacts/retrieval_eval/retrieval_eval.json -->

---

## Milestone 3 — LLM Integration (RAG)
Goal: Grounded answers with citations.

- [ ] RAG pipeline: `src/llm/rag_pipeline.py`
  - [ ] Steps: retrieve → select → prompt‑compose → call LLM → postprocess
  - [ ] Enforce citations (every claim comes from context with source URLs/titles)
  - [ ] Refusal/guardrails for out‑of‑scope or insufficient context
- [ ] Prompt templates: `src/llm/prompts/`
  - [ ] Answer template with explicit citation slots
  - [ ] System rules: factual, concise, cite; no speculation
  - [ ] Tests: required placeholders exist; no missing keys
- [ ] LLM backends
  - [ ] Pluggable: OpenAI / HF Inference / local
  - [ ] Retry + backoff; timeouts; token budgeting
- [ ] Chunk re‑ranking (e.g., MMR) before prompting
- [ ] Make target: `make rag_local` (demo Q&A on a few queries)

**Acceptance checks**
- [ ] End‑to‑end: question → grounded answer with citations within target latency

---

## Milestone 4 — Evaluation & Monitoring
Goal: Objective evaluation & basic observability.

- [ ] Offline evaluation
  - [ ] Retrieval: Recall@k, nDCG@k (from Milestone 2)
  - [ ] Generation: faithfulness & relevance via LLM‑as‑judge on 50–100 QA
  - [ ] Ablations: lexical‑only vs vector‑only vs hybrid
  - [ ] Save CSVs/plots to `artifacts/`
- [ ] Tracing/telemetry
  - [ ] Log latency, token counts, cache hits, retrieved doc IDs
  - [ ] Optional: Langfuse/OpenTelemetry integration
- [ ] Feedback loop
  - [ ] UI thumbs‑up/down + comment → `data/feedback/*.jsonl`
  - [ ] Aggregation script → `artifacts/feedback_report.csv`
- [ ] Make target: `make eval_rag` (runs LLM‑judge → `artifacts/rag_eval.csv`)

**Acceptance checks**
- [ ] Reproducible eval runs with artifacts + README section on interpreting results

---

## Milestone 5 — Interface & Serving
Goal: End‑to‑end app + API for demo.

- [ ] Streamlit UI: `src/interface/streamlit_app.py`
  - [ ] Inputs: query, plant, disease, top_k, model selection
  - [ ] Display: final answer with citations; expandable retrieved chunks
  - [ ] Feedback widget (thumbs + comment) → `data/feedback/`
  - [ ] Safety banner: “Educational tool — not medical advice”
- [ ] FastAPI: `src/interface/api.py`
  - [ ] Endpoints: `/healthz`, `/search`, `/rag`
  - [ ] Pydantic schemas
- [ ] Docker/Compose
  - [ ] Compose services: api, ui, (optional) vector DB / ES
  - [ ] One‑liner run: `docker compose up --build`

**Acceptance checks**
- [ ] Fresh clone → compose up → ask “apple scab” → grounded, cited answer appears

---

## Milestone 6 — Capstone Packaging (LLM Zoomcamp)
Goal: Meet rubric; easy to run, understand, and evaluate.

- [ ] Documentation
  - [ ] README: problem, users, how‑to‑run, evaluation, design decisions, limitations
  - [ ] Architecture doc: `docs/architecture.md` + `docs/architecture.png`
  - [ ] Data card + licensing notes
- [ ] Reproducibility
  - [ ] Make targets; pinned versions; seeds; `.env.example`
  - [ ] Minimal sample KB for quick demos
- [ ] CI
  - [ ] GitHub Actions: ruff/flake8 + pytest (+ optional mypy)
  - [ ] Cache embeddings between runs if feasible
- [ ] Demo
  - [ ] 2–5 min video (ingestion → retrieval → RAG → UI)
  - [ ] Optional hosted demo (HF Space/small VM)
- [ ] Submission checklist aligned with course rubric

**Acceptance checks**
- [ ] Reviewer can clone, run, reproduce key metrics & watch the demo

---

## Cross‑Cutting Improvements / Hardening
(Implement alongside milestones when convenient.)

- [ ] Centralized config (`src/common/config.py` using Pydantic Settings)
  - [ ] Paths, model names, k‑values, timeouts, feature flags
  - [ ] `.env.example` validation (fail fast on missing keys)
- [ ] Index persistence
  - [ ] Persist FAISS/Qdrant to `models/index/` with versioned filenames
  - [ ] `make rebuild_index` to force re‑embed
- [ ] Data governance
  - [ ] Keep `source_attribution` in manifest; respect licenses
  - [ ] Document dedup/cleaning in data card
- [ ] Prompt/versioning
  - [ ] Version prompts; log prompt IDs with outputs
- [ ] Telemetry
  - [ ] CSV logs by default; optional Langfuse if configured
- [ ] Testing
  - [ ] Unit tests for ingestion, retrieval, prompts, RAG pipeline
  - [ ] CI smoke test: 3 canonical queries (skip external calls if keys absent)
- [ ] CI/CD
  - [ ] PR checks: lint + tests (block on red)
  - [ ] Optional: Docker build + push on `main`
- [ ] Security & safety
  - [ ] Input size limits; URL sanitization
  - [ ] Refuse medical diagnosis; always cite sources
- [ ] Performance
  - [ ] Caching for embeddings & retrieval responses
  - [ ] Batch embedding; chunk‑level cache keys

---

## Backlog / Stretch Goals (Optional)
- [ ] Train a lightweight cross‑encoder reranker on labeled set
- [ ] Multilingual query support (FR/AR/DE)
- [ ] Incremental ingestion (delta crawl) + re‑embed only changed chunks
- [ ] Active‑learning loop using user feedback to improve retrieval
- [ ] Deploy a tiny demo KB to Hugging Face Spaces

---

## Suggested GitHub Issues (titles you can paste)
- [ ] Milestone 1: Build KB ingestion pipeline
- [ ] Milestone 2: Implement hybrid retrieval (BM25 + FAISS) with RRF
- [ ] Milestone 2: Create mini labeled set and retrieval evaluator
- [ ] Milestone 3: Implement RAG pipeline with strict citations
- [ ] Milestone 3: Add prompt templates + tests
- [ ] Milestone 4: RAG evaluation (LLM‑as‑judge) + artifacts
- [ ] Milestone 4: Add tracing/logging + feedback sink
- [ ] Milestone 5: Streamlit UI + feedback widget
- [ ] Milestone 5: FastAPI endpoints (/search, /rag) + schemas
- [ ] Milestone 6: Documentation & demo video
- [ ] Hardening: Centralized config with Pydantic
- [ ] Hardening: Persisted vector index + Make targets
- [ ] Hardening: CI (ruff/pytest) + smoke test workflow
