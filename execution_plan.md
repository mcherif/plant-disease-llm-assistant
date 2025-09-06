# Execution Plan — Plant-Disease-LLM Assistant

Legend: `- [ ]` = TODO · `- [x]` = DONE · `- [~]` = IN PROGRESS

---

## Milestone 0 — Foundation & Scaffolding — Done
Goal: Set the repo so everything is reproducible and testable.

- [x] Project scaffold ready: `src/`, `data/`, `docs/`, `tests/`, `models/`, `artifacts/`
- [x] Classifier project cloned/migrated into this repo (reuse components where helpful)
- [x] Dev flow: `docker compose up --build` starts local app (Streamlit UI on :8501)
- [x] Pinned dependencies + `.env.example`
- [x] Basic tests: `pytest -q` runs at least one placeholder test
- [x] Makefile basics (`run`, `test`, etc.)
- [x] README Quick Start + initial architecture diagram/notes
- [x] Testing guide: docs/testing.md (how to run mock vs. OpenAI integration tests; logging tips)
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

## Milestone 2 — Retrieval (Hybrid Search) — Done
Goal: High‑recall retrieval fusing lexical + vector.

- [x] Lexical search
  - [x] Option A: lightweight BM25 (rank_bm25 over candidates)
  - [ ] Option B: Elasticsearch/OpenSearch (containerized) — deferred to Backlog
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
- [x] Evaluation scaffolding
  - [x] Mini labeled set: data/kb/labels.jsonl
  - [x] Script: src/retrieval/evaluate.py → Recall@k, nDCG@k
  - [x] Tests: tests/test_retrieval.py
- [x] Make targets
  - [x] make index (embed + build index)
  - [x] make eval_retrieval (run evaluate.py)

**Acceptance checks**
- [x] Hybrid retrieval measurable & repeatable; metrics saved to `artifacts/`  <!-- artifacts/retrieval_eval/retrieval_eval.json -->

---

## Milestone 3 — LLM Integration (RAG) — Done
Goal: Grounded answers with citations.

- [x] RAG pipeline: `src/llm/rag_pipeline.py`
- [x] Steps: retrieve → select → prompt‑compose → call LLM → postprocess
- [x] Enforce citations (postprocess verifies/ensures [n] within range)
- [x] Refusal/guardrails for empty context (no hits)
  - Early return with a polite refusal + one fallback retrieval without plant filter
  - Test: [`tests/test_guardrails.py`](tests/test_guardrails.py)
- [x] Prompt templates: `src/llm/prompts/`
  - [x] Answer template with explicit citation instructions (`src/llm/prompts/answer.txt`)
  - [x] Tests: required placeholders exist; no missing keys (`tests/test_prompts.py`)
- [x] LLM backends/robustness
  - [x] Temperature and timeout in generation calls
  - [x] Retry with simple exponential backoff
  - [ ] Token budgeting (defer to improvements)
- [x] Tests
  - [x] Unit: mocked LLM (`tests/test_rag_pipeline.py`)
  - [x] Guardrail: refuse on no-context (`tests/test_guardrails.py`)
  - [x] Citation enforcement (`tests/test_citations.py`)
  - [x] Integration smoke (OpenAI) when key is set

**Acceptance checks**
- [x] End‑to‑end: question → grounded answer with citations within target latency

---

## Milestone 4 — RAG evaluation

Goals
- Generate a focused QA dataset from the KB manifest.
- Run RAG end-to-end and score answers with LLM-as-judge (faithfulness, relevance).
- Produce JSON/CSV artifacts for inspection.

What’s done
- [x] Dataset created: data/eval/rag_qa.jsonl (120 prompts; from data/kb/manifest.parquet)
- [x] Evaluation run
  - [x] Command: python -m src.eval.evaluate_rag --dataset data\eval\rag_qa.jsonl --out artifacts\rag_eval --n 120 --skip-if-no-key
  - [x] Judge model: Default gpt-4o-mini (override via OPENAI_MODEL)
  - [x] Artifacts: artifacts/rag_eval/rag_eval.json and artifacts/rag_eval/rag_eval.csv
- [x] Ergonomics: Judge max_tokens capped; optional progress/timeouts

Observations
- Some answers show low faithfulness (claims not fully supported by retrieved context).
- Off-topic drift when plant/disease aren’t enforced in retrieval/prompting.

Queued improvements (tracked for M5.x)
- [ ] Retrieval recall: increase top_k to 4–5; raise ctx_chars to 1600–2000
- [ ] Add reranker (e.g., bge-reranker) over top-20 to select 4–5 best chunks
- [ ] Revisit chunking (700–900 tokens, overlap 120–150)
- [ ] Tighten prompt: “Use only context; if unknown, say so; cite [n]”
- [ ] Temperature 0; optionally drop sentences without citations in postprocess
- [ ] Judge visibility: --save-context flag and partial writes (CSV flush, partial JSON)

Repro/commands
- Generate dataset:
  - python -m src.eval.make_rag_qa
- Evaluate (full):
  - python -m src.eval.evaluate_rag --dataset data\eval\rag_qa.jsonl --out artifacts\rag_eval --n 120 --skip-if-no-key
- Evaluate (fast sanity):
  - python -m src.eval.evaluate_rag --dataset data\eval\rag_qa.jsonl --out artifacts\rag_eval --n 30 --top-k 4 --ctx-chars 1600 --skip-if-no-key

---

## Milestone 5 — Minimal UI/API

What’s done
- [x] Streamlit UI
  - [x] Classifier top-1 auto-fills detected_plant/detected_disease
  - [x] Query enrichment and filters passed to RAG
  - [x] Default question fallback (“What can I do to treat this?”)
  - [x] Sources panel with titles/URLs; Windows Make target
- [x] Docs: docs/STREAMLIT.md and README links
- [x] Pipeline fixes
  - [x] Retrieval hits normalized to dicts with meta (no tuple .get errors)
  - [x] Plant/disease filters applied in retrieval/prompt

Next steps (M5.x)
- [ ] Retrieval: add reranker; expose fusion/alpha in UI
- [ ] Prompting: enforce citations and stricter grounding
- [ ] API: FastAPI /rag endpoint with env-configurable index/top_k
- [ ] Feedback: thumbs up/down logging to data/feedback
- [ ] Update Hugging Face Space/app repo (sync Streamlit app, README, and secrets)

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

---

## Immediate Focus — Skeleton E2E Freeze (before disease expansion)

Objective
Ship a minimal but runnable vertical slice (ingestion → retrieval → RAG → UI/API) that a new user can start in <10 min.

Scope (DO NOW)
- [x] API: minimal FastAPI app (/health, /rag) reusing RAGPipeline
- [ ] Docker: single Dockerfile + docker-compose exposing UI (8501) + API (8000)
- [ ] Make targets: make api, make ui, make run (compose), make sample_index (tiny KB)
- [ ] Tiny sample KB (2 plants, 3 diseases) under data/sample_kb/ + index build
- [ ] README Quick Start (clone → make sample_index → make ui → ask question)
- [ ] Streamlit: link to API usage (curl example) + footer with version/hash
- [ ] Basic telemetry: log answer latency & retrieved count to stdout
- [ ] Execution plan cleanup (remove duplicate legacy section)
- [ ] Add docs/ARCHITECTURE.md (diagram + 1‑page flow)
- [ ] .env.example updated (OPENAI_API_KEY, INDEX_DIR, MODEL_NAME)

Acceptance
- Fresh clone + sample index: user gets an answer with sources (no manual edits)
- All tests pass locally (pytest -q)
- Ruff clean (ruff check .)
- Docker image runs UI + API with sample index

Deferrals (NEXT AFTER FREEZE)
- Disease/plant taxonomy expansion
- Reranker & fusion tuning
- Citation sentence post‑processing
- Feedback logging
- HF Space deployment
- Full KB rebuild & eval reruns

## Backlog (after skeleton)
- Disease expansion (taxonomy list, ingestion seeds, classifier alignment)
- Retrieval reranker (bge-reranker over top 20 → top 5)
- Prompt hardening & citation enforcement per sentence
- Feedback loop (thumbs up/down JSONL)
- HF Space + lightweight demo KB
- Judge enhancements (--save-context, partial flush)

---

## Technical Debt (tracked items)
- [ ] Mojibake / garbled title normalization:
      Replace naive replaces in _clean_title with ingestion‑time UTF-8 verification
      (consider ftfy or strict decode + logging). Add test with a sample malformed title.
- [ ] Centralize text normalization utilities (titles, plant/disease aliases) in a single module.
- [ ] Move citation enforcement + prompt building into dedicated component to simplify RAGPipeline.
