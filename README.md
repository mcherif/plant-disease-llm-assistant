---
title: Plant-Disease-LLM-Assistant
emoji: 🌿
pinned: false
short_description: RAG-powered assistant for plant disease guidance
license: mit
tags: ["rag", "llm", "retrieval", "bm25", "vector-search", "streamlit"]
sdk: streamlit
app_file: src/interface/streamlit_app.py
app_port: 7860
---

<!-- Logo/banner at top -->
<p align="center">
  <img src="images/plant-disease-llm-assistant-logo.png" alt="Plant Disease LLM Assistant Logo" width="300"/>
</p>

# 🌿 Plant-Disease-LLM-Assistant

An LLM-powered assistant that helps identify plant disease issues and provides guidance by retrieving from a curated knowledge base (docs, PDFs, web pages). It combines classic and vector retrieval with an LLM to provide grounded answers.

> Note: The Gradio app (src/app_gradio.py) is the working, maintained end-to-end interface. Streamlit is also working but may not be maintained throughout all commits.  FastAPI are used for component tests only.

## Quick start (Docker)

- Compose (recommended):
```powershell
docker compose up --build
```
Open http://localhost:8501

- Plain Docker:
```powershell
docker build -t plant-llm-assistant .
docker run -p 8501:8501 -v ${PWD}:/code plant-llm-assistant
```

## Project structure

```
plant-disease-llm-assistant/
│
├── README.md                  # Overview, setup instructions, project goals
├── docker-compose.yml         # Multi-service container setup
├── requirements.txt           # Python dependencies (with versions pinned)
├── .env.example               # Environment variables template
├── Makefile                   # Optional: shortcuts for build, run, test
│
├── data/                      # Datasets & docs (or scripts to download them)
│   ├── raw/                   # Original PlantVillage or other data
│   ├── processed/             # Preprocessed/augmented data
│   └── kb/                    # Knowledge base docs (PDFs, scraped pages, etc.)
│
├── notebooks/                 # Prototyping, EDA, experiments
│   ├── 01_classifier_review.ipynb
│   ├── 02_doc_ingestion.ipynb
│   └── 03_rag_experiments.ipynb
│
├── src/                       # Core source code
│   ├── classifier/            # Existing image classifier pipeline
│   │   ├── train.py
│   │   ├── infer.py
│   │   └── utils.py
│   │
│   ├── ingestion/             # Scripts to fetch and preprocess knowledge base
│   │   ├── ingest_docs.py     # Load docs into vector DB
│   │   └── pipelines.py       # Prefect/Airflow ingestion flows
│   │
│   ├── retrieval/             # Retrieval logic
│   │   ├── bm25_retriever.py
│   │   ├── vector_retriever.py
│   │   ├── hybrid_retriever.py
│   │   └── evaluation.py      # Retrieval evaluation methods
│   │
│   ├── llm/                   # LLM interaction layer
│   │   ├── rag_pipeline.py    # Orchestration of retrieval + LLM
│   │   ├── query_rewriter.py  # User query rewriting with LLM
│   │   └── evaluation.py      # LLM evaluation strategies
│   │
│   ├── monitoring/            # Metrics and dashboards
│   │   ├── feedback_collector.py
│   │   └── dashboard.py
│   │
│   └── interface/             # UI/API for end users
│       ├── streamlit_app.py   # Streamlit front-end
│       └── api.py             # FastAPI backend (optional)
│
├── tests/                     # Unit & integration tests
│   └── test_retrieval.py
│
└── docs/                      # Documentation, design diagrams
    ├── architecture.png
    └── project_plan.md

```

## Development

- Create env and install:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

- Run Streamlit locally:
```powershell
streamlit run src/interface/streamlit_app.py
```

- Optional API (FastAPI):
```powershell
uvicorn src.interface.api:app --host 0.0.0.0 --port 8000
```

## Dataset
See docs/data_card.md for sources, build steps, and limitations.

### Expanding disease coverage
If you expand the disease classes (e.g., add grape diseases), do the following:
- Update the seed pairs in data/plantvillage_kb.json (via src/ingestion/refresh_kb_descriptions.py or manual edit).
- Extend WIKI_NORMALIZE in src/ingestion/build_kb.py for known name variants (e.g., “haunglongbing” → “huanglongbing”, “leaf mould” → “leaf mold”).
- Relax or adjust the relevance filter (_wiki_is_relevant in src/ingestion/build_kb.py) to accept new pages (e.g., allow grape list/overview pages if useful).
- Rebuild the KB and re-index your retrieval store.

Rebuild command (example):
```powershell
python -m src.ingestion.build_kb --sources plantvillage,wikipedia --out data\kb --min_tokens 50 --max_tokens 400 --overlap 80 --dedup minhash --dedup-threshold 0.9 --wiki-lang en --wiki-interval 0.5 --verbose
```

## Retrieval
See docs/retrieval.md for building the FAISS index, BM25, and hybrid fusion (sum/RRF) with examples.

## Notes

- Place unversioned datasets/docs in data/raw, data/processed, data/kb (see .gitignore).
- Add or update ingestion scripts under src/ingestion and retrieval logic under src/retrieval.
- Configure RAG pipeline in src/llm/rag_pipeline.py.

## License

MIT
