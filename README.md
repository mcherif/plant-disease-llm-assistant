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
  <img src="images/plant-disease-logo.png" alt="Plant Disease Classifier Logo" width="300"/>
</p>

# 🌿 Plant-Disease-LLM-Assistant

An LLM-powered assistant that helps identify plant disease issues and provides guidance by retrieving from a curated knowledge base (docs, PDFs, web pages). It combines classic and vector retrieval with an LLM to provide grounded answers.

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

## Notes

- Place unversioned datasets/docs in data/raw, data/processed, data/kb (see .gitignore).
- Add or update ingestion scripts under src/ingestion and retrieval logic under src/retrieval.
- Configure RAG pipeline in src/llm/rag_pipeline.py.

## License

MIT
