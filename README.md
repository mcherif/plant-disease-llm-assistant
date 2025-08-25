---
title: Plant-Disease-LLM-Assistant
emoji: 🌿
pinned: false
short_description: RAG-powered assistant for plant disease guidance
license: mit
tags: ["rag", "llm", "retrieval", "bm25", "vector-search", "streamlit", "fastapi", "mlops"]
sdk: docker
app_file: src/interface/streamlit_app.py
app_port: 8501
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
├── README.md
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── Makefile
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── kb/
│
├── notebooks/
│   ├── 01_classifier_review.ipynb
│   ├── 02_doc_ingestion.ipynb
│   └── 03_rag_experiments.ipynb
│
├── src/
│   ├── classifier/
│   ├── ingestion/
│   ├── retrieval/
│   ├── llm/
│   ├── monitoring/
│   └── interface/
│       ├── streamlit_app.py
│       └── api.py
│
├── tests/
│   └── test_retrieval.py
└── docs/
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