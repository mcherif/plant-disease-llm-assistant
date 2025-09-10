---
title: Plant-Disease-RAG-Assistant
emoji: 🌿
pinned: false
short_description: RAG-powered assistant for plant disease diagnosis and guidance
license: mit
tags: ["plant", "plant-disease", "plantvillage", "rag", "llm", "retrieval", "bm25", "vector-search", "streamlit", "gradio", "fastapi"]
sdk: streamlit
app_file: src/interface/streamlit_app.py
app_port: 7860
---

<!-- Logo/banner at top -->
<p align="center">
  <img src="images/plant-disease-rag-assistant-logo.png" alt="Plant Disease RAG Assistant Logo" width="300"/>
</p>

# 🌿 Plant-Disease-RAG-Assistant

A Retrieval-Augmented Generation (RAG) assistant for plant disease diagnosis and guidance. It leverages classic (BM25) and vector-based retrieval, orchestrated with an LLM, to provide grounded answers from curated sources (docs, PDFs, web pages). The system supports image classification, document search, and conversational Q&A.

## Features

- **Image classification**: Diagnose plant diseases from uploaded images.
- **Document retrieval**: Search and retrieve relevant information from a knowledge base.
- **Conversational assistant**: Ask questions and receive context-aware, grounded answers.
- **Hybrid retrieval**: Combines BM25 and vector search (FAISS) for robust results.
- **APIs**: RESTful endpoints for integration and automation.
- **Multi-UI support**: Gradio (main), Streamlit (optional), FastAPI backend.

## Quick Start

### Docker Compose (recommended)
```powershell
docker compose up --build
```
- Gradio UI: http://localhost:7860
- Streamlit UI: http://localhost:8501

### Plain Docker
```powershell
docker build -t plant-llm-assistant .
docker run -p 7860:7860 -p 8501:8501 -v ${PWD}:/code plant-llm-assistant
```

## Project Structure

```
plant-disease-llm-assistant/
│
├── README.md                  # Project overview, setup, usage, and documentation links
├── docker-compose.yml         # Multi-container orchestration for API and UI services
├── requirements.txt           # Python dependencies for the project
├── .env.example               # Example environment variables for local/dev setup
├── Makefile                   # Common build, run, and test commands
│
├── data/
│   ├── raw/                   # Unprocessed source datasets (images, docs, etc.)
│   ├── processed/             # Cleaned/normalized data ready for ingestion
│   └── kb/                    # Final knowledge base files (JSON, Parquet, etc.)
│
├── notebooks/
│   ├── 01_classifier_review.ipynb   # EDA and review of image classifier results
│   ├── 02_doc_ingestion.ipynb       # KB ingestion and document processing experiments
│   └── 03_rag_experiments.ipynb     # RAG pipeline and retrieval experiments
│
├── src/
│   ├── classifier/            # Image classification models and utilities
│   ├── ingestion/             # Scripts for scraping, cleaning, and building the KB
│   │   ├── build_kb.py             # Main ingestion script: collects, chunks, normalizes, and deduplicates raw data from sources (PlantVillage, Wikipedia...) to build the initial KB.
│   │   ├── refresh_kb_descriptions.py # Enrichment script: updates or fills in missing fields (description, symptoms, cause, etc.) in the KB by scraping PlantVillage pages.
│   │   ├── validate_kb_urls.py     # Utility script for validating URLs in the KB.
│   │   ├── scrape_plantvillage_infos.py # Scraper for PlantVillage 'infos' pages, used for initial disease/crop info extraction.
│   │   └── ...                     # Other helpers/utilities for KB construction and cleaning.
│   ├── retrieval/             # BM25, FAISS, and hybrid retrieval logic
│   ├── llm/                   # RAG pipeline, LLM integration, and prompt logic
│   ├── monitoring/            # Logging, metrics, and health checks
│   └── interface/
│       ├── streamlit_app.py   # Main Streamlit UI for interactive Q&A and classification
│       ├── app_gradio.py      # (Obsolete) Gradio UI for quick demo/testing; use Streamlit for all new features
│       └── api.py             # FastAPI backend for RESTful endpoints
│
├── tests/
│   └── test_retrieval.py      # Unit tests for retrieval and RAG pipeline
│
└── docs/
    ├── architecture.png       # System architecture diagram
    ├── project_plan.md        # Project goals, milestones, and planning notes
    ├── STREAMLIT.md           # Streamlit UI usage and customization guide
    ├── data_card.md           # Dataset sources, build steps, and limitations
    ├── retrieval.md           # Retrieval pipeline documentation
    ├── testing.md             # Testing and debugging instructions
    ├── improvements.md        # Roadmap, backlog, and enhancement ideas
    └── artifacts.md           # Evaluation results and experiment artifacts
```

## API Endpoints

- **Image Classification**: `/api/classify` (POST, image upload)
- **Document Search**: `/api/search` (POST, query text)
- **Conversational Q&A**: `/api/ask` (POST, question text)
- **Feedback Collection**: `/api/feedback` (POST, feedback data)

See [docs/api_reference.md](docs/api_reference.md) for full details.

## Usage

### Local Development

- Create environment and install dependencies:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

- Run Gradio UI:
```powershell
python src/interface/app_gradio.py
```

- Run Streamlit UI:
```powershell
streamlit run src/interface/streamlit_app.py
```

- Run FastAPI backend:
```powershell
uvicorn src.interface.api:app --host 0.0.0.0 --port 8000
```

### Web UIs

- **Gradio**: Main interface for image classification and Q&A.
- **Streamlit**: Alternative UI for document search and Q&A.
- See [docs/STREAMLIT.md](docs/STREAMLIT.md) for details.

## Dataset

See [docs/data_card.md](docs/data_card.md) for sources, build steps, and limitations.

### Expanding Disease Coverage

- Update seed pairs in `data/plantvillage_kb.json` (via `src/ingestion/refresh_kb_descriptions.py` or manual edit).
- Extend normalization and relevance filters in `src/ingestion/build_kb.py`.
- Rebuild KB and re-index retrieval store:
```powershell
python -m src.ingestion.build_kb --sources plantvillage,wikipedia --out data\kb --min_tokens 50 --max_tokens 400 --overlap 80 --dedup minhash --dedup-threshold 0.9 --wiki-lang en --wiki-interval 0.5 --verbose
```

## Retrieval

See [docs/retrieval.md](docs/retrieval.md) for building FAISS index, BM25, and hybrid fusion. Evaluation artifacts in [docs/artifacts.md](docs/artifacts.md).

## Testing

See [docs/testing.md](docs/testing.md) for running tests, OpenAI integration, and debugging tips.

## Roadmap

See [docs/improvements.md](docs/improvements.md) for prioritized backlog and test ideas.

## Notes

- Place unversioned datasets/docs in `data/raw`, `data/processed`, `data/kb`.
- Add/update ingestion scripts under `src/ingestion`, retrieval logic under `src/retrieval`.
- Configure RAG pipeline in `src/llm/rag_pipeline.py`.

## License

MIT
