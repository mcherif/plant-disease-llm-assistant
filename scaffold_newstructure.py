import os

structure = [
    "data/raw",
    "data/processed",
    "data/kb",
    "notebooks",
    "src/classifier",
    "src/ingestion",
    "src/retrieval",
    "src/llm",
    "src/monitoring",
    "src/interface",
    "tests",
    "docs"
]

files = {
    "README.md": "# Plant Disease LLM Assistant\n\nExtends the Plant Disease Classifier into a Retrieval-Augmented QA assistant for farmers and gardeners.\n",
    "requirements.txt": "transformers\nprefect\nmlflow\nstreamlit\nfaiss-cpu\nchromadb\nfastapi\nuvicorn\nscikit-learn\npandas\nnumpy\nmatplotlib\n",
    "docker-compose.yml": "version: '3.9'\nservices:\n  app:\n    build: .\n    command: streamlit run src/interface/streamlit_app.py\n    ports:\n      - '8501:8501'\n    volumes:\n      - .:/code\n",
    ".env.example": "# Add API keys here\nOPENAI_API_KEY=your_key_here\n",
    "src/ingestion/ingest_docs.py": "# Script to load docs into vector DB\n",
    "src/retrieval/bm25_retriever.py": "# BM25 retriever\n",
    "src/retrieval/vector_retriever.py": "# Vector retriever using embeddings\n",
    "src/retrieval/hybrid_retriever.py": "# Hybrid retriever (BM25 + vector)\n",
    "src/retrieval/evaluation.py": "# Retrieval evaluation methods\n",
    "src/llm/rag_pipeline.py": "# RAG orchestration pipeline\n",
    "src/llm/query_rewriter.py": "# LLM query rewriting\n",
    "src/llm/evaluation.py": "# LLM evaluation methods\n",
    "src/monitoring/feedback_collector.py": "# Collect user feedback\n",
    "src/monitoring/dashboard.py": "# Dashboard with charts\n",
    "src/interface/streamlit_app.py": "# Streamlit UI for demo\n",
    "src/interface/api.py": "# FastAPI backend\n",
    "tests/test_retrieval.py": "# Tests for retrieval pipeline\n",
    "docs/project_plan.md": "# Project Plan\n"
}

# Create folders
for folder in structure:
    os.makedirs(folder, exist_ok=True)

# Create files
for filepath, content in files.items():
    with open(filepath, "w") as f:
        f.write(content)

print("Project scaffold created successfully!")
