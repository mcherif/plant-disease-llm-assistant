# Project Makefile: common dev tasks (Windows-friendly).
# Requires GNU Make (e.g., Chocolatey: choco install make -y).
# Use TABs (not spaces) to indent recipe lines.

.PHONY: install kb kb-all index eval_retrieval eval_rag

install:
        pip install -r requirements.txt

# Build a KB from PlantVillage into data/kb (chunks + manifest)
# Adjust token bounds/overlap as needed. Wikipedia ingestion is stubbed for now.
kb:
    python -m src.ingestion.build_kb --sources plantvillage --out data/kb --min_tokens 50 --max_tokens 1000 --overlap 100 --dedup minhash --dedup-threshold 0.9 --verbose

# Build PV + Wikipedia (polite)
kb-all:
    python -m src.ingestion.build_kb --sources plantvillage,wikipedia --out data/kb --min_tokens 50 --max_tokens 1000 --overlap 100 --dedup minhash --dedup-threshold 0.9 --wiki-lang en --wiki-interval 0.5 --verbose

# Build FAISS index from the KB manifest
index:
    python -m src.retrieval.build_index --manifest data\kb\manifest.parquet --out models\index\kb-faiss-bge --model BAAI/bge-small-en-v1.5 --batch-size 64 --device cuda --doc-prefix "passage: "

# Offline retrieval evaluation (Recall@k, nDCG@k)
eval_retrieval:
    python -m src.retrieval.evaluate --index-dir models\index\kb-faiss-bge --labels data\kb\labels.jsonl --device cuda --fusion sum --alpha 0.7 --top-ks 5,10

# RAG evaluation with LLM-as-judge (faithfulness/relevance)
# Requires OPENAI_API_KEY; with --skip-if-no-key it exits cleanly when unset.
# Artifacts: artifacts/rag_eval/rag_eval.json and rag_eval.csv
eval_rag:
    python -m src.eval.evaluate_rag --dataset data/kb/labels.jsonl --out artifacts/rag_eval --n 50 --skip-if-no-key
