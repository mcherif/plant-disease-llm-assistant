install:
		pip install -r requirements.txt

# Make targets for local development. Run with `make <target>`.

# Build a KB from PlantVillage into data/kb (chunks + manifest)
# Adjust token bounds/overlap as needed. Wikipedia ingestion is stubbed for now.
kb:
	python -m src.ingestion.build_kb --sources plantvillage --out data/kb --min_tokens 50 --max_tokens 1000 --overlap 100 --dedup minhash --dedup-threshold 0.9 --verbose

# Build PV + Wikipedia (polite)
kb-all:
	python -m src.ingestion.build_kb --sources plantvillage,wikipedia --out data/kb --min_tokens 50 --max_tokens 1000 --overlap 100 --dedup minhash --dedup-threshold 0.9 --wiki-lang en --wiki-interval 0.5 --verbose

index:
	python -m src.retrieval.build_index --manifest data\kb\manifest.parquet --out models\index\kb-faiss-bge --model BAAI/bge-small-en-v1.5 --batch-size 64 --device cuda --doc-prefix "passage: "

eval_retrieval:
	python -m src.retrieval.evaluate --index-dir models\index\kb-faiss-bge --labels data\kb\labels.jsonl --device cuda --fusion sum --alpha 0.7 --top-ks 5,10
