install:
		pip install -r requirements.txt

# Make targets for local development. Run with `make <target>`.

# Build a KB from PlantVillage into data/kb (chunks + manifest)
# Adjust token bounds/overlap as needed. Wikipedia ingestion is stubbed for now.
kb:
	python -m src.ingestion.build_kb --sources plantvillage --out data/kb --min_tokens 50 --max_tokens 1000 --overlap 100 --dedup minhash --dedup-threshold 0.9 --verbose
