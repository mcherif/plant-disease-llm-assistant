# Retrieval Guide (BM25 + FAISS + Hybrid Fusion)

This project uses hybrid retrieval: dense vectors (FAISS) + lexical BM25, then late‑fusion re‑ranking.

## Components

- FAISS (Facebook AI Similarity Search)
  - Builds a vector index over chunk embeddings.
  - We use cosine similarity via inner product on normalized vectors.
- Sentence embeddings
  - Default: sentence-transformers/all-MiniLM-L6-v2 (384‑dim, fast).
  - Stronger alt: BAAI/bge-small-en-v1.5 (needs prefixes).
- BM25 (rank_bm25)
  - Classic keyword scoring. Great for exact terms and proper nouns.
- Fusion
  - Weighted sum: normalize scores per query, then fused = alpha*vec + (1-alpha)*bm25.
  - RRF (Reciprocal Rank Fusion): combine ranks; robust across score scales.

## Indexing

Build an index from the KB manifest (GPU for embedding, CPU FAISS is fine):

```powershell
# MiniLM (no prefixes)
python -m src.retrieval.build_index --manifest data\kb\manifest.parquet ^
  --out models\index\kb-faiss ^
  --model sentence-transformers/all-MiniLM-L6-v2 ^
  --batch-size 64 --device cuda

# BGE with prefixes (recommended for better ranking)
python -m src.retrieval.build_index --manifest data\kb\manifest.parquet ^
  --out models\index\kb-faiss-bge ^
  --model BAAI/bge-small-en-v1.5 ^
  --batch-size 64 --device cuda ^
  --doc-prefix "passage: "
```

Notes
- BGE/E5/GTE families expect prefixes: docs use “passage: ”, queries use “query: ”.
- Config is saved to models/index/.../config.json (includes model, dim, doc_prefix, etc.).

## Searching

Basic vector search:

```powershell
# MiniLM index
python -m src.retrieval.search --index-dir models\index\kb-faiss ^
  --query "tomato yellow leaf curl symptoms" --top-k 5 --device cuda
```

Hybrid fusion with BGE (auto adds "query: " if docs used "passage: "):

```powershell
python -m src.retrieval.search --index-dir models\index\kb-faiss-bge ^
  --query "tomato yellow leaf curl symptoms" --top-k 5 --device cuda ^
  --plant Tomato ^
  --fusion sum --alpha 0.7
# or RRF fusion:
python -m src.retrieval.search --index-dir models\index\kb-faiss-bge ^
  --query "tomato yellow leaf curl symptoms" --top-k 5 --device cuda ^
  --plant Tomato ^
  --fusion rrf
```

Useful flags
- --plant / --disease: filter results by metadata.
- --pretopk: number of FAISS candidates before fusion/filtering (e.g., 50–200).
- --query-prefix: manually set “query: ” if needed.

## How fusion works (short)

- Retrieve K' candidates from FAISS (cosine on normalized vectors).
- BM25 scores computed over the same candidates (tokenized text).
- Weighted sum: min‑max normalize both per query, then combine with alpha.
- RRF: combine by reciprocal ranks: 1/(k + rank_vec) + 1/(k + rank_bm25).

Why it helps
- Vectors capture paraphrases/synonyms (e.g., “TYLCV” ~ “yellow leaf curl”).
- BM25 rewards exact term matches and names.
- Fusion is often more robust across query types.

## GPU vs CPU

- Biggest win: GPU for embeddings during indexing and for query encoding.
- FAISS GPU on Windows is optional and more complex; CPU FAISS is fine for small/medium KBs.

## Troubleshooting

- Model path error on Windows: use forward slashes for HF IDs (sentence-transformers/all-MiniLM-L6-v2), not backslashes.
- Parquet read error: install a reader (pyarrow or fastparquet).
- Empty index: ensure data\kb\manifest.parquet exists and has “text” with length ≥ --min-chars.
