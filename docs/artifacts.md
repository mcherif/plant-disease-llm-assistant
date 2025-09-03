# Artifacts Reference

## Retrieval evaluation (artifacts/retrieval_eval/)
Outputs produced by src/retrieval/evaluate.py. This folder is ignored by Git (see .gitignore); use it for local runs and CI artifacts.

Expected files
- retrieval_eval.json (required): metrics and metadata for the run.
- aggregate.csv, per_query.csv (optional): tabular dumps for spreadsheets.
- plot_* (optional): simple charts of Recall@k / nDCG@k.

JSON schema (retrieval_eval.json)
{
  "meta": {
    "index_dir": "models/index/kb-faiss-bge",
    "fusion": "sum",           // or "rrf", "none"
    "alpha": 0.7,              // when fusion == "sum"
    "top_ks": [5, 10],
    "device": "cpu",
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "timestamp": "2025-09-03T12:34:56Z",
    "version": "1"
  },
  "aggregate": {
    "5":  { "recall": 0.92, "ndcg": 0.88 },
    "10": { "recall": 0.97, "ndcg": 0.91 }
  },
  "per_query": [
    {
      "query": "tomato yellow leaf curl symptoms",
      "plant": "Tomato",
      "metrics": {
        "5":  { "recall": 1.0, "ndcg": 1.0 },
        "10": { "recall": 1.0, "ndcg": 1.0 }
      }
    }
  ]
}

Notes
- Keys under aggregate and per_query[*].metrics use string ks ("5", "10") for JSON object keys.
- Older runs may miss meta/aggregate; current evaluator should populate both.
- Location: artifacts/retrieval_eval/retrieval_eval.json