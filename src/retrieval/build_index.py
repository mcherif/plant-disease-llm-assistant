"""
Build a FAISS vector index from the KB manifest.

Usage (Windows):
  python -m src.retrieval.build_index --manifest data\kb\manifest.parquet --out models\index\kb-faiss --model sentence-transformers\all-MiniLM-L6-v2 --batch-size 64 --device cuda
"""
import argparse
import json
from pathlib import Path
from typing import List

import faiss  # type: ignore
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import trange


def read_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def normalize_embeddings(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def embed_corpus(
    texts: List[str],
    model_name: str,
    batch_size: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device)
    embs: List[np.ndarray] = []
    for i in trange(0, len(texts), batch_size, desc="Embed"):
        batch = texts[i : i + batch_size]
        embs.append(
            model.encode(
                batch,
                batch_size=len(batch),
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        )
    X = np.vstack(embs).astype("float32")
    return normalize_embeddings(X)


def build_faiss_index(X: np.ndarray) -> faiss.Index:
    d = int(X.shape[1])
    index = faiss.IndexFlatIP(d)  # cosine similarity via normalized vectors
    index.add(X)
    return index


def save_metadata(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = df.to_dict(orient="records")
    (out_dir / "meta.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )


def save_config(cfg: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build FAISS index from KB manifest")
    ap.add_argument("--manifest", default="data/kb/manifest.parquet")
    ap.add_argument("--out", default="models/index/kb-faiss")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--min-chars", type=int, default=20, help="Skip very short chunks")
    ap.add_argument("--doc-prefix", default="", help="Prefix for doc texts (e.g., 'passage: ' for E5/GTE/BGE)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out)

    df = read_manifest(manifest_path)
    # Keep core fields if present
    cols = [
        "doc_id",
        "title",
        "plant",
        "disease",
        "section",
        "lang",
        "split_idx",
        "url",
        "text",
    ]
    df = df[[c for c in cols if c in df.columns]].copy()
    df = df[df["text"].astype(str).str.len() >= args.min_chars].reset_index(drop=True)

    if df.empty:
        raise RuntimeError("No rows to index after filtering.")

    texts = df["text"].astype(str).tolist()
    if args.doc_prefix:
        texts = [f"{args.doc_prefix}{t}" for t in texts]
    X = embed_corpus(texts, model_name=args.model, batch_size=args.batch_size, device=args.device)
    index = build_faiss_index(X)

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "faiss.index"))
    save_metadata(df, out_dir)
    save_config(
        {
            "manifest": str(manifest_path),
            "out_dir": str(out_dir),
            "model": args.model,
            "num_vectors": int(index.ntotal),
            "dim": int(X.shape[1]),
            "normalize": True,
            "metric": "cosine (IP on normalized vectors)",
            "device": args.device,
            "min_chars": args.min_chars,
            "doc_prefix": args.doc_prefix,
        },
        out_dir,
    )
    print(f"[index] wrote {index.ntotal} vectors to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())