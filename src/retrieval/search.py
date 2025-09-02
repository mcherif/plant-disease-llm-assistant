import argparse
import json
import re
from pathlib import Path
from typing import List, Dict

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


def load_meta(index_dir: Path) -> List[Dict]:
    meta_path = index_dir / "meta.jsonl"
    rows = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_config(index_dir: Path) -> Dict:
    cfg_path = index_dir / "config.json"
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _tok(s: str) -> List[str]:
    return re.findall(r"\w+", str(s).lower())


def embed_query(text: str, model_name: str, device: str = "cpu") -> np.ndarray:
    model = SentenceTransformer(model_name, device=device)
    v = model.encode([text], convert_to_numpy=True,
                     normalize_embeddings=False).astype("float32")
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description="Search FAISS KB index")
    ap.add_argument("--index-dir", default="models/index/kb-faiss")
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--model", default=None,
                    help="Override model (defaults to index config)")
    ap.add_argument("--query-prefix", default="",
                    help="Prefix for queries (e.g., 'query: ')")
    ap.add_argument("--plant", default=None, help="Optional plant filter")
    ap.add_argument("--disease", default=None,
                    help="Optional disease substring filter")
    ap.add_argument("--pretopk", type=int, default=50,
                    help="Retrieve more then filter")
    ap.add_argument("--fusion", default="none",
                    choices=["none", "sum", "rrf"], help="Hybrid fusion method")
    ap.add_argument("--alpha", type=float, default=0.6,
                    help="Weight for vector score in 'sum' fusion")
    args = ap.parse_args()

    index_dir = Path(args.index_dir)
    index = faiss.read_index(str(index_dir / "faiss.index"))
    cfg = load_config(index_dir)
    meta = load_meta(index_dir)

    if index.ntotal != len(meta):
        raise RuntimeError(
            f"Index size {index.ntotal} != meta rows {len(meta)}")

    model_name = args.model or cfg.get(
        "model", "sentence-transformers/all-MiniLM-L6-v2")

    # Use query_prefix; auto-infer for BGE/E5/GTE if docs used 'passage: '
    query_prefix = args.query_prefix
    if not query_prefix:
        doc_prefix = str(cfg.get("doc_prefix", "")).strip().lower()
        if doc_prefix.startswith("passage:"):
            query_prefix = "query: "

    q_text = f"{query_prefix}{args.query}" if query_prefix else args.query
    q = embed_query(q_text, model_name=model_name, device=args.device)
    k = max(args.top_k, args.pretopk)
    vec_scores, vec_idxs = index.search(q, k)
    vec_scores, vec_idxs = vec_scores[0].tolist(), vec_idxs[0].tolist()

    # Prepare candidate set
    cand = list(zip(vec_scores, vec_idxs))

    # Optional BM25 fusion on candidates
    if args.fusion != "none" and cand:
        cand_idxs = [i for _, i in cand]
        cand_texts = [meta[i].get("text", "") for i in cand_idxs]
        bm25 = BM25Okapi([_tok(t) for t in cand_texts])
        bm25_scores = bm25.get_scores(_tok(args.query)).tolist()

        # Normalize to [0,1] per score type
        v = np.array([s for s, _ in cand], dtype=np.float32)
        b = np.array(bm25_scores, dtype=np.float32)
        def norm(x):
            mn, mx = float(x.min()), float(x.max())
            return (x - mn) / (mx - mn + 1e-12) if mx > mn else np.zeros_like(x)
        v_n = norm(v)
        b_n = norm(b)

        fused = []
        if args.fusion == "sum":
            f = args.alpha * v_n + (1.0 - args.alpha) * b_n
            for score, idx, fs in zip(v, cand_idxs, f.tolist()):
                fused.append((fs, score, idx))  # keep vec score for tiebreak
            fused.sort(key=lambda x: (x[0], x[1]), reverse=True)
            cand = [(score, idx) for _, score, idx in fused]
        elif args.fusion == "rrf":
            # Reciprocal Rank Fusion with k=60
            k_rrf = 60.0
            # ranks from vector
            vec_order = {i: r for r, (_, i) in enumerate(sorted(cand, key=lambda x: x[0], reverse=True), 1)}
            # ranks from bm25
            bm25_order = {i: r for r, i in enumerate([cand_idxs[j] for j in np.argsort(b)[::-1]], 1)}
            fused_rrf = []
            for idx in cand_idxs:
                s = 1.0 / (k_rrf + vec_order[idx]) + 1.0 / (k_rrf + bm25_order[idx])
                fused_rrf.append((s, idx))
            fused_rrf.sort(key=lambda x: x[0], reverse=True)
            # map back to original vec scores for printing
            vec_map = {i: s for s, i in cand}
            cand = [(vec_map[i], i) for _, i in fused_rrf]

    # Apply filters and keep top-k
    results = []
    for s, i in cand:
        m = meta[i]
        if args.plant and str(m.get("plant", "")).lower() != args.plant.lower():
            continue
        if args.disease and args.disease.lower() not in str(m.get("disease", "")).lower():
            continue
        results.append((s, i))
        if len(results) >= args.top_k:
            break

    for rank, (s, i) in enumerate(results, 1):
        m = meta[i]
        print(f"{rank:2d}. score={s: .3f}  title={m.get('title','')}  plant={m.get('plant','')}  disease={m.get('disease','')}")
        print(f"    url={m.get('url','')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
