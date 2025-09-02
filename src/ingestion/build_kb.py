r"""
Unified KB builder for PlantVillage (+ Wikipedia soon).

- Reads PlantVillage KB JSON (from refresh_kb_descriptions.py)
- Normalizes sections to markdown, sanitizes minor noise
- Chunks sentence-aware with overlap
- Optional MinHash/LSH dedup
- Writes chunks/ and a manifest (Parquet or CSV fallback)

Usage:
  python -m src.ingestion.build_kb --sources plantvillage --out data\\kb --min_tokens 50 --max_tokens 1000 --overlap 100 --dedup minhash --dedup-threshold 0.9 --verbose
"""
import argparse
import datetime as dt
import json
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict, Union


class Doc(TypedDict, total=False):
    doc_id: str
    url: str
    title: str
    plant: Optional[str]
    disease: Optional[str]
    section: Optional[str]
    lang: str
    crawl_date: str  # ISO date
    text: str


def parse_args() -> argparse.Namespace:
    """CLI for selecting sources, output directory, chunking, and dedup options."""
    ap = argparse.ArgumentParser(
        description="Build a normalized, chunked KB from PlantVillage (+Wikipedia soon)."
    )
    ap.add_argument(
        "--sources",
        default="plantvillage",
        help="Comma-separated sources: plantvillage,wikipedia",
    )
    ap.add_argument(
        "--out",
        default=str(Path("data") / "kb"),
        help="Output directory for chunks/ and manifest",
    )
    ap.add_argument(
        "--min_tokens", type=int, default=50, help="Minimum tokens per chunk (approx.)"
    )
    ap.add_argument(
        "--max_tokens", type=int, default=1000, help="Maximum tokens per chunk (approx.)"
    )
    ap.add_argument(
        "--overlap", type=int, default=100, help="Token overlap between chunks (approx.)"
    )
    ap.add_argument(
        "--kb", default=str(Path("data") / "plantvillage_kb.json"),
        help="Path to PlantVillage KB JSON (produced by refresh_kb_descriptions.py)"
    )
    ap.add_argument(
        "--dedup", default="none", choices=["none", "minhash"],
        help="Deduplicate chunks (minhash uses LSH; requires 'datasketch')"
    )
    ap.add_argument(
        "--dedup-threshold", type=float, default=0.9,
        help="MinHash Jaccard threshold for near-duplicate chunks (0.8–0.95 typical)"
    )
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def now_date() -> str:
    return dt.date.today().isoformat()


def word_tokens(text: str) -> List[str]:
    return re.findall(r"\w+|\S", text, flags=re.UNICODE)


def count_tokens(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def sentence_split(text: str) -> List[str]:
    """Very simple sentence splitter; replace with spaCy/NLTK for better accuracy."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(0-9])", text.strip())
    return [p.strip() for p in parts if p and not p.isspace()]


def normalize_to_markdown(text: str) -> str:
    """Light normalization placeholder; collapse stray whitespace/newlines."""
    return re.sub(r"\s+\n", "\n", text.strip())


def _collapse_repeated_word_labels(text: str) -> str:
    """
    Collapse repeated single-word labels (e.g., 'Bacterium Fungus ...') while preserving order.
    Only applies to lines without punctuation to avoid mangling prose.
    """
    lines = text.splitlines()
    out_lines: List[str] = []
    for line in lines:
        # Skip lines that look like sentences (contain punctuation)
        if re.search(r"[.,;:()\-–—/]", line):
            out_lines.append(line)
            continue
        words = re.findall(r"[A-Za-z]+", line)
        if not words:
            out_lines.append(line)
            continue
        # Apply only if there are duplicates and mostly single-word labels
        if len(set(w.lower() for w in words)) < len(words):
            seen = set()
            uniq = []
            for w in words:
                wl = w.lower()
                if wl in seen:
                    continue
                seen.add(wl)
                uniq.append(w)
            out_lines.append(", ".join(uniq))
        else:
            out_lines.append(line)
    return "\n".join(out_lines).strip()


def _sanitize_cause(text: str) -> str:
    """Sanitize 'Cause' sections (collapse repeated labels)."""
    return _collapse_repeated_word_labels(text)


def _strip_leading_title_repeat(text: str, disease: str) -> str:
    """Remove a leading line that repeats the disease title (e.g., 'Late Blight')."""
    if not text or not disease:
        return text
    lines = text.strip().splitlines()
    if not lines:
        return text
    first = lines[0].strip().strip("—:- ").casefold()
    dz = disease.strip().strip("—:- ").casefold()
    if first.startswith(dz):
        return "\n".join(lines[1:]).lstrip()
    return text


def chunk_text_sentence_aware(
    text: str, max_tokens: int, overlap_tokens: int
) -> List[str]:
    """Pack sentences into chunks up to max_tokens, carrying overlap from the end of the previous chunk."""
    if not text:
        return []
    sents = sentence_split(text)
    chunks: List[str] = []
    cur: List[str] = []
    cur_tok = 0
    for s in sents:
        st = count_tokens(s)
        if cur and cur_tok + st > max_tokens:
            chunks.append(" ".join(cur).strip())
            # Build overlap tail
            if overlap_tokens > 0:
                tail = []
                tok_sum = 0
                for ss in reversed(cur):
                    tok_sum += count_tokens(ss)
                    tail.append(ss)
                    if tok_sum >= overlap_tokens:
                        break
                cur = list(reversed(tail))
                cur_tok = sum(count_tokens(ss) for ss in cur)
            else:
                cur = []
                cur_tok = 0
        cur.append(s)
        cur_tok += st
    if cur:
        chunks.append(" ".join(cur).strip())
    return [c for c in chunks if count_tokens(c) >= 1]


def ensure_out_dirs(out_dir: Union[str, Path]) -> Tuple[Path, Path]:
    """Create the output directory and chunks/ subfolder."""
    out_dir = Path(out_dir)
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, chunks_dir


def write_manifest(rows: List[Dict], out_dir: Path, verbose: bool = False) -> Path:
    """Write a manifest to Parquet if possible; fall back to CSV. Returns the manifest path."""
    out_parquet = out_dir / "manifest.parquet"
    out_csv = out_dir / "manifest.csv"
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(rows)
        # Ensure column order
        cols = [
            "doc_id",
            "url",
            "title",
            "plant",
            "disease",
            "section",
            "lang",
            "crawl_date",
            "split_idx",
            "text",
            "n_tokens",
        ]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        df = df[cols]
        df.to_parquet(out_parquet, index=False)
        if verbose:
            print(f"[manifest] wrote {out_parquet}")
        return out_parquet
    except Exception as e:
        if verbose:
            print(
                f"[manifest] parquet unavailable ({e}); writing CSV fallback")
        try:
            import csv

            with out_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "doc_id",
                        "url",
                        "title",
                        "plant",
                        "disease",
                        "section",
                        "lang",
                        "crawl_date",
                        "split_idx",
                        "text",
                        "n_tokens",
                    ],
                )
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
            if verbose:
                print(f"[manifest] wrote {out_csv}")
            return out_csv
        except Exception as e2:
            raise RuntimeError(f"Failed to write manifest: {e2}") from e2


def write_chunk_file(chunks_dir: Path, doc_id: str, split_idx: int, text: str) -> Path:
    """Write a single chunk to chunks/{doc_id}_{split_idx}.md and return its path."""
    safe_id = re.sub(r"[^A-Za-z0-9_\-]", "_", doc_id)
    path = chunks_dir / f"{safe_id}_{split_idx:04d}.md"
    with path.open("w", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")
    return path


def deduplicate_chunks(rows: List[Dict], method: str, threshold: float = 0.9, verbose: bool = False) -> List[Dict]:
    """Near-duplicate removal using MinHash/LSH. Keeps the first chunk in each near-dup group."""
    if method == "none":
        return rows
    try:
        from datasketch import MinHash, MinHashLSH  # type: ignore
    except Exception as e:
        if verbose:
            print(f"[dedup] datasketch unavailable ({e}); skipping dedup")
        return rows

    def shingles(text: str, n: int = 5) -> List[str]:
        toks = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        if len(toks) <= n:
            return [" ".join(toks)] if toks else []
        return [" ".join(toks[i:i+n]) for i in range(0, len(toks) - n + 1)]

    num_perm = 64
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: List[Dict] = []
    mh_index: Dict[str, MinHash] = {}

    def row_key(r: Dict) -> str:
        return f"{r.get('doc_id','')}_{int(r.get('split_idx', 0))}"

    dup_count = 0
    for r in rows:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        mh = MinHash(num_perm=num_perm)
        for g in shingles(text):
            mh.update(g.encode("utf-8"))
        # Query LSH for near-dups
        candidates = lsh.query(mh)
        is_dup = False
        for cid in candidates:
            # Final check using MinHash estimate to be safe
            if mh.jaccard(mh_index[cid]) >= threshold:
                is_dup = True
                break
        if is_dup:
            dup_count += 1
            continue
        kid = row_key(r)
        lsh.insert(kid, mh)
        mh_index[kid] = mh
        kept.append(r)

    if verbose:
        print(
            f"[dedup] kept {len(kept)} / {len(rows)} chunks (removed {dup_count})")
    return kept


def load_docs_from_plantvillage(kb_path: Union[str, Path], verbose: bool = False) -> List[Doc]:
    """Load docs from PlantVillage KB JSON and synthesize markdown sections."""
    kb_path = Path(kb_path)
    if not kb_path.exists():
        raise FileNotFoundError(
            f"PlantVillage KB not found: {kb_path}. Run refresh_kb_descriptions.py first.")
    with kb_path.open("r", encoding="utf-8") as f:
        kb = json.load(f)

    docs: List[Doc] = []
    crawl_date = now_date()
    for plant, diseases in kb.items():
        for disease, entry in diseases.items():
            if disease.lower() == "healthy":
                continue
            desc = (entry or {}).get("description") or ""
            symptoms = (entry or {}).get("symptoms") or ""
            cause = (entry or {}).get("cause") or ""
            url = (entry or {}).get("source") or ""
            title = f"{plant} — {disease}"
            # Sanitize known noisy sections
            if cause:
                cause = _sanitize_cause(cause)
            if desc:
                desc = _strip_leading_title_repeat(desc, disease)
            parts = []
            if desc:
                parts.append(f"# {title}\n\n{desc}")
            if symptoms:
                parts.append(f"## Symptoms\n\n{symptoms}")
            if cause:
                parts.append(f"## Cause\n\n{cause}")
            full_text = normalize_to_markdown("\n\n".join(parts).strip())
            if not full_text:
                continue
            doc: Doc = {
                "doc_id": str(uuid.uuid4()),
                "url": url,
                "title": title,
                "plant": plant,
                "disease": disease,
                "section": None,
                "lang": "en",  # TODO: optional lang detection
                "crawl_date": crawl_date,
                "text": full_text,
            }
            docs.append(doc)
    if verbose:
        print(f"[pv] loaded {len(docs)} docs from {kb_path}")
    return docs


def load_docs_from_wikipedia(verbose: bool = False) -> List[Doc]:
    """Stub for Wikipedia ingestion; returns an empty list for now."""
    # TODO: Implement Wikipedia ingestion (fetch or reuse cached HTML, then normalize text)
    if verbose:
        print("[wiki] ingestion is stubbed (returns 0 docs)")
    return []


def build_kb(
    sources: List[str],
    out_dir: Union[str, Path],
    kb_path: Union[str, Path],
    min_tokens: int,
    max_tokens: int,
    overlap: int,
    dedup_method: str,
    dedup_threshold: float = 0.9,
    verbose: bool = False,
) -> Path:
    """Main pipeline: load docs, chunk, optionally dedup, and write manifest."""
    out_dir, chunks_dir = ensure_out_dirs(out_dir)

    # Load documents
    docs: List[Doc] = []
    srcs = [s.strip().lower() for s in sources if s.strip()]
    if "plantvillage" in srcs:
        docs.extend(load_docs_from_plantvillage(kb_path, verbose=verbose))
    if "wikipedia" in srcs:
        docs.extend(load_docs_from_wikipedia(verbose=verbose))

    if not docs:
        raise RuntimeError("No documents loaded. Check --sources and inputs.")

    # Chunk
    rows: List[Dict] = []
    for d in docs:
        text = d["text"]
        chunks = chunk_text_sentence_aware(
            text, max_tokens=max_tokens, overlap_tokens=overlap)
        # Ensure min_tokens
        chunks = [c for c in chunks if count_tokens(c) >= min_tokens]
        for i, ch in enumerate(chunks):
            n_tokens = count_tokens(ch)
            row = {
                "doc_id": d["doc_id"],
                "url": d["url"],
                "title": d["title"],
                "plant": d.get("plant"),
                "disease": d.get("disease"),
                "section": d.get("section"),
                "lang": d.get("lang", "en"),
                "crawl_date": d.get("crawl_date", now_date()),
                "split_idx": i,
                "text": ch,
                "n_tokens": n_tokens,
            }
            rows.append(row)
            write_chunk_file(chunks_dir, d["doc_id"], i, ch)
    if verbose:
        print(f"[chunk] wrote {len(rows)} chunks to {chunks_dir}")

    # Deduplicate (stub)
    rows = deduplicate_chunks(
        rows, method=dedup_method, threshold=dedup_threshold, verbose=verbose)

    # Manifest
    manifest_path = write_manifest(rows, out_dir, verbose=verbose)
    return manifest_path


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    sources = args.sources.split(",") if args.sources else []
    try:
        manifest = build_kb(
            sources=sources,
            out_dir=args.out,
            kb_path=args.kb,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            overlap=args.overlap,
            dedup_method=args.dedup,
            dedup_threshold=args.dedup_threshold,
            verbose=args.verbose,
        )
        if args.verbose:
            print(f"[done] manifest at {manifest}")
        return 0
    except Exception as e:
        print(f"[error] {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
