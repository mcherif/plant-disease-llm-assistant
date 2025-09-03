"""
RAG pipeline: retrieve → compose prompt with citations → call LLM.

Overview
- Retrieval: FAISS dense vector search over KB chunk embeddings, with optional BM25
  late-fusion (weighted sum or Reciprocal Rank Fusion) to blend lexical + semantic signals.
- Prompting: Compose a concise prompt containing the user's question and a numbered
  context block. The LLM is instructed to cite sources inline as [n] matching the
  provided context entries.
- Generation: Call an LLM backend (OpenAI by default) and return an answer with
  citations plus the source metadata.

Index layout (produced by src/retrieval/build_index.py under models/index/)
- <index_dir>/faiss.index        — FAISS index with vectors
- <index_dir>/meta.jsonl         — one JSON per vector (aligned order) with fields like:
                                   { "doc_id", "title", "plant", "disease", "url", "text", ... }
- <index_dir>/config.json        — includes model name, dim, and optional "doc_prefix"

Model prefixes
- Some embedding families (BGE, E5, GTE) benefit from special prefixes:
  - Documents indexed with "passage: " (doc_prefix)
  - Queries encoded with "query: " (inferred here automatically)
- If no doc_prefix is present, queries are used as-is.

Fusion methods
- none: pure vector similarity from FAISS.
- sum: per-query min–max normalization then fused = alpha * vec + (1 - alpha) * bm25.
- rrf: Reciprocal Rank Fusion combining ranks from vector and BM25 lists.

Defaults and how to change
- Default fusion is "sum" with alpha=0.7 (see RetrievalConfig defaults).
- Programmatic control:
  - Global: cfg = RetrievalConfig(..., fusion="rrf", alpha=0.6); rag = RAGPipeline(cfg)
  - Per-call: rag.answer("...", fusion="none"); rag.answer("...", fusion="sum", alpha=0.5)
- Evaluator CLI examples:
  - python -m src.retrieval.evaluate --fusion sum --alpha 0.7
  - python -m src.retrieval.evaluate --fusion rrf
  - python -m src.retrieval.evaluate --fusion none

Environment
- OPENAI_API_KEY must be set to use the default OpenAI backend; OPENAI_MODEL overrides
  model name (default: gpt-4o-mini).

Usage example
    from pathlib import Path
    from src.llm.rag_pipeline import RAGPipeline, RetrievalConfig

    cfg = RetrievalConfig(
        index_dir=Path("models/index/kb-faiss-bge"),
        device="cuda",
        fusion="sum",
        alpha=0.7,
        top_k=4,
    )
    rag = RAGPipeline(cfg)
    res = rag.answer("tomato yellow leaf curl symptoms", plant="Tomato")
    print(res["answer"])
    print("Sources:", [(s["id"], s["title"], s["url"]) for s in res["sources"]])

Returns schema (answer())
{
  "answer": str,                         # model output with inline [n] citations
  "sources": [ { "id": int, "title": str, "url": str }, ... ],   # numbered to match citations
  "retrieved": [ { "score": float, "meta": { ... } }, ... ]      # raw top-k retrieval results
}

Notes
- Performance: The encoder is instantiated once and cached in the pipeline instance.
- Portability: _load_prompt expects src/llm/prompts/answer.txt. Keep that template synced.
- Extensibility: Swap _generate() with a different backend, add rerankers, or
  implement safety filters before returning the final answer.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss  # type: ignore
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def _tok(s: str) -> List[str]:
    """Lightweight tokenizer for BM25: lowercase alphanumeric tokens."""
    return re.findall(r"\w+", str(s).lower())


def _minmax(x: np.ndarray) -> np.ndarray:
    """Min–max normalize a 1D score array to [0, 1]; returns zeros if degenerate."""
    mn, mx = float(x.min()), float(x.max())
    return (x - mn) / (mx - mn + 1e-12) if mx > mn else np.zeros_like(x)


@dataclass
class RetrievalConfig:
    """Configuration for the retrieval phase.

    Args:
        index_dir: Directory containing faiss.index, meta.jsonl, and config.json.
        device: "cpu" or "cuda" for query encoding.
        model_name: Optional override for the embedding model (defaults to index config).
        fusion: "none" | "sum" | "rrf" — how to combine vector and BM25 signals.
        alpha: Weight for vector score in "sum" fusion (0..1). Higher favors embeddings.
        top_k: Number of chunks to return to the LLM context.
    """
    index_dir: Path
    device: str = "cpu"
    model_name: Optional[str] = None
    fusion: str = "sum"  # none | sum | rrf
    alpha: float = 0.7
    top_k: int = 4


class RAGPipeline:
    """RAG pipeline orchestrating retrieval, prompt composition, and generation.

    Responsibilities:
    - Load FAISS index + aligned metadata and index config.
    - Encode queries (adding "query: " if documents were indexed with "passage: ").
    - Retrieve candidates via FAISS and optionally fuse with BM25 over candidates.
    - Compose a prompt with numbered context blocks to enable inline citations.
    - Call an LLM backend to produce the final answer.

    Raises:
        RuntimeError: if the index files are inconsistent or the LLM backend fails.
    """

    def __init__(self, cfg: RetrievalConfig):
        """Initialize the pipeline and cache encoder and index handles."""
        self.cfg = cfg
        self.index_dir = Path(cfg.index_dir)

        # Load FAISS and metadata/config
        self.index = faiss.read_index(str(self.index_dir / "faiss.index"))
        self.meta = self._load_meta(self.index_dir)
        self.config = self._load_config(self.index_dir)

        # Determine embedding model and query prefix behavior
        self.model_name = cfg.model_name or self.config.get(
            "model", "sentence-transformers/all-MiniLM-L6-v2")
        self.doc_prefix = str(self.config.get(
            "doc_prefix", "")).strip().lower()
        self.query_prefix = "query: " if self.doc_prefix.startswith(
            "passage:") else ""

        # Cache encoder model to avoid reloading each query
        self.encoder = SentenceTransformer(self.model_name, device=cfg.device)

        # Sanity check: vector count must match metadata rows
        if self.index.ntotal != len(self.meta):
            raise RuntimeError(
                f"Index size {self.index.ntotal} != meta rows {len(self.meta)}")

    @staticmethod
    def _load_meta(index_dir: Path) -> List[Dict]:
        """Load metadata rows aligned with vectors (one JSON per line)."""
        rows = []
        with (index_dir / "meta.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    @staticmethod
    def _load_config(index_dir: Path) -> Dict:
        """Load index configuration (e.g., model name, dim, doc_prefix)."""
        return json.loads((index_dir / "config.json").read_text(encoding="utf-8"))

    def _embed_query(self, text: str) -> np.ndarray:
        """Encode a single query; returns L2-normalized float32 vector of shape (1, dim)."""
        v = self.encoder.encode(
            [text], convert_to_numpy=True, normalize_embeddings=False).astype("float32")
        v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        return v

    def _retrieve(
        self,
        query: str,
        plant: Optional[str] = None,
        top_k: Optional[int] = None,
        fusion: Optional[str] = None,
        alpha: Optional[float] = None,
    ) -> List[Tuple[float, int]]:
        """Retrieve top candidates from FAISS and apply optional BM25 fusion + filters.

        Args:
            query: User question (prefix "query: " added automatically if needed).
            plant: Optional plant name to filter results by exact match on metadata.
            top_k: Override for number of results to return (defaults to cfg.top_k).
            fusion: "none" | "sum" | "rrf" overrides cfg.fusion.
            alpha: Weight for vector score in sum fusion; overrides cfg.alpha.

        Returns:
            List of (vector_score, meta_index) sorted by fused rank, truncated to top_k.
            The score is the original vector score (pre-fusion), useful for diagnostics.

        Notes:
            - BM25 is computed only on the candidate pool, not the full corpus, for speed.
            - pretopk is set to max(50, top_k) to ensure enough candidates for fusion.
            - Plant filter is applied after fusion/ordering.
        """
        k = top_k or self.cfg.top_k
        fusion = fusion or self.cfg.fusion
        alpha = alpha if alpha is not None else self.cfg.alpha

        # Encode query (add "query: " if documents used "passage: ")
        q_text = f"{self.query_prefix}{query}" if self.query_prefix else query
        q = self._embed_query(q_text)

        # Initial candidate set from FAISS
        pretopk = max(50, k)
        vec_scores, vec_idxs = self.index.search(q, pretopk)
        cand = list(zip(vec_scores[0].tolist(), vec_idxs[0].tolist()))

        # Optional hybrid fusion with BM25 over candidate texts
        if fusion != "none" and cand:
            cand_idxs = [i for _, i in cand]
            cand_texts = [self.meta[i].get("text", "") for i in cand_idxs]
            bm25 = BM25Okapi([_tok(t) for t in cand_texts])
            bm_scores = np.array(bm25.get_scores(
                _tok(query)), dtype=np.float32)

            v = np.array([s for s, _ in cand], dtype=np.float32)
            v_n = _minmax(v)
            b_n = _minmax(bm_scores)

            if fusion == "sum":
                # Weighted sum on normalized scores, then re-order candidates
                fused = alpha * v_n + (1.0 - alpha) * b_n
                order = np.argsort(fused)[::-1].tolist()
                cand = [(float(v[j]), cand_idxs[j]) for j in order]
            else:
                # Reciprocal Rank Fusion (RRF), robust to score-scale differences
                k_rrf = 60.0
                vec_rank = {i: r for r, (_, i) in enumerate(
                    sorted(cand, key=lambda x: x[0], reverse=True), 1)}
                bm_order = np.argsort(bm_scores)[::-1].tolist()
                bm_rank = {cand_idxs[j]: r for r, j in enumerate(bm_order, 1)}
                scored = []
                for i in cand_idxs:
                    s = 1.0 / (k_rrf + vec_rank[i]) + \
                        1.0 / (k_rrf + bm_rank[i])
                    scored.append((s, i))
                scored.sort(key=lambda x: x[0], reverse=True)
                # keep original vector scores for output
                vmap = {i: s for s, i in cand}
                cand = [(float(vmap[i]), i) for _, i in scored]

        # Apply metadata filters and truncate to top_k
        results: List[Tuple[float, int]] = []
        for s, i in cand:
            m = self.meta[i]
            if plant and str(m.get("plant", "")).lower() != str(plant).lower():
                continue
            results.append((s, i))
            if len(results) >= k:
                break
        return results

    def _load_prompt(self) -> str:
        """Load the answer prompt template.

        Expects an answer.txt template at src/llm/prompts/answer.txt within the repo.
        The template must include placeholders:
            {context} — concatenated, numbered source blocks
            {question} — the user question
        """
        p = self.index_dir.parents[2] / "src" / \
            "llm" / "prompts" / "answer.txt"
        return p.read_text(encoding="utf-8")

    def _compose(self, query: str, hits: List[Tuple[float, int]]) -> Tuple[str, List[Dict]]:
        """Compose the final prompt and assemble a compact sources list.

        Args:
            query: User question.
            hits: List of (score, meta_index) from retrieval.

        Returns:
            (prompt_text, sources):
              prompt_text: string with numbered context blocks and the question.
              sources: [{ "id": n, "title": str, "url": str }, ...] matching [n] citations.

        Note:
            - Context blocks are numbered in the same order as provided in hits.
            - Keep context compact to reduce token cost and improve focus.
        """
        sources = []
        context_blocks = []
        for rank, (_, idx) in enumerate(hits, 1):
            m = self.meta[idx]
            title = m.get("title", "") or m.get("disease", "") or "Source"
            url = m.get("url", "")
            text = m.get("text", "")
            sources.append({"id": rank, "title": title, "url": url})
            context_blocks.append(
                f"[{rank}] {title}\nURL: {url}\n---\n{text}\n")

        prompt_tpl = self._load_prompt()
        prompt = prompt_tpl.format(
            question=query, context="\n\n".join(context_blocks))
        return prompt, sources

    def _generate(self, prompt: str, model: Optional[str] = None) -> str:
        """Call the LLM backend (OpenAI) and return the answer text.

        Args:
            prompt: Fully composed prompt including context and question.
            model: Optional override for the OpenAI model (OPENAI_MODEL default applies).

        Returns:
            LLM response text (str). May include inline [n] citations.

        Raises:
            RuntimeError: if OPENAI_API_KEY is missing or the API call fails.

        Tip:
            - Set environment variables:
                OPENAI_API_KEY=...    required
                OPENAI_MODEL=gpt-4o-mini (default) or another compatible model
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Set it to use the OpenAI backend.")
        try:
            # OpenAI Python SDK v1.x
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a concise assistant. Answer with citations like [1], [2]. "
                            "If context is insufficient, say you don't know."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            # Provide actionable error context upstream
            raise RuntimeError(f"LLM call failed: {e}") from e

    def answer(
        self,
        query: str,
        plant: Optional[str] = None,
        top_k: Optional[int] = None,
        fusion: Optional[str] = None,
        alpha: Optional[float] = None,
        model: Optional[str] = None,
    ) -> Dict:
        """Run the full RAG flow and return answer plus sources and raw retrieval.

        Args:
            query: User question to be answered.
            plant: Optional plant filter for retrieval.
            top_k: Override for number of chunks used as context.
            fusion: Fusion strategy override ("none" | "sum" | "rrf").
            alpha: Weight override for "sum" fusion.
            model: LLM model override for the generation step.

        Returns:
            Dict with:
              - "answer": model output text (with inline [n] citations)
              - "sources": list of numbered source dicts (id, title, url)
              - "retrieved": list of {score, meta} for the returned top_k chunks

        Contract:
            - sources[n-1]["id"] equals citation index [n] used in the answer.
            - retrieved[i]["meta"] is an element from meta.jsonl (unchanged).
        """
        hits = self._retrieve(query, plant=plant,
                              top_k=top_k, fusion=fusion, alpha=alpha)
        prompt, sources = self._compose(query, hits)
        output = self._generate(prompt, model=model)
        return {
            "answer": output,
            "sources": sources,
            "retrieved": [{"score": float(s), "meta": self.meta[i]} for s, i in hits],
        }
