"""Build a small, diverse RAG QA evaluation set as JSONL.

Prefers sampling plant+disease pairs from data/kb/manifest.parquet.
Falls back to a hand-seeded list if the manifest/pandas is unavailable.
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # noqa: N816

INTENT_TEMPLATES: Dict[str, List[str]] = {
    "symptoms": [
        "{plant} {disease} symptoms",
        "What are the symptoms of {disease} on {plant}?",
    ],
    "diagnosis": [
        "Leaves issue: could it be {disease} on {plant}?",
        "How to recognize {disease} on {plant}?",
    ],
    "treatment": [
        "How to manage {disease} in {plant}?",
        "Treatment for {plant} {disease}",
        "Best way to control {disease}?",
    ],
    "prevention": [
        "How to prevent {disease} in {plant}?",
        "Ways to reduce risk of {disease} in {plant}",
    ],
    "ipm": [
        "Integrated management for {disease} in {plant}",
    ],
    "sanitation": [
        "Sanitation steps for {disease} on {plant}",
    ],
    "chemicals": [
        "Is it safe to use copper for {disease} on {plant}?",
    ],
}

FALLBACK_SEEDS = [
    {"plant": "Tomato", "disease": "Late blight"},
    {"plant": "Tomato", "disease": "Tomato yellow leaf curl virus"},
    {"plant": "Potato", "disease": "Late blight"},
    {"plant": "Apple", "disease": "Apple scab"},
    {"plant": "Grape", "disease": "Downy mildew"},
    {"plant": "Zucchini", "disease": "Powdery mildew"},
    {"plant": "Pepper", "disease": "Bacterial spot"},
    {"plant": "Orange", "disease": "Citrus greening"},
    {"plant": "Banana", "disease": "Panama disease"},
    {"plant": "Strawberry", "disease": "Gray mold"},
    {"plant": "Cucumber", "disease": "Downy mildew"},
]


def build_from_manifest(manifest: Path, n_per_pair: int, cap: int, seed: int) -> List[Dict]:
    """
    Generates a list of unique question-answer pairs for plant-disease intents based on a manifest file.

    Args:
        manifest (Path): Path to the manifest file containing plant and disease data in Parquet format.
        n_per_pair (int): Number of questions to generate per plant-disease-intent combination.
        cap (int): Maximum number of unique question-answer pairs to return.
        seed (int): Random seed for reproducibility.

    Returns:
        List[Dict]: A list of dictionaries, each containing:
            - "question": The generated question string.
            - "plant": The plant name.
            - "disease": The disease name.
            - "intent": The intent type.

    Notes:
        - Only rows with non-null plant and disease values are considered.
        - Duplicate plant-disease pairs are removed before question generation.
        - Questions are generated using intent-specific templates.
        - The output is deduplicated and capped at the specified limit.
    """
    random.seed(seed)
    assert pd is not None
    df = pd.read_parquet(manifest)
    df = df.loc[df["plant"].notna() & df["disease"].notna(), ["plant", "disease"]].drop_duplicates()
    rows: List[Dict] = []
    for _, r in df.iterrows():
        plant = str(r["plant"]).strip()
        disease = str(r["disease"]).strip()
        for intent, templates in INTENT_TEMPLATES.items():
            for _ in range(n_per_pair):
                tpl = random.choice(templates)
                q = tpl.format(plant=plant, disease=disease)
                rows.append({"question": q, "plant": plant, "disease": disease, "intent": intent})
    # Deduplicate and cap
    uniq, seen = [], set()
    for row in rows:
        key = (row["question"].lower(), (row.get("plant") or "").lower())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(row)
        if len(uniq) >= cap:
            break
    return uniq


def build_fallback(n_per_pair: int, cap: int, seed: int) -> List[Dict]:
    """
    Generates a list of unique fallback question-answer pairs for plant-disease intents.

    Args:
        n_per_pair (int): Number of questions to generate per plant-disease-intent combination.
        cap (int): Maximum number of unique question rows to return.
        seed (int): Random seed for reproducibility.

    Returns:
        List[Dict]: A list of dictionaries, each containing a generated question, plant, disease, and intent.
    """
    random.seed(seed)
    pairs = FALLBACK_SEEDS[:]
    rows: List[Dict] = []
    for p in pairs:
        plant, disease = p["plant"], p["disease"]
        for intent, templates in INTENT_TEMPLATES.items():
            for _ in range(n_per_pair):
                tpl = random.choice(templates)
                q = tpl.format(plant=plant, disease=disease)
                rows.append({"question": q, "plant": plant, "disease": disease, "intent": intent})
    # Deduplicate and cap
    uniq, seen = [], set()
    for row in rows:
        key = (row["question"].lower(), (row.get("plant") or "").lower())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(row)
        if len(uniq) >= cap:
            break
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a RAG QA JSONL dataset.")
    ap.add_argument("--manifest", type=Path, default=Path("data/kb/manifest.parquet"))
    ap.add_argument("--out", type=Path, default=Path("data/eval/rag_qa.jsonl"))
    ap.add_argument("--n-per-pair", type=int, default=2)
    ap.add_argument("--cap", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict]
    if pd is not None and args.manifest.exists():
        rows = build_from_manifest(args.manifest, args.n_per_pair, args.cap, args.seed)
        src = f"manifest: {args.manifest}"
    else:
        rows = build_fallback(args.n_per_pair, min(args.cap, 80), args.seed)
        src = "fallback seeds"

    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} prompts to {args.out} (source: {src})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
