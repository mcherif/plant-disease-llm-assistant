"""
Build a tiny sample KB (2 plants, 3 diseases) for quick demo / Docker smoke.

Creates:
  data/sample_kb/meta.jsonl   (chunk metadata with text lines)
Then calls:
  python -m src.retrieval.build_index --manifest data/sample_kb/meta.jsonl --out models/index/sample-faiss-bge

Run:
  python -m src.ingestion.build_sample_kb
"""
from pathlib import Path
import json
import shutil
import subprocess
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "sample_kb"
INDEX_DIR = ROOT / "models" / "index" / "sample-faiss-bge"
META_PATH = DATA_DIR / "meta.jsonl"
PARQUET_PATH = DATA_DIR / "manifest.parquet"

CHUNKS = [
    {
        "doc_id": "tomato_bacterial_spot_0",
        "title": "Tomato Bacterial Spot",
        "plant": "Tomato",
        "disease": "Bacterial spot",
        "url": "https://example.org/tomato/bacterial_spot",
        "text": "Bacterial spot causes dark leaf and fruit lesions. Remove debris; apply copper-based sprays per label.",
        "section": "overview",
        "lang": "en",
    },
    {
        "doc_id": "tomato_early_blight_0",
        "title": "Tomato Early Blight",
        "plant": "Tomato",
        "disease": "Early blight",
        "url": "https://example.org/tomato/early_blight",
        "text": "Early blight shows concentric rings on older leaves and can defoliate plants. Rotate crops and use protectant fungicides.",
        "section": "overview",
        "lang": "en",
    },
    {
        "doc_id": "peach_powdery_mildew_0",
        "title": "Peach Powdery Mildew",
        "plant": "Peach",
        "disease": "Powdery mildew",
        "url": "https://example.org/peach/powdery_mildew",
        "text": "Powdery mildew: white fungal growth on young tissue. Improve airflow; sulfur or other fungicides may help.",
        "section": "overview",
        "lang": "en",
    },
]

def clean():
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

def write_meta():
    # Write JSONL
    with META_PATH.open("w", encoding="utf-8") as f:
        for row in CHUNKS:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    # Write Parquet (preferred by build_index)
    df = pd.DataFrame(CHUNKS)
    # Ensure required column exactly named 'text'
    assert "text" in df.columns, "Missing 'text' column"
    df.to_parquet(PARQUET_PATH, index=False)
    print(f"Wrote {META_PATH} and {PARQUET_PATH} ({len(CHUNKS)} chunks)")

def build_index():
    cmd = [
        "python",
        "-m",
        "src.retrieval.build_index",
        "--manifest",
        str(PARQUET_PATH),
        "--out",
        str(INDEX_DIR),
        "--min-chars",
        "1",  # allow tiny demo text
        "--device",
        "cpu",
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    clean()
    write_meta()
    build_index()
    print("Sample KB + index ready at", INDEX_DIR)

if __name__ == "__main__":
    main()
