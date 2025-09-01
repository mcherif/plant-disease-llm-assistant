# PlantVillage KB

This file documents how data/plantvillage_kb.json was generated/refreshed.

Source
- PlantVillage topics “infos” pages (scoped per-disease sections).

Extractor
- src/ingestion/refresh_kb_descriptions.py
- Scoped extraction via `_extract_inline_for_disease`.

Reproduce
- Preview:
  python -m src.ingestion.refresh_kb_descriptions --in data\plantvillage_kb.json --out data\plantvillage_kb.updated.json --only-empty --max-sentences 2 --verbose
- Update in place:
  python -m src.ingestion.refresh_kb_descriptions --in data\plantvillage_kb.json --out data\plantvillage_kb.json --force --max-sentences 2 --allow-google --verbose

Notes
- JSON does not support comments; this README captures provenance.
- Verifier: python -m src.ingestion.verify_scoped_parsing --all --only-diseases --no-trunc