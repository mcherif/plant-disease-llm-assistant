# Data Card — Plant-Disease KB

Version: 0.1.0
Maintainer: <Mohamed Cherif/innerloopinc@gmail.com>
Last updated: YYYY-MM-DD

## Overview
Purpose: Knowledge base of plant diseases for retrieval-augmented assistants (PlantVillage + Wikipedia summaries).
Intended use: Research and prototyping. Not a substitute for professional agronomy advice.

## Sources and Licensing
- PlantVillage website (topic/disease pages). Terms: <link> (confirm and respect site terms).
- Wikipedia REST summaries (Creative Commons licenses). See https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use
Note: Verify rights before redistribution; this KB is for educational/research use.

## Collection and Reproducibility
Command (example):
`python -m src.ingestion.build_kb --sources plantvillage,wikipedia --out data/kb --min_tokens 50 --max_tokens 1000 --overlap 100 --dedup minhash --dedup-threshold 0.9 --wiki-lang en --wiki-interval 0.5`
Politeness: Custom User-Agent, robots.txt checks, retry/backoff, rate limiting.

## Processing Pipeline
- Normalization: light markdown formatting, minor text sanitizers (title-line strip, collapse repeated cause labels).
- Chunking: sentence-aware; max 1000 tokens; 100-token overlap.
- Deduplication: MinHash/LSH near-duplicate removal (threshold 0.9).

## Data Fields (manifest)
doc_id, url, title, plant, disease, section, lang, split_idx, text, n_tokens, crawl_date

## Quality and Validation
- URL validation script: src/ingestion/validate_kb_urls.py
- Spot-checks recommended: 10 random chunks for clean text and correct tags.
- Known issues: occasional short descriptions; Wikipedia fallback may not exist for all pairs.

## Limitations and Risks
- English-only; summaries can be shallow.
- Source noise possible; not medically/agronomically authoritative.
- Potential license constraints on redistribution.

## Ethical/Responsible Use
- Cite sources when using content.
- Respect robots.txt and rate limits.
- Do not misrepresent content as professional diagnosis.

## Maintenance & Versioning
- Version the KB by date and command options.
- Record changes in this card’s Changelog.

## Changelog
- 0.1.0: Initial PlantVillage + Wikipedia summaries; chunking + MinHash dedup.