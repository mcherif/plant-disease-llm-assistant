# Data Card — Plant-Disease KB

Version: 0.1.0
Maintainer: <Mohamed Cherif/innerloopinc@gmail.com>
Last updated: 2025-09-02

## Overview
Purpose: Knowledge base of plant diseases for retrieval-augmented assistants (PlantVillage + Wikipedia summaries).
Intended use: Research and prototyping. Not a substitute for professional agronomy advice.

## Sources and Licensing
- PlantVillage (topic/disease pages)
  - Source: https://www.plantvillage.psu.edu/ (check exact page origins in your seeds)
  - Licensing/terms: review and comply with the site’s Terms of Use before redistribution. Prefer storing derived summaries and link back to the source.
- Wikipedia REST/Action API summaries
  - Text license: Creative Commons Attribution-ShareAlike 4.0 (CC BY‑SA 4.0)
    - https://creativecommons.org/licenses/by-sa/4.0/
    - https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use
  - Requirements: provide attribution and links to the original pages; share-alike applies to redistributed text.

Note: This project aggregates information for educational use. Verify rights before public redistribution of the built KB.

## Collection and Reproducibility
Example command:
`python -m src.ingestion.build_kb --sources plantvillage,wikipedia --out data\kb --min_tokens 50 --max_tokens 1000 --overlap 100 --dedup minhash --dedup-threshold 0.9 --wiki-lang en --wiki-interval 0.5 --verbose`

Politeness: custom User-Agent, Action API usage, retry/backoff, robots.txt checks for REST, per-host rate limiting.

## Processing Pipeline
- Normalization: light markdown formatting and minor sanitization
- Chunking: sentence-aware splits; configurable max tokens (e.g., 512–1,000); overlap
- Deduplication: MinHash/LSH near-duplicate removal (threshold ~0.9)

## Data Fields (manifest)
doc_id, url, title, plant, disease, section, lang, split_idx, text, n_tokens, crawl_date

## Structured KB Schema

Each knowledge base entry (row/chunk) should include the following required fields:

- **doc_id**: Unique identifier for the document or chunk (UUID)
- **plant**: Crop name (e.g., "Maize", "Peach", "Tomato")
- **disease**: Disease name (e.g., "Common Rust", "Powdery Mildew")
- **title**: Title or summary of the entry
- **section**: Section/category (e.g., "Symptoms", "Management", "Prevention"); indicates the category or part of the document the chunk represents; can be null if not split
- **symptoms**: Description of disease symptoms (text)
- **cause**: Description of the cause (pathogen, environmental, etc.)
- **management**: Actionable advice for managing/treating the disease (text)
- **prevention**: Preventive measures (text)
- **references**: List of source URLs or citations
- **lang**: Language code (e.g., "en")
- **text**: Full text (may combine above fields if not split); this is a convenience field for retrieval
- **n_tokens**: Number of tokens in the chunk; helps manage chunk size for efficient retrieval and LLM usage
- **split_idx**: Indicates the position of the chunk in the original document when splitting for KB ingestion
- **crawl_date**: Date the entry was collected

**Notes:**
- All entries must have non-empty `symptoms`, `cause`, and `management` fields.
- `prevention` and `references` are recommended but may be empty if not available.
- Use this schema for all future scraping, ingestion, and validation.


## Quality and Validation
- URL validation: src/ingestion/validate_kb_urls.py
- Spot-check recommended: 10 random chunks for clean text and correct tags
- Dedup sanity: ensure near-duplicates are removed while preserving coverage

## Limitations and Risks
- English-only; Wikipedia summaries can be shallow
- Source noise possible; not authoritative
- Potential license constraints on redistribution

## Ethical/Responsible Use
- Cite sources; include URLs/titles in outputs
- Respect robots.txt and rate limits
- Do not present content as professional diagnosis

## Maintenance & Versioning
- Version the KB by date and build options
- Track changes (sources, thresholds, chunk sizes) in this card

## Changelog
- 0.1.0: Initial PlantVillage + Wikipedia summaries; sentence-aware chunking; MinHash/LSH dedup; polite fetching.
