"""
KB Schema Validator

Checks each KB entry for completeness and correctness according to the schema in docs/data_card.md.
Flags missing or empty required fields and optionally checks field types and URL validity.
"""

import re

REQUIRED_FIELDS = [
    "doc_id", "plant", "disease", "title", "section",
    "symptoms", "cause", "management", "lang", "text", "n_tokens", "split_idx", "crawl_date"
]
RECOMMENDED_FIELDS = ["prevention", "references"]

def is_valid_url(url):
    return re.match(r"https?://", url or "")

def validate_entry(entry):
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in entry or not entry[field]:
            errors.append(f"Missing or empty required field: {field}")
    # Check recommended fields
    for field in RECOMMENDED_FIELDS:
        if field not in entry or not entry[field]:
            errors.append(f"Missing or empty recommended field: {field}")
    # Check references (should be a list of URLs)
    if "references" in entry and entry["references"]:
        if not isinstance(entry["references"], list) or not all(is_valid_url(url) for url in entry["references"]):
            errors.append("Invalid references: should be a list of URLs")
    return errors

def validate_kb(kb):
    report = []
    for idx, entry in enumerate(kb):
        errs = validate_entry(entry)
        if errs:
            # Try to include source info if available (e.g., url, title, plant, disease)
            source = entry.get("url") or entry.get("references") or ""
            report.append({
                "idx": idx,
                "doc_id": entry.get("doc_id"),
                "plant": entry.get("plant"),
                "disease": entry.get("disease"),
                "title": entry.get("title"),
                "source": source,
                "errors": errs
            })
    return report

# Example usage:
# import json
# with open("data/kb/manifest.json") as f:
#     kb = json.load(f)
# report = validate_kb(kb)
# print(json.dumps(report, indent=2))