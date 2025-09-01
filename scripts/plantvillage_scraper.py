"""
PlantVillage single-page scraper.

Purpose:
- Scrape a single PlantVillage plant "infos" page (or a saved HTML file).
- Extract per-disease blocks (h4 sections) and map Symptoms/Cause correctly.
- Optionally merge the parsed data into data/plantvillage_kb.json.

How it works:
- Finds the #diseases anchor and iterates subsequent h4 disease headings.
- Uses classed divs (div.symptoms, div.cause) or nearby h5 labels ("Symptoms", "Cause")
  to capture the correct text.
- Keeps the h4 title as "description" (includes Latin name when present).

Usage:
- Print parsed JSON:
  python scripts\plantvillage_scraper.py --url https://plantvillage.psu.edu/topics/apple/infos
  python scripts\plantvillage_scraper.py --file path\to\apple_infos.html
- Merge into KB:
  python scripts\plantvillage_scraper.py --url https://plantvillage.psu.edu/topics/apple/infos --write-kb

Notes:
- Best for ad-hoc additions or debugging layout issues.
- For bulk refresh across many plants/diseases, use:
  python -m src.ingestion.refresh_kb_descriptions
- Developer utility (not for production KB refresh).
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import requests
from bs4 import BeautifulSoup, Tag
import warnings


def load_html(url: Optional[str], file: Optional[Path]) -> Tuple[str, str]:
    if url:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.text, url
    elif file:
        html = file.read_text(encoding="utf-8", errors="ignore")
        return html, file.as_uri()
    else:
        raise ValueError("Provide either --url or --file")


def get_plant_name(soup: BeautifulSoup) -> Optional[str]:
    h1 = soup.select_one("h1 span")
    return h1.get_text(strip=True) if h1 else None


def get_diseases_container(soup: BeautifulSoup) -> Optional[Tag]:
    # The anchor is <div id="diseases"><h3>...</h3></div> then the content is in the next sibling <div>
    anchor = soup.select_one("#diseases")
    if not anchor:
        return None
    # some pages put content right after; find the next element div that actually contains disease blocks
    node = anchor
    while node and (not isinstance(node, Tag) or node.name != "div"):
        node = node.next_sibling
    # first sibling is the small wrapper, often an h4 Category inside; if not, use next div
    if isinstance(node, Tag) and node.name == "div":
        return node
    return None


def iter_tags_until_next_h4(start_h4: Tag):
    for sib in start_h4.next_siblings:
        if isinstance(sib, Tag) and sib.name == "h4":
            break
        yield sib


def clean_text(t: str) -> str:
    return " ".join(t.split())


def parse_diseases(soup: BeautifulSoup, source_url: str) -> Dict[str, Dict]:
    container = get_diseases_container(soup)
    if not container:
        return {}

    diseases: Dict[str, Dict] = {}
    for h4 in container.find_all("h4"):
        title = h4.get_text(" ", strip=True)
        # skip category headers like "Category : Fungal"
        if title.lower().startswith("category"):
            continue

        # Disease name is the text content of the h4 without the italic species part if present
        # First child of h4 is usually the disease name text node
        disease_name = h4.contents[0].strip() if h4.contents else title

        symptoms = None
        cause = None

        for node in iter_tags_until_next_h4(h4):
            if not isinstance(node, Tag):
                continue

            # Either look for label h5 then next div, or directly the classed divs
            if node.name == "h5":
                label = node.get_text(strip=True).lower()
                if "symptom" in label:
                    next_div = node.find_next_sibling(lambda t: isinstance(
                        t, Tag) and t.name == "div" and "symptoms" in (t.get("class") or []))
                    if next_div:
                        symptoms = clean_text(
                            next_div.get_text(" ", strip=True))
                elif "cause" in label:
                    next_div = node.find_next_sibling(lambda t: isinstance(
                        t, Tag) and t.name == "div" and "cause" in (t.get("class") or []))
                    if next_div:
                        cause = clean_text(next_div.get_text(" ", strip=True))
            elif node.name == "div":
                classes = node.get("class") or []
                if "symptoms" in classes and not symptoms:
                    symptoms = clean_text(node.get_text(" ", strip=True))
                if "cause" in classes and not cause:
                    cause = clean_text(node.get_text(" ", strip=True))

        diseases[disease_name] = {
            "matched_title": disease_name,
            "description": title,  # keeps the header including species text if present
            "source": source_url,
        }
        if symptoms:
            diseases[disease_name]["symptoms"] = symptoms
        if cause:
            diseases[disease_name]["cause"] = cause

    return diseases


def parse_plant_page(html: str, source_url: str) -> Dict[str, Dict]:
    soup = BeautifulSoup(html, "lxml")
    plant = get_plant_name(soup) or "Unknown"
    diseases = parse_diseases(soup, source_url)
    return {plant: diseases}


def merge_into_kb(kb_path: Path, new_data: Dict[str, Dict]) -> Dict:
    if kb_path.exists():
        kb = json.loads(kb_path.read_text(encoding="utf-8"))
    else:
        kb = {}
    # Merge/replace per-plant diseases with newly parsed ones
    for plant, diseases in new_data.items():
        kb.setdefault(plant, {})
        kb[plant].update(diseases)
    kb_path.write_text(json.dumps(
        kb, ensure_ascii=False, indent=2), encoding="utf-8")
    return kb


def main():
    ap = argparse.ArgumentParser(
        description="Developer utility: scrape a single PlantVillage 'infos' page and map Symptoms/Cause. For production KB refresh use: python -m src.ingestion.refresh_kb_descriptions")
    ap.add_argument(
        "--url", help="PlantVillage plant page URL, e.g., https://plantvillage.psu.edu/topics/apple/infos")
    ap.add_argument("--file", type=Path, help="Local HTML file path")
    ap.add_argument("--write-kb", action="store_true",
                    help="Write/merge into data/plantvillage_kb.json (dev use; prefer the refresher for full KB)")
    args = ap.parse_args()

    html, src = load_html(args.url, args.file)
    data = parse_plant_page(html, src)

    if args.write_kb:
        warnings.warn("plantvillage_scraper.py is a developer utility. For production KB refresh (with summary/symptoms/cause), use src/ingestion/refresh_kb_descriptions.py", UserWarning)
        kb_path = Path(__file__).parents[1] / "data" / "plantvillage_kb.json"
        merge_into_kb(kb_path, data)
        print(f"Merged into {kb_path}")
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
