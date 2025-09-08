"""
Optional utility: build a consolidated JSON knowledge base (plantvillage_kb.json) from PlantVillage.

DEPRECATED: Prefer src/ingestion/refresh_kb_descriptions.py, which extracts summary, symptoms, and cause
using scoped parsing of PlantVillage "infos" sections. This script only gathers a short description and
is best used for quick bootstrapping or debugging.

What this script does
- Scrapes a PlantVillage crop "infos" page and, if needed, disease pages.
- Collects a short description per disease (no scoped symptoms/cause here).
- Falls back to Wikipedia when PV text is missing.
- Writes a single JSON file for inspection/testing: data/plantvillage_kb.json.

When to use
- Ad-hoc or legacy workflows that prefer a single JSON artifact.
- Debugging/bootstrapping before using the production refresher.

Relation to production ingestion
- The production tool refresh_kb_descriptions.py updates summary/symptoms/cause
  using scoped parsing of PV "infos" sections and should be preferred for RAG.
- This script remains useful for quick end-to-end scrapes or initial KB seeding.

Usage
  python -m src.ingestion.scrape_plantvillage_infos --dataset_dir data\PlantVillage-Dataset\raw\color --out data\plantvillage_kb.json
"""

import os
import re
import json
import argparse
import warnings
import requests
from bs4 import BeautifulSoup
from difflib import get_close_matches
from tqdm import tqdm
from urllib.parse import urljoin
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
try:
    import wikipedia
    wikipedia.set_lang("en")
except Exception:
    wikipedia = None

# Paths
DEFAULT_DATASET_DIR = "data/PlantVillage-Dataset/raw/color"
DEFAULT_OUTPUT_FILE = "data/plantvillage_kb.json"

# Add (or ensure) base infos URL
BASE_URL = "https://plantvillage.psu.edu/topics/{}/infos"

# Headers/session for scraping
HEADERS = {
    "User-Agent": "plant-disease-rag-assistant/1.0 (+https://github.com/DataTalksClub/plant-disease-rag-assistant)",
    "Accept-Language": "en-US,en;q=0.8",
}
# Known PV URLs for tricky names/typos
CANONICAL_URL_OVERRIDES = {
    ("orange", "huanglongbing"): "https://plantvillage.psu.edu/topics/orange/diseases-and-pests/huanglongbing-citrus-greening",
    ("orange", "huanglongbing (citrus greening)"): "https://plantvillage.psu.edu/topics/orange/diseases-and-pests/huanglongbing-citrus-greening",
    ("orange", "citrus greening"): "https://plantvillage.psu.edu/topics/orange/diseases-and-pests/huanglongbing-citrus-greening",
}

# Alias map for misspellings
DISEASE_ALIASES = {
    "haunglongbing": ["huanglongbing", "huanglongbing (citrus greening)", "citrus greening", "hlb"],
    "haunglongbing (citrus greening)": ["huanglongbing (citrus greening)", "huanglongbing", "citrus greening", "hlb"],
    "hlb": ["huanglongbing", "citrus greening"],
}

# Map dataset crop names to PlantVillage slugs
CROP_SLUGS = {
    "Apple": "apple",
    "Blueberry": "blueberry",
    "Cherry (including sour)": "cherry",
    "Corn (maize)": "maize",
    "Grape": "grape",
    "Orange": "orange",
    "Peach": "peach",
    "Pepper, bell": "bell-pepper",
    "Potato": "potato",
    "Raspberry": "raspberry",
    "Soybean": "soybean",
    "Squash": "squash",
    "Strawberry": "strawberry",
    "Tomato": "tomato",
}

USER_AGENT = "plant-disease-rag-assistant/0.1 (+https://github.com/mcherif/plant-disease-rag-assistant)"

# Logo reference updated to plant-disease-rag-assistant-logo.png.

# --- Add these helpers (alias table + slug + URL fixer) ---
PV_TOPIC_ALIASES = {
    "Corn (maize)": ["corn-maize", "maize", "corn"],
    "Cherry (including sour)": ["cherry-including-sour", "cherry"],
    "Pepper, bell": ["pepper-bell", "bell-pepper", "pepper"],
    # add more as needed
}


def _slugify_topic(name: str) -> str:
    s = (name or "").lower().replace("&", "and")
    s = re.sub(r"[^\w]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


def _pv_candidate_slugs(plant: str) -> list[str]:
    # Prefer curated aliases; otherwise fall back to slugify(name)
    return PV_TOPIC_ALIASES.get(plant, [_slugify_topic(plant)])


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    retry = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET"],
        raise_on_status=False,
    )
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def ensure_valid_pv_infos_url(plant_name: str) -> str | None:
    """
    Return a working PlantVillage crop infos URL for the given plant name,
    trying alias slugs like 'corn-maize' if the default slug fails.
    """
    sess = _make_session()
    # Try default slug first, then aliases
    tried = set()
    for slug in [_slugify_topic(plant_name), *_pv_candidate_slugs(plant_name)]:
        if slug in tried:
            continue
        tried.add(slug)
        url = f"https://plantvillage.psu.edu/topics/{slug}/infos"
        try:
            r = sess.head(url, allow_redirects=True, timeout=10)
            if r.status_code == 200:
                return url
        except requests.RequestException:
            pass
    return None
# --- end helpers ---


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _slugify(s: str) -> str:
    s = _norm(s).lower()
    s = re.sub(r"\([^)]*\)", "", s)              # drop parentheticals
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")  # non-alnum -> hyphen
    s = re.sub(r"-{2,}", "-", s)                 # dedupe hyphens
    return s


def _extract_lead_text(soup: BeautifulSoup, max_chars: int = 800) -> str | None:
    og = soup.find("meta", attrs={"property": "og:description"})
    if og and og.get("content"):
        txt = _norm(og["content"])
        if len(txt) > 40:
            return txt[:max_chars]

    container = soup.find(["main", "article"]) or soup
    paras = []
    for p in container.find_all("p"):
        text = _norm(p.get_text(" "))
        if len(text) >= 40:
            paras.append(text)
        if sum(len(x) for x in paras) >= max_chars:
            break
    return (" ".join(paras)[:max_chars]) if paras else None


def _session() -> requests.Session:
    s = requests.Session()
    r = Retry(total=5, backoff_factor=0.4, status_forcelist=[
              429, 500, 502, 503, 504], allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.mount("http://", HTTPAdapter(max_retries=r))
    return s


def get_crop_diseases(crop_slug):
    """
    Return mapping: { disease_title: { 'desc': str, 'url': str } }
    Tries the 'infos' page; if empty, falls back to 'diseases-and-pests' detail pages.
    """
    sess = _session()
    diseases = {}

    # 1) Try the infos page for inline descriptions
    infos_url = BASE_URL.format(crop_slug)
    try:
        r = sess.get(infos_url, headers=HEADERS, timeout=20)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            cards = soup.select("div.pv-info-card")
            for card in cards:
                title_el = card.select_one("h3, h2")
                if not title_el:
                    continue
                title = title_el.get_text(strip=True)
                desc = ""
                p = card.find("p")
                if p:
                    desc = _norm(p.get_text(" "))
                link_el = card.find("a", href=True)
                detail_url = urljoin(
                    infos_url, link_el["href"]) if link_el else infos_url

                # If inline desc is too short, fetch detail page
                if len(desc) < 40 and link_el:
                    r2 = sess.get(detail_url, headers=HEADERS, timeout=20)
                    if r2.status_code == 200:
                        soup2 = BeautifulSoup(r2.text, "html.parser")
                        lead = _extract_lead_text(soup2)
                        if lead:
                            desc = lead
                            # be nice to the server
                            time.sleep(0.3)

                if title:
                    diseases[title] = {"desc": desc, "url": detail_url}
    except requests.RequestException:
        pass

    # 2) Fallback: scrape diseases-and-pests listing and detail pages
    if not diseases:
        listing = f"https://plantvillage.psu.edu/topics/{crop_slug}/diseases-and-pests"
        try:
            r = sess.get(listing, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                for a in soup.select(f'a[href*="/topics/{crop_slug}/"][href*="/"]'):
                    href = a.get("href")
                    title = _norm(a.get_text(" "))
                    # Heuristics to filter list items that look like disease pages
                    if not href or len(title) < 3 or "/topics/" not in href:
                        continue
                    detail_url = urljoin(listing, href)
                    try:
                        r2 = sess.get(detail_url, headers=HEADERS, timeout=20)
                        if r2.status_code == 200:
                            soup2 = BeautifulSoup(r2.text, "html.parser")
                            lead = _extract_lead_text(soup2)
                            if lead and title not in diseases:
                                diseases[title] = {
                                    "desc": lead, "url": detail_url}
                                time.sleep(0.3)
                    except requests.RequestException:
                        continue
        except requests.RequestException:
            pass

    return diseases


def normalize_name(name):
    """Clean up disease/crop names."""
    name = name.replace("_", " ").replace(
        "___", " ").replace(",", " ").replace("  ", " ")
    return name.strip()


def canonicalize_disease_name(name: str) -> str:
    """Return a canonical/most useful variant for lookups."""
    base = _norm(name).lower()
    # prefer the most explicit alias if misspelled
    if (base in DISEASE_ALIASES):
        for v in DISEASE_ALIASES[base]:
            if "huanglongbing (citrus greening)" in v:
                return v
        return DISEASE_ALIASES[base][0]
    return name


def variants_from_disease(name: str) -> list[str]:
    base = _norm(name).lower()
    alts = {
        base,
        re.sub(r"[_\-]+", " ", base),
        re.sub(r"\s*\([^)]*\)\s*", " ", base).strip(),
    }
    alts.update(DISEASE_ALIASES.get(base, []))
    out = []
    for v in alts:
        if v:
            out.extend([v, v.title()])
    # de-dup
    return list(dict.fromkeys(out))


def _wiki_fallback(crop: str, disease: str) -> tuple[str, str]:
    if not wikipedia:
        return "⚠️ No description found yet.", BASE_URL.format(CROP_SLUGS.get(crop, crop.lower()))
    query = f"{crop} {disease}"
    try:
        summary = wikipedia.summary(query, sentences=3, auto_suggest=False)
    except Exception:
        try:
            summary = wikipedia.summary(query, sentences=3, auto_suggest=True)
        except Exception:
            return "⚠️ No description found yet.", BASE_URL.format(CROP_SLUGS.get(crop, crop.lower()))
    return summary, f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"


def try_plantvillage_direct(crop_slug: str, disease_name: str) -> tuple[str | None, str | None]:
    sess = _session()
    # try canonical override first
    key = (crop_slug, _norm(disease_name).lower())
    if key in CANONICAL_URL_OVERRIDES:
        url = CANONICAL_URL_OVERRIDES[key]
        try:
            r = sess.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                lead = _extract_lead_text(soup)
                if lead:
                    return lead, url
        except requests.RequestException:
            pass

    # then variants -> slug guesses
    candidates = []
    for dv in variants_from_disease(disease_name):
        slug = _slugify(dv)
        candidates.extend([
            f"https://plantvillage.psu.edu/topics/{crop_slug}/diseases-and-pests/{slug}",
            f"https://plantvillage.psu.edu/topics/{crop_slug}/infos/{slug}",
            f"https://plantvillage.psu.edu/topics/{crop_slug}/diseases/{slug}",
        ])
    seen = set()
    dedup = []
    for u in candidates:
        if u not in seen:
            seen.add(u)
            dedup.append(u)

    for url in dedup:
        try:
            r = sess.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                lead = _extract_lead_text(soup)
                if lead:
                    return lead, url
        except requests.RequestException:
            continue
    return None, None


def parse_args():
    ap = argparse.ArgumentParser(
        description="[DEPRECATED] Build plantvillage_kb.json by scraping PlantVillage 'infos' pages (no scoped symptoms/cause). "
                    "Prefer: python -m src.ingestion.refresh_kb_descriptions")
    ap.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR,
                    help="Path to PlantVillage 'raw/color' directory")
    ap.add_argument("--out", default=DEFAULT_OUTPUT_FILE,
                    help="Output JSON file path")
    return ap.parse_args()


def main():
    # Emit a visible deprecation warning at runtime
    warnings.simplefilter("always", DeprecationWarning)
    warnings.warn(
        "scrape_plantvillage_infos.py is deprecated. Prefer src/ingestion/refresh_kb_descriptions.py "
        "for summary + symptoms + cause via scoped parsing.",
        DeprecationWarning,
        stacklevel=2,
    )
    args = parse_args()
    dataset_dir = args.dataset_dir
    out_file = args.out

    if not os.path.isdir(dataset_dir):
        raise SystemExit(
            f"Dataset dir not found: {dataset_dir}. Pass --dataset_dir to the script.")

    kb = {}

    # Gather (crop, disease) pairs from dataset folders
    crops = [d for d in os.listdir(dataset_dir) if os.path.isdir(
        os.path.join(dataset_dir, d))]

    for folder in tqdm(crops, desc="Processing crops"):
        parts = folder.split("___")
        crop_name = normalize_name(parts[0])
        disease_name = normalize_name(
            parts[1]) if len(parts) > 1 else "healthy"

        if crop_name not in CROP_SLUGS:
            continue

        slug = CROP_SLUGS[crop_name]
        if crop_name not in kb:
            kb[crop_name] = {}
            # Scrape diseases once per crop (title -> {desc,url})
            kb[crop_name]["_all_diseases"] = get_crop_diseases(slug)

        # Normalize requested disease (handles Haung->Huang)
        disease_canon = canonicalize_disease_name(disease_name)

        disease_map = kb[crop_name]["_all_diseases"]
        candidates = list(disease_map.keys())

        # Default to the requested disease to avoid UnboundLocalError
        matched = disease_name

        # Try alias variants to find closest match in scraped list
        match = None
        for dv in variants_from_disease(disease_canon):
            m = get_close_matches(dv, candidates, n=1, cutoff=0.5)
            if m:
                match = m[0]
                break

        if match:
            matched = match
            entry = disease_map.get(matched, {})
            description = entry.get("desc") or ""
            # If scraped desc is weak, try direct PV disease page using alias variants
            if len(description) < 40:
                desc2, _ = try_plantvillage_direct(slug, disease_canon)
                if desc2:
                    description = desc2
            if len(description) < 40:
                description = _wiki_fallback(crop_name, disease_canon)[0]
        else:
            # Not found in listing: try direct PV disease page with aliases first
            desc1, _ = try_plantvillage_direct(slug, disease_canon)
            description = desc1
            if not description:
                description = _wiki_fallback(crop_name, disease_canon)[0]
            # last resort removed (source_url was unused)

        # Handle healthy with a static line (override matched/title)
        if disease_name.lower() == "healthy":
            matched = "healthy"
            description = f"{crop_name} sample labeled healthy (no disease symptoms)."
            # (source_url removed; we always use base_infos below)

        base_infos = ensure_valid_pv_infos_url(
            f"https://plantvillage.psu.edu/topics/{slug}/infos") or f"https://plantvillage.psu.edu/topics/{_slugify_topic(crop_name)}/infos"

        kb[crop_name][disease_name] = {
            "matched_title": matched,
            "description": description if description else "⚠️ No description found yet.",
            "source": base_infos,
        }

    # Remove temporary full listings
    for crop in list(kb.keys()):
        kb[crop].pop("_all_diseases", None)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)

    print(f"✅ Knowledge base saved to {out_file}")


if __name__ == "__main__":
    main()
