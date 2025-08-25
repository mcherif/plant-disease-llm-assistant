"""
Optional utility: build a consolidated JSON knowledge base from PlantVillage.

Primary ingestion now uses fetch_kb_texts.py to create per-class text files for RAG.
Keep this script if you want a single JSON (data/plantvillage_kb.json) for
inspection, testing, or alternative pipelines. Otherwise, it can be removed.
"""

import os
import re
import json
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
DATASET_DIR = "../plant-disease-mlops-full/PlantVillage-Dataset/raw/color"
OUTPUT_FILE = "data/plantvillage_kb.json"

# Add (or ensure) base infos URL
BASE_URL = "https://plantvillage.psu.edu/topics/{}/infos"

# Headers/session for scraping
HEADERS = {
    "User-Agent": "plant-disease-llm-assistant/1.0 (+https://github.com/DataTalksClub/plant-disease-llm-assistant)",
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
    if base in DISEASE_ALIASES:
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


def main():
    kb = {}

    # Gather (crop, disease) pairs from dataset folders
    crops = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(
        os.path.join(DATASET_DIR, d))]

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
            source_url = entry.get("url") or BASE_URL.format(slug)
            # If scraped desc is weak, try direct PV disease page using alias variants
            if len(description) < 40:
                desc2, url2 = try_plantvillage_direct(slug, disease_canon)
                if desc2:
                    description, source_url = desc2, url2
            if len(description) < 40:
                description, source_url = _wiki_fallback(
                    crop_name, disease_canon)
        else:
            # Not found in listing: try direct PV disease page with aliases first
            description, source_url = try_plantvillage_direct(
                slug, disease_canon)
            if not description:
                description, source_url = _wiki_fallback(
                    crop_name, disease_canon)
            # last resort: point to canonical PV URL if we have it
            if not description:
                key = (slug, _norm(disease_canon).lower())
                source_url = CANONICAL_URL_OVERRIDES.get(
                    key, BASE_URL.format(slug))

        # Handle healthy with a static line (override matched/title)
        if disease_name.lower() == "healthy":
            matched = "healthy"
            description = f"{crop_name} sample labeled healthy (no disease symptoms)."
            source_url = f"https://plantvillage.psu.edu/topics/{slug}/infos"

        kb[crop_name][disease_name] = {
            "matched_title": matched,
            "description": description if description else "⚠️ No description found yet.",
            "source": source_url,
        }

    # Remove temporary full listings
    for crop in list(kb.keys()):
        kb[crop].pop("_all_diseases", None)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)

    print(f"✅ Knowledge base saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
