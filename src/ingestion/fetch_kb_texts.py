#!/usr/bin/env python3
"""
Builds a lightweight textual knowledge base for PlantVillage classes:  writes per-class .txt files for RAG ingestion (with Wikipedia-first and PlantVillage fallback)

What it does:
- Discovers class folders in RAW_DATA_DIR (prefers raw/color; ignores grayscale/segmented)
- Parses "<Crop>___<Disease>" labels and normalizes names
- Attempts to retrieve a short description from Wikipedia first (title + summary)
- Falls back to scraping PlantVillage crop “infos” and disease detail pages
- Handles common aliases/misspellings (e.g., haunglongbing/HLB)
- Writes one text file per class to OUTPUT_KB_DIR with a "Source:" header
- Emits:
    - classes_cleaned.txt  (raw_label, normalized_query, filename, source)
    - failed.txt           (labels with no description found)

Usage:
    python src/ingestion/fetch_kb_texts.py

Configuration:
    RAW_DATA_DIR      Path to the PlantVillage dataset root (default: ../plant-disease-mlops-full/PlantVillage-Dataset/raw)
    OUTPUT_KB_DIR     Output directory for text snippets (default: data/kb)
    WIKIPEDIA_LANG    Wikipedia language code (default: "en")
    USER_AGENT        Custom UA string for HTTP requests

Requirements:
    pip install wikipedia requests beautifulsoup4 tqdm

Notes:
- Network access required. The script uses polite delays and a custom User-Agent.
- “healthy” classes get a short static description without scraping.
"""
import os
import re
import time
import wikipedia
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urljoin
import unicodedata
import warnings
from bs4 import GuessedAtParserWarning

# Silence BS4 parser warning emitted by the wikipedia package internally
warnings.filterwarnings("ignore", category=GuessedAtParserWarning)

# CONFIG
# where your dataset class folders live (change if needed)
# importing from this path to save space on the project
RAW_DATA_DIR = "../plant-disease-mlops-full/PlantVillage-Dataset/raw"
OUTPUT_KB_DIR = "data/kb"
CLASSES_FILE = os.path.join(OUTPUT_KB_DIR, "classes_cleaned.txt")
FAILED_FILE = os.path.join(OUTPUT_KB_DIR, "failed.txt")
WIKIPEDIA_LANG = "en"
USER_AGENT = "plant-disease-kb-fetcher/1.0 (+https://github.com/mcherif/plant-disease-llm-assistant)"

wikipedia.set_lang(WIKIPEDIA_LANG)
HEADERS = {"User-Agent": USER_AGENT}

os.makedirs(OUTPUT_KB_DIR, exist_ok=True)

# Common disease aliases/misspellings
DISEASE_ALIASES = {
    # Huanglongbing (Citrus greening) common typos/aliases
    "haunglongbing": ["huanglongbing", "citrus greening", "huanglongbing (citrus greening)", "hlb"],
    "haunglongbing (citrus greening)": ["huanglongbing (citrus greening)", "huanglongbing", "citrus greening", "hlb"],
    "hlb": ["huanglongbing", "citrus greening"],
}


def variants_from_disease(name: str) -> list[str]:
    base = _norm(name).lower()
    # normalize simple shape variants
    alts = {
        base,
        re.sub(r"[_\-]+", " ", base),
        re.sub(r"\s*\([^)]*\)\s*", " ", base).strip(),  # drop parentheticals
    }
    alts.update(DISEASE_ALIASES.get(base, []))
    # include title-case forms too
    out = []
    for v in alts:
        if v:
            out.extend([v, v.title()])
    # dedupe while preserving order
    return list(dict.fromkeys(_norm(v) for v in out if v))


# Helper: discover classes from folder names (fallback: look for labels.txt)
IGNORE_DIRS = {"segmented", "grayscale", "color", "__pycache__", ".git"}
IGNORE_LABELS = {"segmented", "background", "unknown"}


def discover_classes(raw_dir=RAW_DATA_DIR):
    """
    Return a list of PlantVillage class folder names like 'Apple___Apple_scab'.
    - If raw_dir contains 'color' or 'grayscale', prefer 'color' (else 'grayscale').
    - Ignore non-class folders like 'segmented'.
    - Fallback to labels.txt if present.
    """
    base = raw_dir
    for sub in ("color", "grayscale"):
        p = os.path.join(raw_dir, sub)
        if os.path.isdir(p):
            base = p
            break

    classes = []
    if os.path.isdir(base):
        for name in os.listdir(base):
            path = os.path.join(base, name)
            if not os.path.isdir(path):
                continue
            if name.lower() in IGNORE_DIRS:
                continue
            # PlantVillage class dirs have '___' delimiter
            if "___" not in name:
                continue
            classes.append(name)

    # Fallback: labels.txt in base
    labels_txt = os.path.join(base, "labels.txt")
    if not classes and os.path.isfile(labels_txt):
        with open(labels_txt, "r", encoding="utf-8") as f:
            for line in f:
                lab = line.strip()
                if not lab or lab.lower() in IGNORE_LABELS:
                    continue
                classes.append(lab)

    return sorted(set(classes))

# Normalization helpers


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _slugify(s: str) -> str:
    s = _norm(s).lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s

# Normalization / cleaning


def clean_label(raw):
    # replace underscores and multiple separators with single space
    s = raw.replace("-", " ").replace("/", " ").replace(".", " ")
    s = re.sub(r"[_\s]+", " ", s).strip()
    s = s.replace(",", " ")
    # canonical query and fs-safe filename
    query = _norm(s)
    filename = re.sub(r"[^\w\s\-]", "", query).strip().replace(" ", "_")
    return query, filename

# Try to fetch Wikipedia page content


def fetch_wikipedia(query: str, sentences: int = 3) -> tuple[str | None, str | None]:
    """Return (canonical_title, summary) or (None, None)."""
    query = _norm(query)
    try:
        summary = wikipedia.summary(
            query, sentences=sentences, auto_suggest=False)
        try:
            page = wikipedia.page(query, auto_suggest=False, preload=False)
            title = page.title
        except Exception:
            title = query
        return title, summary
    except Exception:
        # fallback with autosuggest
        try:
            summary = wikipedia.summary(
                query, sentences=sentences, auto_suggest=True)
            try:
                page = wikipedia.page(query, auto_suggest=True, preload=False)
                title = page.title
            except Exception:
                title = query
            return title, summary
        except Exception:
            # final fallback: search and try the top hit
            try:
                hits = wikipedia.search(query)
                if hits:
                    hit = hits[0]
                    summary = wikipedia.summary(
                        hit, sentences=sentences, auto_suggest=True)
                    try:
                        page = wikipedia.page(
                            hit, auto_suggest=True, preload=False)
                        title = page.title
                    except Exception:
                        title = hit
                    return title, summary
            except Exception:
                pass
            return None, None


def _extract_lead_text(soup: BeautifulSoup, max_chars: int = 700) -> str | None:
    # Prefer og:description if present
    og = soup.find("meta", attrs={"property": "og:description"})
    if og and og.get("content"):
        txt = _norm(og["content"])
        if len(txt) > 40:
            return txt[:max_chars]

    # Otherwise collect first meaningful paragraphs in main/article/content
    container = soup.find(["main", "article"])
    if not container:
        # heuristic: any div with content-ish class
        container = soup.find("div", class_=re.compile(
            "content|article|body", re.I)) or soup

    paras = []
    for p in container.find_all("p"):
        text = _norm(p.get_text(" "))
        if len(text) >= 40:
            paras.append(text)
        if sum(len(x) for x in paras) >= max_chars:
            break

    if paras:
        joined = " ".join(paras)
        return joined[:max_chars]
    return None


def fetch_plantvillage(crop: str, disease: str) -> tuple[str | None, str | None]:
    """
    Try to find the disease detail page starting from the crop 'infos' page,
    then extract a short description. Returns (description, source_url).
    """
    crop_slug = _slugify(crop)
    disease_norm = _norm(disease).lower()

    base_infos = f"https://plantvillage.psu.edu/topics/{crop_slug}/infos"
    try:
        r = requests.get(base_infos, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None, None
        soup = BeautifulSoup(r.text, "html.parser")

        # Find the disease card/link on the infos page
        candidate_href = None
        for a in soup.find_all("a", href=True):
            text = _norm(a.get_text()).lower()
            if not text:
                continue
            # loose match on name in link text
            if disease_norm in text or text in disease_norm:
                candidate_href = a["href"]
                # ensure it looks like a topic detail link
                if "/topics/" in candidate_href:
                    break

        # If found, follow it and extract description
        if candidate_href:
            detail_url = urljoin(base_infos, candidate_href)
            r2 = requests.get(detail_url, headers=HEADERS, timeout=15)
            if r2.status_code == 200:
                soup2 = BeautifulSoup(r2.text, "html.parser")
                desc = _extract_lead_text(soup2)
                if desc:
                    return desc, detail_url

        # If not found, try some common detail URL patterns directly
        disease_slug = _slugify(disease)
        candidates = [
            f"https://plantvillage.psu.edu/topics/{crop_slug}/diseases-and-pests/{disease_slug}",
            f"https://plantvillage.psu.edu/topics/{crop_slug}/infos/{disease_slug}",
            f"https://plantvillage.psu.edu/topics/{crop_slug}/diseases/{disease_slug}",
        ]
        for url in candidates:
            r3 = requests.get(url, headers=HEADERS, timeout=15)
            if r3.status_code == 200:
                soup3 = BeautifulSoup(r3.text, "html.parser")
                desc = _extract_lead_text(soup3)
                if desc:
                    return desc, url
    except requests.RequestException:
        pass

    return None, None


def save_text(filename, title_or_url, content):
    outpath = os.path.join(OUTPUT_KB_DIR, filename + ".txt")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(f"# Source: {title_or_url}\n\n")
        f.write(content)
    return outpath


def main():
    classes = discover_classes()
    print(f"Discovered {len(classes)} classes.")
    failed = []
    cleaned_map = []

    for raw in tqdm(classes):
        query, fname = clean_label(raw)
        content = None
        source = None

        # Parse crop/disease from label
        if "___" in raw:
            a, b = raw.split("___", 1)
            crop_name = _norm(a.replace("_", " "))
            disease_name = _norm(b.replace("_", " "))
        else:
            toks = query.split()
            crop_name = toks[0] if toks else query
            disease_name = " ".join(toks[1:]) if len(toks) > 1 else query

        # Healthy shortcut
        if disease_name.lower() == "healthy":
            content = f"{crop_name} sample labeled healthy (no disease symptoms)."
            source = "dataset:label"

        # Wikipedia first, trying alias variants
        if not content:
            disease_variants = variants_from_disease(disease_name)
            query_variants = []
            for dv in disease_variants:
                query_variants.extend(
                    [f"{crop_name} {dv}", dv, f"{dv} {crop_name}"])
            query_variants = list(dict.fromkeys(
                q for q in query_variants if q))

            for qv in query_variants:
                title, txt = fetch_wikipedia(qv)
                if txt:
                    content = txt
                    source = f"wikipedia:{title}"
                    break
                time.sleep(0.3)

        # PlantVillage fallback, try aliases too
        if not content:
            for dv in (disease_variants if 'disease_variants' in locals() else [disease_name]):
                desc, url = fetch_plantvillage(crop_name, dv)
                if desc:
                    content = desc
                    source = url
                    break

        if content:
            save_text(fname, source, content)
            cleaned_map.append((raw, query, fname + ".txt", source))
        else:
            failed.append(raw)

        time.sleep(0.2)

    # write cleaned classes map
    with open(CLASSES_FILE, "w", encoding="utf-8") as f:
        for raw, query, fname, source in cleaned_map:
            f.write(f"{raw}\t{query}\t{fname}\t{source}\n")

    # write failures
    with open(FAILED_FILE, "w", encoding="utf-8") as f:
        for r in failed:
            f.write(r + "\n")

    print(
        f"\nDone. Saved {len(cleaned_map)} docs, {len(failed)} failures. See {OUTPUT_KB_DIR}")


if __name__ == "__main__":
    main()
