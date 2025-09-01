"""
HTML parsing debug helper.

Purpose:
- Parse a local HTML file and extract a concise text snippet for quick inspection.
- Supports Wikipedia (lead text) and PlantVillage "infos" pages.

How it works:
- Wikipedia: uses the shared _extract_lead_text to grab the intro.
- PlantVillage: unwraps view-source dumps, removes chrome, collects paragraphs/list items.

Usage:
  python scripts\parse_html_debug.py --html path\to\file.html --source wikipedia
  python scripts\parse_html_debug.py --html path\to\file.html --source plantvillage
  python scripts\parse_html_debug.py --html path\to\file.html --out out.txt
"""

from ingestion.fetch_kb_texts import _extract_lead_text  # type: ignore
import argparse
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Tuple, Optional
from html import unescape
import sys

# Make project `src/` importable when running the script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Reuse Wikipedia lead-text extractor


def _unwrap_view_source_html(html: str) -> str:
    """If this is a saved `view-source:` page, pull original HTML from <pre> and unescape."""
    if "<pre" in html and "&lt;" in html:
        soup = BeautifulSoup(html, "html.parser")
        pre = soup.find("pre")
        if pre:
            raw = pre.get_text("\n", strip=False)
            candidate = unescape(raw)
            if "<html" in candidate or "<body" in candidate:
                return candidate
    if "&lt;html" in html and "<html" not in html:
        return unescape(html)
    return html


def parse_wikipedia_html(html: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract (title, lead_text) from a Wikipedia article HTML."""
    soup = BeautifulSoup(html, "html.parser")
    title = None
    h1 = soup.select_one("#firstHeading")
    if h1:
        title = h1.get_text(strip=True)
    if not title and soup.title:
        title = soup.title.get_text(strip=True)
    text = _extract_lead_text(soup, max_chars=700)
    return title, text


def parse_plantvillage_html(html: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract (title, short_text) from a PlantVillage infos HTML (after unwrapping and stripping chrome)."""
    html = _unwrap_view_source_html(html)
    soup = BeautifulSoup(html, "html.parser")
    # Try common title locations
    title = None
    title_el = soup.select_one(
        "h1#page-title, h1.node-title, main h1, article h1, .content h1, .page-title, h1")
    if not title_el:
        og = soup.find("meta", attrs={"property": "og:title"})
        if og:
            title = og.get("content")
    if not title:
        title = title_el.get_text(strip=True) if title_el else (
            soup.title.get_text(strip=True) if soup.title else None)

    # Drop chrome like nav/footers
    for sel in ["nav", "header", "footer", "aside", ".site-header", ".site-footer", ".menu", ".breadcrumbs"]:
        for tag in soup.select(sel):
            tag.decompose()

    # Heuristic: paragraphs from likely content containers
    paras = []
    for p in soup.select("main p, article p, .content p, .entry-content p, .field-name-body p, .field-item p, #content p, .node-content p, p"):
        t = p.get_text(" ", strip=True)
        if t:
            paras.append(t)
        if sum(len(x) for x in paras) > 900:
            break

    # Fallback to list items if no paragraphs
    if not paras:
        for li in soup.select("main li, article li, .content li, .field-name-body li, .field-item li, li"):
            t = li.get_text(" ", strip=True)
            if t:
                paras.append(f"- {t}")
            if sum(len(x) for x in paras) > 900:
                break

    text = " ".join(paras)[:1200] if paras else None
    return title, text


def main():
    ap = argparse.ArgumentParser(
        description="Parse a local HTML file to test extraction.")
    ap.add_argument("--html", required=True, help="Path to local HTML file")
    ap.add_argument("--source", choices=["wikipedia", "plantvillage"], default="plantvillage",
                    help="Selector strategy for parsing")
    ap.add_argument(
        "--out", help="Optional path to write a KB-style .txt (with Source header)")
    args = ap.parse_args()

    html = Path(args.html).read_text(encoding="utf-8")

    if args.source == "wikipedia":
        title, text = parse_wikipedia_html(html)
    else:
        title, text = parse_plantvillage_html(html)

    print(f"Title: {title or '(none)'}\n")
    print(f"Extracted text:\n{text or '(no text extracted)'}\n")

    if args.out and text:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(f"""# Source: local:{Path(args.html).name}

{text}
""", encoding="utf-8")
        print(f"Saved: {outp}")


if __name__ == "__main__":
    main()
