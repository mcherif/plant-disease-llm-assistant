import argparse
import os, sys
from pathlib import Path
# Ensure project src/ is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingestion.refresh_kb_descriptions import _extract_inline_for_disease
from bs4 import BeautifulSoup, Tag
import re

DEFAULT_HTML = r"data\disease_and_pests.html"

"""
Verifier for scoped parsing against a saved PlantVillage 'infos' HTML.

Use cases:
- Inspect summary/symptoms/cause extracted for specific diseases.
- List and parse all detected diseases on the page.
- Optionally filter to pathology categories (--only-diseases).
"""


def trunc(s, n=None):
    """Pretty-printer truncation helper (0/None disables truncation)."""
    if not s or n is None or n <= 0:
        return s
    if len(s) <= n:
        return s
    cut = s[:n].rstrip()
    # Prefer cutting at a sentence boundary if possible
    for sep in ".!?;:":
        idx = cut.rfind(sep)
        if idx >= int(n * 0.6):
            return cut[:idx + 1]
    return cut + " …"


def take_first_sentences(text: str, n: int) -> str:
    """Return the first n sentences from text (n <= 0 returns original)."""
    import re
    if not text or n <= 0:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(parts[:n]).strip()


def detect_disease_names(soup: BeautifulSoup, only_diseases: bool = False) -> list[str]:
    """
    Detect disease names from <h4> blocks on the infos page.

    Heuristics:
    - Skip generic headers (About, For farmer, Category…).
    - Track current Category and optionally keep only pathology categories.
    - Require either a Latin name (<i>) in the h4 or nearby scope markers
      (links-disease-* gallery, div.symptoms/div.cause, or h5/h6 headings).
    """
    names: list[str] = []
    seen: set[str] = set()
    generic = {"about", "for farmer", "for research", "profile"}
    # Categories considered plant diseases (keep); others (e.g., insects, mites) are filtered out
    disease_cats = {"fungal", "bacterial", "oomycete", "viral",
                    "viroid", "phytoplasma", "fungus", "bacteria", "virus"}
    cat_re = re.compile(r"^category\s*:\s*(.+)$", re.I)
    current_cat: str | None = None

    def has_scope_markers(h: Tag) -> bool:
        # Greedy but bounded scan of next siblings to confirm this h4 is a disease block.
        # Stops at the next h4, returns True if we see known markers for a disease section.
        steps = 0
        sib = h.next_sibling
        while sib and steps < 40:
            steps += 1
            if isinstance(sib, Tag):
                # Stop at next disease block
                if sib.name == "h4":
                    break
                # Known gallery container
                if (sib.get("id") or "").startswith("links-disease-"):
                    return True
                cls = set(sib.get("class") or [])
                # Direct content blocks
                if "symptoms" in cls or "cause" in cls:
                    return True
                # Headings that introduce sections
                if sib.name in ("h5", "h6", "strong", "b"):
                    ht = " ".join(sib.get_text(
                        " ", strip=True).split()).lower()
                    if any(k in ht for k in ("symptom", "cause", "causal", "etiology", "pathogen")):
                        return True
            sib = sib.next_sibling
        return False

    for h in soup.find_all("h4"):
        if not isinstance(h, Tag):
            continue
        # Direct text inside the h4 (before span/i Latin)
        t = h.find(string=True, recursive=False)
        name = " ".join(t.split()) if t else None
        if not name:
            continue
        low = name.strip().lower()

        # Update current category when encountering "Category : …" headings
        m = cat_re.match(low)
        if m:
            current_cat = m.group(1).strip().lower()
            continue

        # Skip obvious non-disease headers
        if low in generic or low.startswith("question"):
            continue

        # If requested, keep only items under disease categories
        if only_diseases and current_cat is not None:
            if current_cat not in disease_cats:
                continue

        # Heuristics: require Latin name in <i> OR presence of scope markers nearby
        i_present = h.find("i") is not None
        if not i_present and not has_scope_markers(h):
            continue

        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def main():
    ap = argparse.ArgumentParser(
        "Verify inline scoped parsing on a saved infos HTML")
    ap.add_argument("--html", default=DEFAULT_HTML,
                    help="Path to saved infos HTML")
    ap.add_argument("--disease", "-d", action="append",
                    help="Disease name to extract (can be repeated)")
    ap.add_argument("--all", action="store_true",
                    help="Parse all detected diseases in the HTML")
    ap.add_argument("--list", action="store_true",
                    help="List detected diseases and exit")
    ap.add_argument("--only-diseases", action="store_true",
                    help="Exclude insects/mites etc. by using Category sections")
    ap.add_argument("--desc-sentences", type=int, default=2,
                    help="Max sentences for the summary (0 = unlimited)")
    ap.add_argument("--symp-sentences", type=int, default=0,
                    help="Max sentences for symptoms when printing (0 = unlimited)")
    ap.add_argument("--cause-sentences", type=int, default=0,
                    help="Max sentences for cause when printing (0 = unlimited)")
    # NEW: control truncation widths (0 or negative disables truncation)
    ap.add_argument("--summary-width", type=int, default=220,
                    help="Max characters to print for summary (0 = no truncation)")
    ap.add_argument("--symptoms-width", type=int, default=500,
                    help="Max characters to print for symptoms (0 = no truncation)")
    ap.add_argument("--cause-width", type=int, default=400,
                    help="Max characters to print for cause (0 = no truncation)")
    ap.add_argument("--no-trunc", action="store_true",
                    help="Disable truncation for all fields")
    args = ap.parse_args()

    html_path = args.html or DEFAULT_HTML
    if not os.path.exists(html_path):
        print(f"Missing file: {html_path}")
        return

    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    detected = detect_disease_names(soup, only_diseases=args.only_diseases)

    if args.list:
        print("Detected diseases:")
        for name in detected:
            print(f"- {name}")
        return

    diseases = detected if args.all else (
        args.disease or ["Apple scab", "Black rot", "Cedar apple rust"])

    # Resolve widths (no-trunc overrides)
    sum_w = 0 if args.no_trunc else args.summary_width
    sym_w = 0 if args.no_trunc else args.symptoms_width
    cau_w = 0 if args.no_trunc else args.cause_width

    for d in diseases:
        desc, sym, cause = _extract_inline_for_disease(
            soup, d, max_desc_sentences=args.desc_sentences
        )
        sym_out = take_first_sentences(
            sym, args.symp_sentences) if sym else None
        cau_out = take_first_sentences(
            cause, args.cause_sentences) if cause else None

        print(f"\n=== {d} ===")
        print("Summary:", trunc(desc, sum_w))
        print("Symptoms:", trunc(sym_out, sym_w))
        print("Cause:", trunc(cau_out, cau_w))


if __name__ == "__main__":
    main()
