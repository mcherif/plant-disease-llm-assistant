"""
Validate and optionally fix 'source' URLs in data/plantvillage_kb.json.

What it does:
- Normalizes URLs (force https, strip trailing slash).
- Probes each URL with HEAD (fallback to GET) and follows redirects.
- Updates entries to the final canonical URL when redirected.
- Backfills missing sources with the PlantVillage topic infos URL for the plant.
- Supports regex filters to limit plants/diseases.
- Dry-run by default; use --apply to write changes (to --out if provided, else --in).

Examples:
- Dry run with logs:
  python -m src.ingestion.validate_fix_kb_urls --verbose
- Apply fixes in place:
  python -m src.ingestion.validate_fix_kb_urls --apply --verbose
- Preview to a new file:
  python -m src.ingestion.validate_fix_kb_urls --apply --out data\\plantvillage_kb.updated.json --verbose
"""

import argparse
import json
import os
import re
import sys
from typing import Optional, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Reuse slug helpers/aliases for PlantVillage topics
from src.ingestion.refresh_kb_descriptions import _topic_slugs_for

PV_HOST = "plantvillage.psu.edu"


def make_session() -> requests.Session:
    """Create a requests session with retries and a custom User-Agent."""
    sess = requests.Session()
    retry = Retry(
        total=3, connect=3, read=3, backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("HEAD", "GET"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry, pool_connections=16, pool_maxsize=16)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    sess.headers.update({"User-Agent": "pv-kb-validator/0.1"})
    return sess


def normalize_url(url: str) -> str:
    """Normalize a URL by forcing https and removing any trailing slash."""
    if not url:
        return url
    url = url.strip()
    # force https and strip trailing slash
    url = re.sub(r"^http://", "https://", url, flags=re.I)
    if url.endswith("/"):
        url = url[:-1]
    return url


def probe_url(sess: requests.Session, url: str, timeout: float = 8.0) -> Tuple[Optional[int], Optional[str]]:
    """
    Probe a URL with HEAD (fallback to GET on 4xx/405) and follow redirects.

    Returns:
      (status_code, final_url) where final_url is normalized (https, no trailing slash).
      Returns (None, None) on network errors.
    """
    try:
        r = sess.head(url, allow_redirects=True, timeout=timeout)
        # Some hosts don’t support HEAD well, fall back to GET (streaming)
        if r.status_code >= 400 or r.status_code == 405:
            r = sess.get(url, allow_redirects=True,
                         timeout=timeout, stream=True)
        final = str(r.url) if r.url else url
        return r.status_code, normalize_url(final)
    except requests.RequestException:
        return None, None


def best_infos_url(plant: str) -> Optional[str]:
    """Return the PlantVillage topic infos URL for a plant, using known aliases."""
    # Prefer PlantVillage topic infos page for this plant
    for slug in _topic_slugs_for(plant):
        return f"https://{PV_HOST}/topics/{slug}/infos"
    return None


def validate_and_fix(kb_path: str, out_path: Optional[str], apply: bool,
                     plant_filter: Optional[str], disease_filter: Optional[str],
                     verbose: bool, timeout: float) -> dict:
    """
    Validate and optionally fix 'source' URLs in the KB.

    Args:
      kb_path: input KB JSON path.
      out_path: output path (used when --apply; defaults to kb_path if None).
      apply: when True, write changes; otherwise dry-run.
      plant_filter: optional regex to include matching plant names only.
      disease_filter: optional regex to include matching disease names only.
      verbose: print per-entry logs.
      timeout: network timeout for URL probes.

    Returns:
      stats dict with counts: checked, ok, updated, missing, error, skipped.
    """
    stats = {"checked": 0, "ok": 0, "updated": 0,
             "missing": 0, "error": 0, "skipped": 0}
    with open(kb_path, "r", encoding="utf-8") as f:
        kb = json.load(f)

    pf_re = re.compile(plant_filter, re.I) if plant_filter else None
    df_re = re.compile(disease_filter, re.I) if disease_filter else None

    sess = make_session()
    changed = False

    for plant, diseases in kb.items():
        if pf_re and not pf_re.search(plant):
            continue
        for disease, entry in diseases.items():
            if disease.lower() == "healthy":
                continue
            if df_re and not df_re.search(disease):
                continue

            stats["checked"] += 1
            src = (entry or {}).get("source")
            if not src:
                stats["missing"] += 1
                # Try to backfill a reasonable default (plant infos)
                guess = best_infos_url(plant)
                if apply and guess:
                    entry["source"] = guess
                    changed = True
                    stats["updated"] += 1
                    if verbose:
                        print(f"[ADD] {plant} / {disease}: source -> {guess}")
                else:
                    if verbose:
                        print(f"[MISS] {plant} / {disease}: no source")
                continue

            old = src
            src = normalize_url(src)
            if src != old:
                if apply:
                    entry["source"] = src
                    changed = True
                    stats["updated"] += 1
                    if verbose:
                        print(f"[NORM] {plant} / {disease}: {old} -> {src}")
                else:
                    if verbose:
                        print(f"[NORM?] {plant} / {disease}: {old} -> {src}")

            code, final = probe_url(sess, src, timeout=timeout)
            if code is None:
                stats["error"] += 1
                if verbose:
                    print(f"[ERR] {plant} / {disease}: {src} (no response)")
                continue

            # If it redirects, update to final canonical URL
            if final and final != src:
                if apply:
                    entry["source"] = final
                    changed = True
                    stats["updated"] += 1
                    if verbose:
                        print(
                            f"[REDIR] {plant} / {disease}: {src} -> {final} ({code})")
                else:
                    if verbose:
                        print(
                            f"[REDIR?] {plant} / {disease}: {src} -> {final} ({code})")
            else:
                if 200 <= (code or 0) < 300:
                    stats["ok"] += 1
                    if verbose:
                        print(f"[OK] {plant} / {disease}: {src} ({code})")
                else:
                    # Try to switch http->https or fall back to plant infos
                    fallback = best_infos_url(plant)
                    if apply and fallback and fallback != entry.get("source"):
                        entry["source"] = fallback
                        changed = True
                        stats["updated"] += 1
                        if verbose:
                            print(
                                f"[FIX] {plant} / {disease}: {src} -> {fallback} ({code})")
                    else:
                        stats["skipped"] += 1
                        if verbose:
                            print(f"[BAD] {plant} / {disease}: {src} ({code})")

    if apply:
        outp = out_path or kb_path
        os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(kb, f, ensure_ascii=False, indent=2)
    return stats


def main():
    ap = argparse.ArgumentParser("Validate and optionally fix KB source URLs")
    ap.add_argument("--in", dest="inp",
                    default=r"data\plantvillage_kb.json", help="Input KB JSON")
    ap.add_argument("--out", dest="outp", default=None,
                    help="Output path (defaults to --in when --apply)")
    ap.add_argument("--apply", action="store_true",
                    help="Apply fixes and write output")
    ap.add_argument("--plant", help="Regex to filter plant names")
    ap.add_argument("--disease", help="Regex to filter disease names")
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    stats = validate_and_fix(
        kb_path=args.inp,
        out_path=args.outp,
        apply=args.apply,
        plant_filter=args.plant,
        disease_filter=args.disease,
        verbose=args.verbose,
        timeout=args.timeout,
    )
    print("Checked={checked} OK={ok} Updated={updated} Missing={missing} Error={error} Skipped={skipped}".format(**stats))


if __name__ == "__main__":
    main()
