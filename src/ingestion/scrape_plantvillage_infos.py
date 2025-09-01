"""
DEPRECATED: This entry point moved to scripts/scrape_plantvillage_infos.py.

Prefer the production refresher:
  python -m src.ingestion.refresh_kb_descriptions

For the legacy single-file scraper, run:
  python scripts\scrape_plantvillage_infos.py --help
"""
from __future__ import annotations
from pathlib import Path
import sys
import warnings


def main():
    warnings.simplefilter("always", DeprecationWarning)
    warnings.warn(
        "scrape_plantvillage_infos.py moved to scripts/. "
        "Prefer src/ingestion/refresh_kb_descriptions.py for production refresh.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Best effort: forward to the script if available on disk
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    target = scripts_dir / "scrape_plantvillage_infos.py"
    if target.exists():
        sys.path.insert(0, str(scripts_dir))
        mod = __import__("scrape_plantvillage_infos")
        if hasattr(mod, "main"):
            return mod.main()
    print("Tip: run `python scripts\\scrape_plantvillage_infos.py --help`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
