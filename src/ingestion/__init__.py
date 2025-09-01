# Define the package API and metadata.

__all__ = [
    "ingest_docs",   # module
    "build_kb",      # function re-export (if present)
]

__version__ = "0.1.0"

# Re-export common entry points (adjust names to match your module)
try:
    from .ingest_docs import build_kb  # noqa: F401
except Exception:
    # Safe if build_kb isn’t implemented yet
    pass

# Public ingestion APIs
from .refresh_kb_descriptions import refresh_descriptions  # noqa: F401
from .refresh_kb_descriptions import _extract_inline_for_disease  # noqa: F401