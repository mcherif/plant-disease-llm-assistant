"""Tests for prompt templates.

Purpose
- Ensure the answer prompt includes required placeholders ({context}, {question}).
- Guard against KeyError at runtime by formatting the template with dummy data.

Rationale
- No external services or models are needed here; this is a fast, deterministic check
  to catch template regressions before they break the RAG pipeline.
"""
from pathlib import Path
import pytest

TPL = Path("src/llm/prompts/answer.txt")


@pytest.mark.skipif(not TPL.exists(), reason="answer.txt not found")
def test_answer_template_placeholders():
    """Validate placeholders exist and template formats without KeyError."""
    text = TPL.read_text(encoding="utf-8")
    assert "{context}" in text and "{question}" in text
    formatted = text.format(context="CTX", question="Q?")
    assert "CTX" in formatted and "Q?" in formatted
