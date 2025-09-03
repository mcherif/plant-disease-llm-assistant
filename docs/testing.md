# Testing Guide

This project uses pytest with live CLI logging (see pytest.ini).

Quick start
- Run all tests:
  - PowerShell: pytest -q
- Only run the OpenAI integration test:
  - Requires OPENAI_API_KEY to be set.
  - PowerShell: pytest -q -k openai
- Only run offline/mock tests:
  - Unset the key: Remove-Item Env:OPENAI_API_KEY
  - Then: pytest -q -k mock

Environment variables
- OPENAI_API_KEY: enables the real LLM call in the integration test (skipped if not set).
  - Set (current terminal): $env:OPENAI_API_KEY="<your-key>"
  - Unset (current terminal): Remove-Item Env:OPENAI_API_KEY
- OPENAI_MODEL (optional): defaults to gpt-4o-mini

Verbosity and logs
- Default logging is INFO (configured in pytest.ini).
- See more details (e.g., HF downloads, OpenAI HTTP calls):
  - pytest -q -o log_cli_level=DEBUG -k openai
- Example logs include:
  - FAISS load capabilities (e.g., AVX2)
  - SentenceTransformer model downloads from Hugging Face (first run only)
  - OpenAI chat.completions request/response and basic answer stats

What the tests do
- tests/test_rag_pipeline.py
  - Unit test with a mocked LLM (_generate) to keep it offline and deterministic.
  - Optional integration smoke test that calls OpenAI when OPENAI_API_KEY is set.
- tests/test_prompts.py
  - Validates that src/llm/prompts/answer.txt has required placeholders and formats safely.

Performance tips
- First run will download the embedding model (BAAI/bge-small-en-v1.5) from Hugging Face; subsequent runs are faster due to local cache.
- If you change logging verbosity frequently, you can override per-run without editing pytest.ini:
  - pytest -q -o log_cli_level=INFO