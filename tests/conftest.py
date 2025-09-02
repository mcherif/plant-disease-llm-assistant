from pathlib import Path
import sys

# Ensure the repo root is on sys.path so `src` is importable during tests.
# This avoids needing an editable install in the dev environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))