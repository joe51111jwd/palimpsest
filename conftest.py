"""Repo-root conftest: make `palimpsest` and `bench` importable, and pin the
offline model/token caches so the whole suite runs with no network.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / "data" / "tiktoken_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# Benchmarks and tests must never spend LLM quota implicitly.
os.environ.setdefault("PALIMPSEST_LLM_OFFLINE", "1")
