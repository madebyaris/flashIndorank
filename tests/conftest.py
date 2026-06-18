"""Shared test setup.

Point the model cache at a stable location so weights are downloaded at most
once across test runs. Must run before ``flashindorank.config`` is imported.
"""

import os
from pathlib import Path

os.environ.setdefault(
    "FLASHINDORANK_CACHE_DIR",
    str(Path(__file__).resolve().parent.parent / ".model_cache"),
)
os.environ.setdefault("FLASHINDORANK_LOG_LEVEL", "ERROR")
