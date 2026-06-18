"""Runtime configuration, read once from environment variables.

Everything here is tuned for low-resource (cheap VPS) deployments: small
defaults, explicit thread control so ONNX Runtime does not oversubscribe a
1-2 vCPU box, and a persistent on-disk model cache so model weights are
downloaded only once.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


def _default_cache_dir() -> str:
    return os.environ.get(
        "FLASHINDORANK_CACHE_DIR",
        str(Path.home() / ".cache" / "flashindorank"),
    )


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_list(name: str) -> List[str]:
    raw = os.environ.get(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass
class Settings:
    """Process-wide settings, resolved from the environment at import time."""

    cache_dir: str = field(default_factory=_default_cache_dir)
    log_level: str = field(default_factory=lambda: os.environ.get("FLASHINDORANK_LOG_LEVEL", "ERROR"))
    max_length: int = field(default_factory=lambda: _env_int("FLASHINDORANK_MAX_LENGTH", 512))

    # ONNX Runtime thread counts. 0 lets ORT decide; on tiny VPS boxes pin to a
    # small number (e.g. 1-2) to avoid context-switch thrash.
    intra_op_threads: int = field(default_factory=lambda: _env_int("FLASHINDORANK_INTRA_OP_THREADS", 0))
    inter_op_threads: int = field(default_factory=lambda: _env_int("FLASHINDORANK_INTER_OP_THREADS", 0))

    # Models to load eagerly on startup so the first request is not slow.
    preload_models: List[str] = field(default_factory=lambda: _env_list("FLASHINDORANK_PRELOAD_MODELS"))

    host: str = field(default_factory=lambda: os.environ.get("FLASHINDORANK_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("FLASHINDORANK_PORT", 8000))


settings = Settings()
