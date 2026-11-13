"""
Structured per-iteration logging and checkpoint utilities.

Per-iteration log entries are written as JSON Lines (.jsonl) for easy
post-processing (`pandas.read_json('mhmc.jsonl', lines=True)`).
Checkpoints save the full sampler state via pickle for resumability.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


class JsonlLogger:
    """Append-only structured per-iteration logger."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate at start of run
        self.path.write_text("")

    def log(self, record: dict[str, Any]) -> None:
        """Append one JSON record (one iteration)."""
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=_json_default) + "\n")


def _json_default(o):
    """Make numpy scalars/arrays JSON-serializable."""
    import numpy as np
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


# ------------------------------------------------------------
#  Checkpoint
# ------------------------------------------------------------

def save_checkpoint(state: Any, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(state, f)


def load_checkpoint(path: str) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)