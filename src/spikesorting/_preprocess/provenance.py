"""What produced a preprocessed binary, and whether it still answers the session.

A preprocessed binary is the size of the recording, so it is made once and
reused -- but only while it is what the session would make now. The record
written beside it (``preprocess.json``) is what decides that: which engine ran,
with which settings, on which source file (path, size and modification time).
Any difference makes the file stale and it is made again; nothing is reused on
the strength of its size alone.

Pure apart from :func:`read` and :func:`write`, which move one small JSON file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["SIDECAR", "normalise", "source_fingerprint", "stale_reasons", "read", "write"]

#: The record's filename, beside the binary it describes.
SIDECAR = "preprocess.json"

#: What has to match for a binary to be reused. ``extractors`` is deliberately
#: not here: it decides only whether the sync edges from the same pass can be
#: reused, which ``_sync.extract`` checks for itself.
_DECIDING = ("engine", "settings", "source")


def normalise(value: Any) -> Any:
    """``value`` as it reads back from JSON, so a tuple and a list compare equal."""
    return json.loads(json.dumps(value, default=str))


def source_fingerprint(path: str | Path) -> dict[str, Any]:
    """Enough of the source file to tell if it changed: path, size, mtime."""
    stat = Path(path).stat()
    return {"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def stale_reasons(recorded: dict[str, Any] | None, expected: dict[str, Any]) -> list[str]:
    """Why a recorded binary does not answer ``expected``; empty if it does."""
    if not recorded:
        return ["no preprocessed binary has been made yet"]
    reasons = []
    for key in _DECIDING:
        was, now = normalise(recorded.get(key)), normalise(expected.get(key))
        if was != now:
            reasons.append(f"{key} changed: was {was}, now {now}")
    return reasons


def read(directory: str | Path) -> dict[str, Any] | None:
    """The record in ``directory``, or None when there is none (or it is torn)."""
    path = Path(directory) / SIDECAR
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def write(directory: str | Path, record: dict[str, Any]) -> Path:
    """Write the record. Called only after the binary is complete."""
    path = Path(directory) / SIDECAR
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(normalise(record), handle, indent=2)
    return path


def discard(directory: str | Path) -> None:
    """Remove the record before rewriting the binary it describes.

    So an interrupted rewrite leaves no record, and a half-written binary is
    never mistaken for a finished one on the next run.
    """
    path = Path(directory) / SIDECAR
    if path.exists():
        path.unlink()
