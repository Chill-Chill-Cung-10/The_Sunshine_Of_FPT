from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
_DATASETS = {
    "val": _DATA_ROOT / "val.json",
    "test": _DATA_ROOT / "test.json",
}
_CACHE: Dict[str, List[Dict[str, Any]]] = {split: [] for split in _DATASETS}


def _load_split(split: str) -> List[Dict[str, Any]]:
    """Load and cache a split (val or test)."""
    if split not in _DATASETS:
        raise KeyError(f"Unknown split: {split!r}. Expected one of {list(_DATASETS)}")

    cache = _CACHE[split]
    if cache:
        return cache

    path = _DATASETS[split]
    with path.open("r", encoding="utf-8") as f:
        cache.extend(json.load(f))
    return cache

def _format_item(item: Dict[str, Any]) -> str:
    """Convert one question record into a single message string."""
    question_text = str(item.get("question", "")).strip()
    choices = item.get("choices", [])

    parts: List[str] = [question_text]
    for idx, choice in enumerate(choices):
        letter = chr(ord("A") + idx)
        parts.append(f"{letter}. {choice}")

    return "\n".join(parts)

def get_message_by_qid(qid: str, split: str = "val") -> str:
    """Return the formatted message for the given qid in the chosen split.

    Raises:
        KeyError: if no record with the requested qid exists or split is unknown.
    """
    for item in _load_split(split):
        if item.get("qid") == qid:
            return _format_item(item)

    raise KeyError(f"No entry found for qid={qid!r} in split {split!r}")


def get_message_by_index(index: int, split: str = "val") -> str:
    """Return the formatted message at the given list index for a split.

    Useful when you just want the nth example (0-based index).
    """
    data = _load_split(split)
    if index < 0 or index >= len(data):
        raise IndexError(f"Index {index} out of range for {split} set of size {len(data)}")

    return _format_item(data[index])
