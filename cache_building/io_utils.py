"""Shared I/O helpers for cache construction."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path
from typing import Any, Iterable


def load_json(path: str | Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: str | Path) -> None:
    """Atomic pickle write (tmp + replace) so kills cannot leave a truncated file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
    tmp.replace(path)


def qtoi(qid: str) -> int:
    return int(qid[1:])


def itoq(i: int) -> str:
    return f"Q{i}"


def iter_wikidata_dump(path: str | Path) -> Iterable[dict]:
    """Yield entity dicts from a Wikidata JSON dump (.json.gz, one entity/line)."""
    with gzip.open(path, "rb") as f:
        for raw in f:
            line = raw.decode("utf-8")
            # Official dumps terminate each line with ',\n' except the last ']'.
            line = line[:-2] if line.endswith(",\n") else line.strip().rstrip(",")
            if not line or line in ("[", "]"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
