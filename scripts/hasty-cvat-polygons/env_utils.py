from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: Path, override: bool = False) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value


def find_env_file(start_dir: Path, filename: str = ".env") -> Path | None:
    start = start_dir.resolve()
    for directory in [start, *start.parents]:
        candidate = directory / filename
        if candidate.exists():
            return candidate
    return None


def load_env_from_parents(start_dir: Path, filename: str = ".env", override: bool = False) -> Path | None:
    env_path = find_env_file(start_dir, filename)
    if env_path is not None:
        load_env_file(env_path, override=override)
    return env_path
