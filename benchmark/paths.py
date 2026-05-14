from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(".")
BENCHMARK_ROOT = PROJECT_ROOT / "benchmark"
DATA_ROOT = BENCHMARK_ROOT / "data"


def ensure_repo_on_path() -> None:
    return None


def repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path
