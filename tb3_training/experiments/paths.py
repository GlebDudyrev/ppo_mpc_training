from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_VERSION_RE = re.compile(r"^version_(\d+)$")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def experiments_root(explicit_root: Optional[str] = None) -> Path:
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()
    return repo_root() / "experiments"


@dataclass(frozen=True)
class ExperimentNamespace:
    model_name: str
    env_name: str
    phase: str  # train or eval

    def base_dir(self, root: Optional[str] = None) -> Path:
        return experiments_root(root) / self.model_name / self.env_name / self.phase


def _existing_versions(base_dir: Path) -> list[int]:
    if not base_dir.exists():
        return []
    versions: list[int] = []
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        match = _VERSION_RE.match(child.name)
        if match:
            versions.append(int(match.group(1)))
    return sorted(versions)


def latest_version(base_dir: Path) -> Optional[int]:
    versions = _existing_versions(base_dir)
    return versions[-1] if versions else None


def make_version_dir(namespace: ExperimentNamespace, root: Optional[str] = None, version: Optional[int] = None) -> Path:
    base_dir = namespace.base_dir(root)
    base_dir.mkdir(parents=True, exist_ok=True)

    if version is None:
        current = latest_version(base_dir)
        version = 0 if current is None else current + 1

    run_dir = base_dir / f"version_{version}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def resolve_version_dir(namespace: ExperimentNamespace, root: Optional[str] = None, version: Optional[int] = None) -> Path:
    base_dir = namespace.base_dir(root)
    if version is None:
        version = latest_version(base_dir)
        if version is None:
            raise FileNotFoundError(f"No runs found in {base_dir}")

    run_dir = base_dir / f"version_{version}"
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    return run_dir
