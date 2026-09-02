"""Tiny config loader: YAML + dotted-path CLI overrides + {polarity} templating.

Kept deliberately dependency-light (just pyyaml). Usage from a script:

    from mft.config import load_config
    cfg = load_config()                      # reads configs/default.yaml + argv
    cfg = load_config("configs/full.yaml")   # explicit base

CLI form:
    python -m mft.sdf.train --set sdf.polarity=unmonitored --set sdf.epochs=8
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


class Config(dict):
    """dict with attribute access, recursively."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Config(val) if isinstance(val, dict) else val

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _coerce(value: str) -> Any:
    """Turn a CLI string into an int/float/bool/None where it obviously is one."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def _set_dotted(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _template(obj: Any, subs: dict[str, str]) -> Any:
    """Recursively .format(**subs) every string, ignoring missing keys."""
    if isinstance(obj, str):
        for k, v in subs.items():
            obj = obj.replace("{" + k + "}", str(v))
        return obj
    if isinstance(obj, dict):
        return {k: _template(v, subs) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_template(v, subs) for v in obj]
    return obj


def load_config(base: str | Path | None = None, argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="path.key=value")
    known, _ = parser.parse_known_args(argv)

    base_path = Path(known.config or base or DEFAULT_CONFIG)
    with open(base_path) as f:
        cfg: dict = yaml.safe_load(f)

    for override in known.set:
        if "=" not in override:
            raise ValueError(f"--set expects path.key=value, got: {override!r}")
        dotted, raw = override.split("=", 1)
        _set_dotted(cfg, dotted.strip(), _coerce(raw.strip()))

    # Resolve {polarity} (and any future slots) throughout the tree.
    polarity = cfg.get("sdf", {}).get("polarity", "monitored")
    cfg = _template(cfg, {"polarity": polarity})

    return Config(cfg)


def resolve_path(p: str | Path) -> Path:
    """Interpret a config path relative to the repo root if not absolute."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p
