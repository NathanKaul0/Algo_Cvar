"""Data-file resolution, so the scripts run from anywhere in the repo.

Every script here reads and writes its data through data_path(). The search
order covers the layouts this folder gets used in: run from code/, run from the
repo root, or run with everything in one directory.
"""

from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CANDIDATES = (
    os.path.join(ROOT, "data"),
    HERE,
    os.path.join(HERE, "data"),
    os.getcwd(),
    os.path.join(os.getcwd(), "data"),
)


def data_path(name: str) -> str:
    """Absolute path for a data file. Existing file wins; else the default."""
    for d in CANDIDATES:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return os.path.join(CANDIDATES[0], name)


def fig_path(name: str) -> str:
    d = os.path.join(ROOT, "figs")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)
