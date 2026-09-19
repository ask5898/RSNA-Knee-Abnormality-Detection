# -*- coding: utf-8 -*-
"""Where the data is.

One resolver instead of a `DATA_PATH` constant copied into every notebook and
script: the same code then runs on Kaggle, on a local checkout, and in CI
without an edit.
"""
from __future__ import annotations

import os

__all__ = ["CANDIDATE_DATA_PATHS", "resolve_data_path"]

# Searched in order. The Kaggle mounts come first because that is where the
# competition data actually lives; `data/` is the local convention.
CANDIDATE_DATA_PATHS: tuple[str, ...] = (
    "/kaggle/input/competitions/rsna-knee-abnormality-detection",
    "/kaggle/input/rsna-knee-abnormality-detection",
    "data",
)


def resolve_data_path(path: str | None = None) -> str:
    """Explicit argument, then $RSNA_DATA_PATH, then the first mount that exists.

    Raises rather than returning a path that is not there: a missing mount
    surfaces here, with the list of what was tried, instead of as an empty
    scan three minutes later.
    """
    if path:
        return path
    env = os.environ.get("RSNA_DATA_PATH")
    if env:
        return env
    for candidate in CANDIDATE_DATA_PATHS:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        "No data directory found. Pass data_path=, set $RSNA_DATA_PATH, or "
        f"create one of: {', '.join(CANDIDATE_DATA_PATHS)}"
    )
