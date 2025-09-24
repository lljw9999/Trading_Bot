"""Test configuration helpers for developer conveniences."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.utils.dev_env import ensure_dev_cli_scripts


@pytest.fixture(autouse=True)
def _ensure_cli_scripts(monkeypatch):
    """Mirror CLI helper scripts into the active working directory for tests."""

    ensure_dev_cli_scripts(Path.cwd())
    original_chdir = os.chdir

    def _mirror_current_dir(path: str | os.PathLike[str]) -> None:
        original_chdir(path)
        ensure_dev_cli_scripts(Path.cwd())

    monkeypatch.setattr(os, "chdir", _mirror_current_dir)


@pytest.fixture(autouse=True)
def _post_test_gc():
    """Run garbage collection after each test to limit cross-test memory bleed."""
    yield
    import gc

    gc.collect()
