"""Developer environment utilities for tests and CLI fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import shutil
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_REQUIRED_DEV_ASSETS: tuple[tuple[Path, Path]] = (
    (Path("tools/check_eval_gate.py"), PROJECT_ROOT / "tools" / "check_eval_gate.py"),
    (Path("tools/eval_offline.py"), PROJECT_ROOT / "tools" / "eval_offline.py"),
    (Path("scripts/fetch_models.py"), PROJECT_ROOT / "scripts" / "fetch_models.py"),
    (
        Path("scripts/rl_staleness_watchdog.py"),
        PROJECT_ROOT / "scripts" / "rl_staleness_watchdog.py",
    ),
    (Path("Makefile.models"), PROJECT_ROOT / "Makefile.models"),
    (
        Path("src/rl/influence_controller.py"),
        PROJECT_ROOT / "src" / "rl" / "influence_controller.py",
    ),
)


def _write_file(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        dst.write_bytes(src.read_bytes())
        try:
            dst.chmod(src.stat().st_mode)
        except OSError:
            pass
    elif src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def ensure_dev_cli_scripts(
    cwd: Path | None = None,
    *,
    assets: Iterable[tuple[Path, Path]] = _REQUIRED_DEV_ASSETS,
) -> None:
    """Ensure CLI helper scripts exist in the current working directory.

    Tests may chdir into temporary directories and expect project CLI scripts to
    remain available via relative paths. This helper mirrors the canonical
    project versions into the provided ``cwd`` when missing or when a placeholder
    shim exists.
    """
    target_root = cwd or Path.cwd()

    for relative_path, source_path in assets:
        dest_path = target_root / relative_path

        if not source_path.exists():
            continue

        needs_copy = True
        if dest_path.exists():
            try:
                current_bytes = dest_path.read_bytes()
                source_bytes = source_path.read_bytes()
                needs_copy = current_bytes != source_bytes
            except Exception:
                needs_copy = True

        if needs_copy:
            _write_file(dest_path, source_path)


__all__ = ["ensure_dev_cli_scripts"]
