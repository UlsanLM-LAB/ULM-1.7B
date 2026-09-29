"""Validation shared by CLI entry points that accept local adapter directories."""

from pathlib import Path


def validate_adapter_path(adapter_path: str | Path | None) -> None:
    if adapter_path is not None and (
        not str(adapter_path).strip() or not Path(adapter_path).is_dir()
    ):
        raise ValueError(f"Adapter directory does not exist: {adapter_path}")
