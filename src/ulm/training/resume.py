"""checkpoint 탐색과 output overwrite 보호."""

from __future__ import annotations

import re
from pathlib import Path

_CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    root = Path(output_dir)
    if not root.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = _CHECKPOINT_PATTERN.match(child.name)
        if match:
            candidates.append((int(match.group(1)), child))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def resolve_resume_checkpoint(
    resume_from_checkpoint: bool | str | None, output_dir: str | Path
) -> Path | None:
    if resume_from_checkpoint in (None, False):
        return None
    if resume_from_checkpoint is True:
        latest = find_latest_checkpoint(output_dir)
        if latest is None:
            raise FileNotFoundError(f"resume할 checkpoint가 없습니다: {output_dir}")
        return latest
    checkpoint = Path(resume_from_checkpoint)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory가 없습니다: {checkpoint}")
    return checkpoint


def ensure_output_dir(output_dir: str | Path, resume_checkpoint: Path | None) -> Path:
    """새 실행은 비어 있지 않은 output을 덮어쓰지 않도록 한다."""

    destination = Path(output_dir)
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(f"output_dir가 directory가 아닙니다: {destination}")
    if destination.exists() and resume_checkpoint is None:
        contents = [p for p in destination.iterdir() if p.name != "config.resolved.json"]
        if contents:
            raise FileExistsError(
                f"기존 output을 덮어쓰지 않습니다: {destination}; "
                "resume_from_checkpoint를 지정하세요"
            )
    destination.mkdir(parents=True, exist_ok=True)
    return destination
