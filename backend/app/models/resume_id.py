"""Resume identifiers are storage keys, never caller-supplied file paths."""

from pathlib import Path, PureWindowsPath
from typing import Annotated

from pydantic import AfterValidator


class InvalidResumeId(ValueError):
    pass


def validate_resume_id(value: str) -> str:
    # Keep existing filename stems (including spaces/Unicode) compatible, but
    # reject path syntax on both Unix and Windows before any storage access.
    if (
        not value.strip()
        or value in {".", ".."}
        or any(character in value for character in ("/", "\\", ":"))
        or any(ord(character) < 32 for character in value)
        or PureWindowsPath(value).is_reserved()
    ):
        raise InvalidResumeId("Invalid resume identifier")
    return value


ResumeId = Annotated[str, AfterValidator(validate_resume_id)]


def resume_file_path(directory: str | Path, resume_id: str, suffix: str) -> Path:
    """Check containment including existing symlinks, without modifying storage."""
    validate_resume_id(resume_id)
    root = Path(directory).resolve()
    candidate = root / f"{resume_id}{suffix}"
    try:
        contained = candidate.resolve().is_relative_to(root)
    except (OSError, RuntimeError) as exc:
        raise InvalidResumeId("Invalid resume identifier") from exc
    if not contained:
        raise InvalidResumeId("Invalid resume identifier")
    return candidate
