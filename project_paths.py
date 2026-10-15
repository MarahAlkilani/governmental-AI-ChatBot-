"""Resolve one official corpus without combining participant packs."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOCUMENT_NAMES = tuple(
    f"{service}_{variant}.txt"
    for service in ("passport", "family_book", "id_card")
    for variant in ("first_issue", "renewal", "lost", "damaged")
)
QUESTION_LOCATIONS = ("questions.json", "participant_pack/questions.json",
                      "hackathon_participant_pack/participant_pack/questions.json")
_active_directory: ContextVar[Path | None] = ContextVar("document_directory", default=None)


def discover_questions(base: Path | None = None) -> Path:
    """Discover a single pack for evaluation; ambiguity requires an explicit path."""
    base = BASE_DIR if base is None else base
    candidates = [(base / relative).resolve() for relative in QUESTION_LOCATIONS]
    found = [path for path in candidates if path.is_file()]
    if len(found) == 1:
        return found[0]
    if found:
        raise ValueError("Multiple questions files; supply an explicit path: " + ", ".join(map(str, found)))
    raise FileNotFoundError("Questions file missing. Checked: " + ", ".join(map(str, candidates)))


def resolve_document_directory(questions: Path | None = None,
                               configured: Path | None = None,
                               base: Path | None = None) -> Path:
    base = BASE_DIR if base is None else base
    explicit = configured or os.getenv("GOVERNMENT_RAG_DATA_DIR")
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())
    if questions is not None:
        candidates.append(questions.resolve().parent / "data")
    candidates.append((base / "docs").resolve())
    checked = []
    for directory in dict.fromkeys(candidates):
        missing = [name for name in DOCUMENT_NAMES if not (directory / name).is_file()]
        if missing:
            checked.append(f"{directory} (missing: {', '.join(missing)})")
            continue
        try:
            for name in DOCUMENT_NAMES:
                if not (directory / name).read_text(encoding="utf-8").strip():
                    raise ValueError(f"Empty document: {name}")
        except (OSError, UnicodeError, ValueError) as exc:
            checked.append(f"{directory} ({exc})")
            continue
        return directory
    raise ValueError("No valid official document dataset. Checked: " + "; ".join(checked))


@contextmanager
def use_document_directory(directory: Path):
    """Scope corpus selection to a run; restore it even after failures."""
    token = _active_directory.set(directory)
    try:
        yield
    finally:
        _active_directory.reset(token)


def active_document_directory() -> Path | None:
    return _active_directory.get()
