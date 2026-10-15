"""Load Arabic government-service documents and their retrieval metadata."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union


logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
_NAVIGATION_LINES = {"الرئيسية", "الخدمات", "دليل الخدمات", "شارك"}
_SECTION_PATTERNS = (
    re.compile(r"^(?P<title>الفئة المستفيدة)(?=\s|$)"),
    re.compile(r"^(?P<title>أماكن تقديم الخدمة|اماكن تقديم الخدمة)(?=\s|$)"),
    re.compile(r"^(?P<title>شروط(?:\s+تقديم\s+الخدمة)?)(?=\s|$)"),
    re.compile(
        r"^(?P<title>الوثائق المطلوبة(?:\s+للحصول\s+على\s+الخدمة)?)(?=\s|$)"
    ),
    re.compile(r"^(?P<title>إجراءات(?:\s+تقديم\s+الخدمة)?)(?=\s|$)"),
    re.compile(r"^(?P<title>الزمن المستغرق(?:\s+لتقديم\s+الخدمة)?)(?=\s|$)"),
    re.compile(r"^(?P<title>شركاء(?:\s+الخدمة)?)(?=\s|$)"),
    re.compile(r"^(?P<title>ملاحظات)(?=\s|$)"),
    re.compile(r"^(?P<title>الرسوم)(?=\s|$)"),
)


@dataclass(frozen=True)
class DocumentSection:
    title: str
    text: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class DocumentRecord:
    filename: str
    service_id: str
    service_name: Optional[str]
    text: str
    sections: Tuple[DocumentSection, ...]
    source_path: Path


def _find_text_files(root: Path) -> List[Path]:
    try:
        return sorted(
            (
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() == ".txt"
            ),
            key=lambda path: str(path).casefold(),
        )
    except OSError as error:
        logger.warning("Unable to discover documents under %s: %s", root, error)
        return []


def _default_data_dir() -> Path:
    participant_pack = BASE_DIR / "hackathon_participant_pack"
    if _find_text_files(participant_pack):
        return participant_pack
    return BASE_DIR / "data"


def _service_name(text: str) -> Optional[str]:
    for line in text.splitlines():
        title = line.strip().lstrip("\ufeff")
        if title and title not in _NAVIGATION_LINES:
            return title
    return None


def _section_title(line: str) -> Optional[str]:
    candidate = line.lstrip()
    for pattern in _SECTION_PATTERNS:
        match = pattern.match(candidate)
        if match:
            return match.group("title")
    return None


def _extract_sections(text: str) -> Tuple[DocumentSection, ...]:
    sections: List[DocumentSection] = []
    current_title: Optional[str] = None
    start_offset = 0
    start_line = 0
    offset = 0

    for line_number, line in enumerate(text.splitlines(keepends=True), start=1):
        title = _section_title(line)
        if title is not None:
            if current_title is not None:
                sections.append(
                    DocumentSection(
                        title=current_title,
                        text=text[start_offset:offset],
                        start_line=start_line,
                        end_line=line_number - 1,
                    )
                )
            current_title = title
            start_offset = offset
            start_line = line_number
        offset += len(line)

    if current_title is not None:
        sections.append(
            DocumentSection(
                title=current_title,
                text=text[start_offset:],
                start_line=start_line,
                end_line=len(text.splitlines()),
            )
        )
    return tuple(sections)


def load_documents(
    data_dir: Optional[Union[str, Path]] = None,
) -> List[DocumentRecord]:
    """Discover and load UTF-8 TXT documents, retaining their source text verbatim."""
    root = Path(data_dir) if data_dir is not None else _default_data_dir()
    if not root.is_dir():
        logger.warning("Document directory does not exist or is not a directory: %s", root)
        return []

    documents: List[DocumentRecord] = []
    for path in _find_text_files(root):
        try:
            with path.open("r", encoding="utf-8", newline="") as source:
                text = source.read()
        except (OSError, UnicodeError) as error:
            logger.warning("Skipping unreadable document %s: %s", path, error)
            continue

        documents.append(
            DocumentRecord(
                filename=path.name,
                service_id=path.stem,
                service_name=_service_name(text),
                text=text,
                sections=_extract_sections(text),
                source_path=path,
            )
        )
    return documents
