"""Optional official-pack fixtures and standalone synthetic corpus fixtures."""
from pathlib import Path
import pytest
from project_paths import (DOCUMENT_NAMES, discover_questions,
                           resolve_document_directory, use_document_directory)


@pytest.fixture
def cli_dataset(tmp_path):
    directory = tmp_path / "synthetic_data"
    directory.mkdir()
    for name in DOCUMENT_NAMES:
        (directory / name).write_text("خدمة تجريبية\nالرسوم 2 دينار\n", encoding="utf-8")
    return directory


@pytest.fixture
def official_corpus():
    try:
        questions = discover_questions()
        directory = resolve_document_directory(questions)
    except (OSError, ValueError) as exc:
        pytest.skip(str(exc))
    with use_document_directory(directory):
        yield questions, directory
