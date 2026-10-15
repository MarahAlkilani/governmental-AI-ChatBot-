"""Portable path handling tests using synthetic files, never organizer data."""
import json
from pathlib import Path
import pytest
import ingest
import main
import project_paths as paths


def corpus(directory, marker="خدمة تجريبية"):
    directory.mkdir(parents=True, exist_ok=True)
    for name in paths.DOCUMENT_NAMES:
        (directory / name).write_text(marker + "\nالرسوم 2 دينار\n", encoding="utf-8")
    return directory.resolve()


@pytest.mark.parametrize("relative", paths.QUESTION_LOCATIONS)
def test_each_questions_location_uses_its_own_data(tmp_path, relative, monkeypatch):
    monkeypatch.delenv("GOVERNMENT_RAG_DATA_DIR", raising=False)
    question = tmp_path / relative
    directory = corpus(question.parent / "data")
    question.write_text('{"chats": []}', encoding="utf-8")
    assert paths.discover_questions(tmp_path) == question.resolve()
    assert paths.resolve_document_directory(question, base=tmp_path) == directory


def test_configured_directory_wins_and_invalid_configuration_falls_back(tmp_path, monkeypatch):
    q = tmp_path / "pack/questions.json"
    sibling = corpus(q.parent / "data")
    configured = corpus(tmp_path / "configured")
    corpus(tmp_path / "docs")
    monkeypatch.setenv("GOVERNMENT_RAG_DATA_DIR", str(configured))
    assert paths.resolve_document_directory(q, base=tmp_path) == configured
    override = corpus(tmp_path / "override")
    assert paths.resolve_document_directory(q, override, tmp_path) == override
    monkeypatch.setenv("GOVERNMENT_RAG_DATA_DIR", str(tmp_path / "missing"))
    assert paths.resolve_document_directory(q, base=tmp_path) == sibling


def test_docs_fallback_excludes_unrelated_files_and_other_packs(tmp_path, monkeypatch):
    monkeypatch.delenv("GOVERNMENT_RAG_DATA_DIR", raising=False)
    directory = corpus(tmp_path / "docs")
    (directory / "unrelated.txt").write_text("unrelated", encoding="utf-8")
    (directory / "architecture.md").write_text("architecture", encoding="utf-8")
    corpus(tmp_path / "other_pack/data", "مصدر آخر")
    selected = paths.resolve_document_directory(tmp_path / "pack/questions.json", base=tmp_path)
    assert selected == directory
    with paths.use_document_directory(selected):
        documents = ingest.load_documents()
    assert len(documents) == 12
    assert {d.filename for d in documents} == set(paths.DOCUMENT_NAMES)
    assert all(d.text.startswith("خدمة تجريبية") for d in documents)
    assert paths.active_document_directory() is None


def test_missing_corpus_lists_checked_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("GOVERNMENT_RAG_DATA_DIR", str(tmp_path / "invalid"))
    q = tmp_path / "pack/questions.json"
    with pytest.raises(ValueError) as error:
        paths.resolve_document_directory(q, base=tmp_path)
    for directory in [tmp_path / "invalid", q.parent / "data", tmp_path / "docs"]:
        assert str(directory) in str(error.value)


def test_discovery_reports_missing_and_ambiguous_questions(tmp_path):
    with pytest.raises(FileNotFoundError) as error:
        paths.discover_questions(tmp_path)
    assert all(str(tmp_path / relative) in str(error.value) for relative in paths.QUESTION_LOCATIONS)
    for relative in paths.QUESTION_LOCATIONS[:2]:
        q = tmp_path / relative
        q.parent.mkdir(exist_ok=True)
        q.write_text('{}', encoding="utf-8")
    with pytest.raises(ValueError, match="Multiple questions"):
        paths.discover_questions(tmp_path)


def test_relative_input_and_explicit_corpus_print_resolved_paths(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    question = tmp_path / "participant_pack/questions.json"
    selected = corpus(question.parent / "data")
    question.write_text('{"chats": []}', encoding="utf-8")
    reference = question.with_name("answers_example.json")
    reference.write_text('{"reference": true}', encoding="utf-8")
    before = question.read_bytes(), reference.read_bytes()
    assert main.main(["participant_pack/questions.json"]) == 0
    printed = capsys.readouterr().out
    for path in [question, selected, reference, question.with_name("answers.json")]:
        assert str(path.resolve()) in printed
    assert json.loads(question.with_name("answers.json").read_text(encoding="utf-8")) == {"chats": []}
    assert before == (question.read_bytes(), reference.read_bytes())


def test_missing_corpus_fails_before_generation(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GOVERNMENT_RAG_DATA_DIR", raising=False)
    monkeypatch.setattr(paths, "BASE_DIR", tmp_path)
    question = tmp_path / "questions.json"
    question.write_text('{"chats": []}', encoding="utf-8")
    monkeypatch.setattr(main, "generate_answer", lambda *a: pytest.fail("no dataset"))
    assert main.main([str(question)]) == 1
    assert str(tmp_path / "data") in capsys.readouterr().err
    assert not question.with_name("answers.json").exists()


def test_selected_directory_is_restored_after_error(tmp_path):
    with pytest.raises(RuntimeError):
        with paths.use_document_directory(tmp_path):
            raise RuntimeError("failure")
    assert paths.active_document_directory() is None


def test_explicit_questions_select_own_pack_with_multiple_datasets(tmp_path, monkeypatch):
    monkeypatch.delenv("GOVERNMENT_RAG_DATA_DIR", raising=False)
    packs = [tmp_path / "participant_pack", tmp_path / "hackathon_participant_pack/participant_pack"]
    for pack in packs:
        corpus(pack / "data")
        (pack / "questions.json").write_text('{"chats": []}', encoding="utf-8")
    for pack in packs:
        assert paths.resolve_document_directory(pack / "questions.json", base=tmp_path) == (pack / "data").resolve()
    with pytest.raises(ValueError, match="Multiple questions"):
        paths.discover_questions(tmp_path)


@pytest.mark.parametrize("configured", [False, True])
def test_standalone_loader_works_without_questions_when_corpus_exists(tmp_path, monkeypatch, configured):
    monkeypatch.setattr(ingest, "BASE_DIR", tmp_path)
    monkeypatch.delenv("GOVERNMENT_RAG_DATA_DIR", raising=False)
    directory = corpus(tmp_path / ("configured" if configured else "docs"))
    if configured:
        monkeypatch.setenv("GOVERNMENT_RAG_DATA_DIR", str(directory))
    documents = ingest.load_documents()
    assert len(documents) == 12
    assert all(document.source_path.parent == directory for document in documents)
