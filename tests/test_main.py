"""Offline CLI and orchestration tests; no running Ollama server needed."""

from copy import deepcopy
import json

import pytest

import main
from generation import NO_CONTEXT_MESSAGE


def questions(*chats):
    return {"chats": [
        {"chat_id": f"chat-{index}", "messages": [
            {"message_id": f"{index}-{turn}", "question": question}
            for turn, question in enumerate(chat)
        ]} for index, chat in enumerate(chats)
    ]}


def test_preserves_structure_exact_contexts_and_isolates_history(monkeypatch):
    data = questions(["  كم الرسوم؟  ", "وما الوثائق؟"], ["كم المدة؟"])
    data["extra"] = {"label": "يبقى كما هو"}
    data["chats"][0]["extra"] = 7
    data["chats"][0]["messages"][0]["extra"] = [1, 2]
    before = deepcopy(data)
    evidence = [
        {"text": "  الرسوم 2 دينار  ", "source": "a.txt", "score": 1},
        {"text": "  الرسوم 2 دينار  ", "source": "duplicate.txt", "score": .9},
        {"text": "نص لم يستخدم", "source": "b.txt", "score": .8},
    ]
    retrieval_calls, generation_calls = [], []

    def retrieve(question, top_k):
        retrieval_calls.append((question, top_k))
        return evidence

    def generate(question, contexts, history):
        generation_calls.append((question, deepcopy(contexts), deepcopy(history)))
        history.clear()  # The callee cannot erase orchestration state.
        return evidence[0]["text"]

    monkeypatch.setattr(main, "retrieve", retrieve)
    monkeypatch.setattr(main, "generate_answer", generate)
    output = main.answer_questions(data)
    expected = deepcopy(data)
    for chat in expected["chats"]:
        for message in chat["messages"]:
            message.update(answer=evidence[0]["text"], contexts=[evidence[0]["text"]])
    assert output == expected
    assert data == before
    assert retrieval_calls == [(q, 3) for chat in data["chats"]
                               for q in [m["question"] for m in chat["messages"]]]
    assert generation_calls[0][2] == []
    assert generation_calls[1][2] == [
        {key: expected["chats"][0]["messages"][0][key]
         for key in ("message_id", "question", "answer", "contexts")}
    ]
    assert generation_calls[2][2] == []
    assert all(call[1] == evidence for call in generation_calls)


def test_multisource_subexcerpts_map_to_exact_retrieved_text(monkeypatch):
    evidence = [{"text": "الوثائق\nالبطاقة الشخصية\n\nدفتر العائلة"},
                {"text": "الرسوم\nالرسوم 2 دينار"}, {"text": "نص غير مستخدم"}]
    monkeypatch.setattr(main, "retrieve", lambda *a, **k: evidence)
    answer = "البطاقة الشخصية\n\nدفتر العائلة\n\nالرسوم 2 دينار"
    monkeypatch.setattr(main, "generate_answer", lambda *a: answer)
    message = main.answer_questions(questions(["ما الوثائق والرسوم؟"]))["chats"][0]["messages"][0]
    assert message["answer"] == answer
    assert message["contexts"] == [evidence[0]["text"], evidence[1]["text"]]


@pytest.mark.parametrize("evidence", [[], [{"text": "نص غير كاف"}]])
def test_abstention_has_no_contexts(monkeypatch, evidence):
    monkeypatch.setattr(main, "retrieve", lambda *a, **k: evidence)
    calls = []

    def generate(question, contexts, history):
        calls.append((question, contexts, history))
        return NO_CONTEXT_MESSAGE

    monkeypatch.setattr(main, "generate_answer", generate)
    message = main.answer_questions(questions(["سؤال"]))["chats"][0]["messages"][0]
    assert message["answer"] == NO_CONTEXT_MESSAGE
    assert message["contexts"] == []
    assert calls == [("سؤال", evidence, [])]


@pytest.mark.parametrize("answer", ["الرسوم 99 دينار", "English answer", "الرسوم 2 دينار"])
def test_untraceable_or_partial_line_answers_abstain(monkeypatch, answer):
    monkeypatch.setattr(main, "retrieve", lambda *a, **k: [{"text": "ليست الرسوم 2 دينار"}])
    monkeypatch.setattr(main, "generate_answer", lambda *a: answer)
    message = main.answer_questions(questions(["كم الرسوم؟"]))["chats"][0]["messages"][0]
    assert message["answer"] == NO_CONTEXT_MESSAGE
    assert message["contexts"] == []


@pytest.mark.parametrize("failure", ["retrieval", "generation", "empty", "nonstring", "model_error"])
def test_failed_turn_does_not_stop_later_messages(monkeypatch, failure):
    data = questions(["الأول", "الثاني"], ["الثالث"])
    evidence = [{"text": "الرسوم 2 دينار"}]

    def retrieve(question, top_k):
        if question == "الأول" and failure == "retrieval":
            raise OSError("missing data")
        return evidence

    histories = []

    def generate(question, contexts, history):
        histories.append(deepcopy(history))
        if question == "الأول":
            if failure == "generation":
                raise RuntimeError("model failed")
            return {"empty": "", "nonstring": None,
                    "model_error": "تعذر الاتصال بخادم Ollama المحلي"}[failure]
        return evidence[0]["text"]

    monkeypatch.setattr(main, "retrieve", retrieve)
    monkeypatch.setattr(main, "generate_answer", generate)
    output = main.answer_questions(data)
    first, second = output["chats"][0]["messages"]
    assert first["answer"] == main.ERROR_MESSAGE
    assert first["contexts"] == []
    assert second["answer"] == evidence[0]["text"]
    assert histories[-2][0]["answer"] == main.ERROR_MESSAGE
    assert histories[-1] == []


def test_cli_writes_utf8_beside_input_and_preserves_input(tmp_path, monkeypatch):
    folder = tmp_path / "inputs"
    folder.mkdir()
    input_path = folder / "questions.json"
    data = questions(["كم الرسوم؟"])
    original = json.dumps(data, ensure_ascii=False)
    input_path.write_text(original, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "retrieve", lambda *a, **k: [{"text": "الرسوم 2 دينار"}])
    monkeypatch.setattr(main, "generate_answer", lambda *a: "الرسوم 2 دينار")
    assert main.main([str(input_path)]) == 0
    raw = (folder / "answers.json").read_text(encoding="utf-8")
    result = json.loads(raw)
    assert "الرسوم" in raw
    assert result["chats"][0]["messages"][0]["contexts"] == ["الرسوم 2 دينار"]
    assert input_path.read_text(encoding="utf-8") == original
    assert not (tmp_path / "answers.json").exists()


@pytest.mark.parametrize("data", [None, [], {}, {"chats": {}}, {"chats": [None]},
    {"chats": [{"messages": []}]}, {"chats": [{"chat_id": "a", "messages": {}}]},
    {"chats": [{"chat_id": "a", "messages": [None]}]},
    {"chats": [{"chat_id": "a", "messages": [{"question": "سؤال"}]}]},
    {"chats": [{"chat_id": "a", "messages": [{"message_id": "m", "question": 3}]}]},
])
def test_malformed_schema_is_rejected_without_processing(tmp_path, monkeypatch, data, capsys):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(main, "retrieve", lambda *a, **k: pytest.fail("must validate first"))
    assert main.main([str(path)]) == 1
    assert not (tmp_path / "answers.json").exists()
    assert capsys.readouterr().err


@pytest.mark.parametrize("contents", [None, b'{bad json', b'\xff'])
def test_missing_file_invalid_json_and_encoding(tmp_path, contents, capsys):
    path = tmp_path / "questions.json"
    if contents is not None:
        path.write_bytes(contents)
    assert main.main([str(path)]) == 1
    assert not (tmp_path / "answers.json").exists()
    assert capsys.readouterr().err


def test_output_write_failure_is_reported(tmp_path, capsys):
    path = tmp_path / "questions.json"
    path.write_text('{"chats": []}', encoding="utf-8")
    (tmp_path / "answers.json").mkdir()
    assert main.main([str(path)]) == 1
    assert capsys.readouterr().err


def test_empty_chats_and_messages_are_valid():
    assert main.answer_questions({"chats": []}) == {"chats": []}
    assert main.answer_questions(questions([])) == questions([])


def test_cli_requires_input_argument():
    with pytest.raises(SystemExit) as exc:
        main.main([])
    assert exc.value.code == 2


def test_real_retrieval_and_generation_pipeline_with_mocked_ollama(tmp_path, monkeypatch):
    import generation
    from types import SimpleNamespace

    question = "كم يستغرق تجديد دفتر العائلة؟"
    evidence = main.retrieve(question, top_k=3)
    assert evidence
    calls = []

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["host"] == "http://localhost:11434"

        def chat(self, **kwargs):
            calls.append(kwargs)
            payload = json.loads(kwargs["messages"][-1]["content"])
            excerpt = payload["evidence"][0]
            return {"message": {"content": json.dumps({
                "status": "supported", "excerpts": [excerpt]
            }, ensure_ascii=False)}}

    monkeypatch.setenv("OLLAMA_MODEL", "offline-test-model")
    monkeypatch.setattr(generation, "ollama", SimpleNamespace(Client=Client))
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(questions([question]), ensure_ascii=False), encoding="utf-8")
    assert main.main([str(path)]) == 0
    result = json.loads((tmp_path / "answers.json").read_text(encoding="utf-8"))
    message = result["chats"][0]["messages"][0]
    assert message["answer"] == evidence[0]["text"]
    assert message["contexts"] == [evidence[0]["text"]]
    assert calls[0]["model"] == "offline-test-model"


def test_script_entrypoint_runs_offline(tmp_path):
    from pathlib import Path
    import subprocess
    import sys

    path = tmp_path / "questions.json"
    path.write_text('{"chats": []}', encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(Path(main.__file__).resolve()), str(path)],
        cwd=tmp_path, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "answers.json").read_text(encoding="utf-8")) == {"chats": []}


@pytest.mark.parametrize("chat_index", [0, 3])
def test_official_questions_real_retrieval_preserves_verbatim_contexts(
    tmp_path, monkeypatch, chat_index
):
    from pathlib import Path
    from retrieval import retrieve

    official = Path(main.__file__).parent / "hackathon_participant_pack/participant_pack/questions.json"
    data = json.loads(official.read_text(encoding="utf-8"))
    data = {"chats": [data["chats"][chat_index]]}
    original = deepcopy(data)
    question = data["chats"][0]["messages"][0]["question"]
    evidence = retrieve(question, top_k=3)
    assert evidence
    if chat_index == 3:
        assert len(evidence) == 2
        assert all(context["text"].endswith("\n") for context in evidence)
    calls = []

    def generate(question, contexts, history):
        calls.append((question, deepcopy(contexts), deepcopy(history)))
        # P06 joins exact excerpts using two newlines. Real source excerpts
        # already end in a newline, yielding three newlines between excerpts.
        return "\n\n".join(context["text"] for context in contexts)

    monkeypatch.setattr(main, "generate_answer", generate)
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert main.main([str(path)]) == 0
    output = json.loads((tmp_path / "answers.json").read_text(encoding="utf-8"))
    message = output["chats"][0]["messages"][0]
    assert calls == [(question, evidence, [])]
    assert message["answer"] == "\n\n".join(c["text"] for c in evidence)
    assert message["contexts"] == [c["text"] for c in evidence]
    assert message["question"] == question
    assert message["message_id"] == original["chats"][0]["messages"][0]["message_id"]
    assert output["chats"][0]["chat_id"] == original["chats"][0]["chat_id"]
    assert json.loads(path.read_text(encoding="utf-8")) == original


def test_model_configuration_failure_reports_reason_with_real_retrieval(monkeypatch, capsys):
    question = "كم يستغرق تجديد دفتر العائلة؟"
    assert main.retrieve(question, top_k=3)
    monkeypatch.setattr(main, "generate_answer", lambda *a: (
        "تعذر إعداد Ollama المحلي: OLLAMA_MODEL is required."
    ))
    message = main.answer_questions(questions([question]))["chats"][0]["messages"][0]
    assert message["answer"] == main.ERROR_MESSAGE
    assert message["contexts"] == []
    assert "OLLAMA_MODEL" in capsys.readouterr().err


def test_abstention_with_real_retrieval_does_not_emit_unused_evidence(monkeypatch):
    question = "كم يستغرق تجديد دفتر العائلة؟"
    assert main.retrieve(question, top_k=3)
    monkeypatch.setattr(main, "generate_answer", lambda *a: NO_CONTEXT_MESSAGE)
    message = main.answer_questions(questions([question]))["chats"][0]["messages"][0]
    assert message["answer"] == NO_CONTEXT_MESSAGE
    assert message["contexts"] == []


def test_real_retrieval_retains_service_identity_in_generation_payload(monkeypatch):
    """A section such as '45 minutes' needs its service filename for routing.

    Captured real Ollama behavior: without source identity the model abstained;
    with family_book_renewal.txt it returned the exact complete source line.
    """
    import generation
    from types import SimpleNamespace

    question = "كم يستغرق تجديد دفتر العائلة؟"
    evidence = main.retrieve(question, top_k=3)
    assert len(evidence) == 1
    assert evidence[0]["source"] == "family_book_renewal.txt"
    calls = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def chat(self, **kwargs):
            calls.append(kwargs)
            payload = json.loads(kwargs["messages"][-1]["content"])
            item = payload["evidence"][0]
            result = {"status": "abstain", "excerpts": []}
            if item.get("source") == "family_book_renewal.txt":
                result = {"status": "supported", "excerpts": [{
                    "evidence_id": item["evidence_id"],
                    "text": evidence[0]["text"].rstrip("\n"),
                }]}
            return {"message": {"content": json.dumps(result, ensure_ascii=False)}}

    monkeypatch.setenv("OLLAMA_MODEL", "command-r7b-arabic:7b")
    monkeypatch.setattr(generation, "ollama", SimpleNamespace(Client=Client))
    message = main.answer_questions(questions([question]))["chats"][0]["messages"][0]
    assert message["answer"] == evidence[0]["text"].rstrip("\n")
    assert message["contexts"] == [evidence[0]["text"]]
    assert calls[0]["model"] == "command-r7b-arabic:7b"
