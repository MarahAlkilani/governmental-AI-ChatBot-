from types import SimpleNamespace

import generation


class FakeClient:
    def __init__(self, host, timeout):
        self.host = host
        self.timeout = timeout
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": {"content": "الإجابة المدعومة"}}


def test_no_contexts_returns_clear_arabic_message(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    assert generation.generate_answer("سؤال", []) == generation.NO_CONTEXT_MESSAGE


def test_generation_uses_explicit_model_and_labels_sources(monkeypatch):
    fake_client = FakeClient("unused", 1)
    monkeypatch.setenv("OLLAMA_MODEL", "team-local-model")
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "12")
    monkeypatch.setattr(
        generation, "ollama", SimpleNamespace(Client=lambda **kwargs: fake_client)
    )

    answer = generation.generate_answer(
        "كم الرسوم؟",
        [{"text": "الرسوم 2 دينار", "source": "id_card_renewal.txt", "score": 0.9}],
        [{"question": "أريد تجديد البطاقة", "answer": "حسناً"}],
    )

    assert answer == "الإجابة المدعومة"
    assert fake_client.host == "unused"
    call = fake_client.calls[0]
    assert call["model"] == "team-local-model"
    assert call["options"] == {"temperature": 0}
    assert call["messages"][1] == {"role": "user", "content": "أريد تجديد البطاقة"}
    prompt = call["messages"][-1]["content"]
    assert "id_card_renewal.txt" in prompt
    assert "الرسوم 2 دينار" in prompt


def test_missing_model_never_falls_back(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.setattr(generation, "ollama", SimpleNamespace(Client=lambda **_: None))

    answer = generation.generate_answer(
        "كم الرسوم؟", [{"text": "2 دينار", "source": "service.txt", "score": 1.0}]
    )

    assert "OLLAMA_MODEL" in answer


def test_timeout_returns_clear_message(monkeypatch):
    class TimeoutClient:
        def __init__(self, **kwargs):
            pass

        def chat(self, **kwargs):
            raise TimeoutError("request timeout")

    monkeypatch.setenv("OLLAMA_MODEL", "team-local-model")
    monkeypatch.setattr(generation, "ollama", SimpleNamespace(Client=TimeoutClient))

    answer = generation.generate_answer(
        "كم الرسوم؟", [{"text": "2 دينار", "source": "service.txt", "score": 1.0}]
    )

    assert "انتهت مهلة" in answer


def test_missing_model_error_is_clear(monkeypatch):
    class MissingModelClient:
        def __init__(self, **kwargs):
            pass

        def chat(self, **kwargs):
            error = RuntimeError("model not found")
            error.status_code = 404
            raise error

    monkeypatch.setenv("OLLAMA_MODEL", "missing-model")
    monkeypatch.setattr(generation, "ollama", SimpleNamespace(Client=MissingModelClient))

    answer = generation.generate_answer(
        "كم الرسوم؟", [{"text": "2 دينار", "source": "service.txt", "score": 1.0}]
    )

    assert "missing-model" in answer
    assert "غير متوفر" in answer


def test_connection_error_is_clear(monkeypatch):
    class ConnectionClient:
        def __init__(self, **kwargs):
            pass

        def chat(self, **kwargs):
            raise ConnectionError("connection refused")

    monkeypatch.setenv("OLLAMA_MODEL", "team-local-model")
    monkeypatch.setattr(generation, "ollama", SimpleNamespace(Client=ConnectionClient))

    answer = generation.generate_answer(
        "كم الرسوم؟", [{"text": "2 دينار", "source": "service.txt", "score": 1.0}]
    )

    assert "تعذر الاتصال" in answer
