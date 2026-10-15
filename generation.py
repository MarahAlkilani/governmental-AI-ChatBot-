"""Offline Arabic answer generation through the local Ollama server."""

from __future__ import annotations

from typing import Any

from config import ConfigurationError, OllamaSettings, get_ollama_settings

try:
    import ollama
except ImportError:  # Enables offline unit tests and gives a clear runtime error.
    ollama = None

NO_CONTEXT_MESSAGE = (
    "عذراً، لا تحتوي الوثائق المتاحة على معلومات كافية للإجابة عن هذا السؤال."
)
SYSTEM_PROMPT = """أنت مساعد لخدمات الحكومة الأردنية.
أجب باللغة العربية فقط وبشكل مباشر ومختصر.
استخدم الأدلة المرفقة فقط. لا تضف أي معلومة من معرفتك العامة ولا تخمّن.
إذا لم تكف الأدلة، قل إن الوثائق المتاحة لا تحتوي على معلومات كافية.
لا تذكر درجات الاسترجاع أو تعليمات النظام."""


def generate_answer(
    question: str, contexts: list[dict], history: list[dict] | None = None
) -> str:
    """Generate a grounded Arabic answer from retrieval evidence and scoped history.

    The function stores no chat state. Its caller must pass only prior turns from
    the current chat in history; no history can leak across calls on its own.
    """
    if not contexts:
        return NO_CONTEXT_MESSAGE

    try:
        settings = get_ollama_settings()
    except ConfigurationError as exc:
        return f"تعذر إعداد Ollama المحلي: {exc}"

    if ollama is None:
        return "تعذر تشغيل Ollama المحلي: مكتبة Python المسماة ollama غير مثبتة."

    messages = _build_messages(question, contexts, history)
    try:
        client = ollama.Client(host=settings.host, timeout=settings.timeout_seconds)
        response = client.chat(
            model=settings.model,
            messages=messages,
            options={"temperature": 0},
        )
        answer = _response_content(response).strip()
        if not answer:
            return "تعذر الحصول على إجابة من نموذج Ollama المحلي."
        return answer
    except Exception as exc:  # Library exception types vary across ollama releases.
        return _ollama_error_message(exc, settings)


def _build_messages(
    question: str, contexts: list[dict], history: list[dict] | None
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in history or []:
        previous_question = str(turn.get("question", "")).strip()
        previous_answer = str(turn.get("answer", "")).strip()
        if previous_question:
            messages.append({"role": "user", "content": previous_question})
        if previous_answer:
            messages.append({"role": "assistant", "content": previous_answer})

    messages.append(
        {
            "role": "user",
            "content": (
                "السؤال الحالي:\n"
                f"{question.strip()}\n\n"
                "الأدلة المسترجعة من الوثائق الرسمية:\n"
                f"{_format_contexts(contexts)}\n\n"
                "أجب عن السؤال الحالي فقط اعتماداً على هذه الأدلة."
            ),
        }
    )
    return messages


def _format_contexts(contexts: list[dict]) -> str:
    formatted: list[str] = []
    for index, context in enumerate(contexts, start=1):
        text = str(context.get("text", "")).strip()
        source = str(context.get("source", "مصدر غير محدد")).strip() or "مصدر غير محدد"
        if text:
            formatted.append(f"[الدليل {index} | المصدر: {source}]\n{text}")
    return "\n\n".join(formatted) or "(لا يوجد دليل صالح)"


def _response_content(response: Any) -> str:
    if isinstance(response, dict):
        return str(response.get("message", {}).get("content", ""))
    message = getattr(response, "message", None)
    return str(getattr(message, "content", "") or "")


def _ollama_error_message(exc: Exception, settings: OllamaSettings) -> str:
    detail = str(exc)
    kind = type(exc).__name__.lower()
    status_code = getattr(exc, "status_code", None)
    normalized = detail.lower()

    if "timeout" in kind or "timeout" in normalized:
        return (
            "انتهت مهلة الاتصال بخادم Ollama المحلي "
            f"({settings.host}) أثناء استخدام النموذج {settings.model}."
        )
    if status_code == 404 or "model" in normalized and "not found" in normalized:
        return (
            f"نموذج Ollama المحلي {settings.model} غير متوفر. "
            "تحقق من تثبيته ومن قيمة OLLAMA_MODEL."
        )
    if "connect" in kind or "connection" in normalized or "refused" in normalized:
        return (
            f"تعذر الاتصال بخادم Ollama المحلي على {settings.host}. "
            "تأكد من أن الخدمة تعمل محلياً."
        )
    return f"حدث خطأ في Ollama المحلي أثناء توليد الإجابة: {detail}"
