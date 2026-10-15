"""Local RAG CLI: python main.py questions.json."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import sys

from generation import NO_CONTEXT_MESSAGE, generate_answer
from retrieval import retrieve
from project_paths import resolve_document_directory, use_document_directory

ERROR_MESSAGE = "تعذر إكمال الإجابة حالياً بسبب خطأ محلي. يرجى المحاولة لاحقاً."


def _validate_input(data: object) -> None:
    """Validate before processing so malformed files cannot produce partial output."""
    if not isinstance(data, dict) or not isinstance(data.get("chats"), list):
        raise ValueError("يجب أن يحتوي الإدخال على قائمة chats.")
    for chat in data["chats"]:
        if not isinstance(chat, dict) or "chat_id" not in chat:
            raise ValueError("كل محادثة تحتاج إلى chat_id.")
        if not isinstance(chat.get("messages"), list):
            raise ValueError("كل محادثة تحتاج إلى قائمة messages.")
        for message in chat["messages"]:
            if not isinstance(message, dict) or "message_id" not in message:
                raise ValueError("كل رسالة تحتاج إلى message_id.")
            if not isinstance(message.get("question"), str):
                raise ValueError("يجب أن يكون question نصاً في كل رسالة.")


def _used_contexts(answer: str, contexts: list[dict]) -> list[str] | None:
    """Map P06's verbatim answer back to exact original retrieval excerpts.

    Blank-line separated answer blocks must each match complete source lines.
    Keep retrieval order and deduplicate without altering the original strings.
    If identical text appears in multiple sources, use the first matching result.
    None means the answer cannot be traced to the current evidence.
    """
    used = set()
    for block in answer.split("\n\n"):
        # P06 joins excerpts with two newlines; source excerpts may already
        # end with a newline. Ignore separator newlines, not source whitespace.
        block = block.strip("\n")
        if not block.strip():
            continue
        match = next(
            (index for index, context in enumerate(contexts)
             if f"\n{block}\n" in f"\n{context['text']}\n"),
            None,
        )
        if match is None:
            return None
        used.add(match)
    if not used:
        return None
    return list(dict.fromkeys(contexts[index]["text"] for index in sorted(used)))


def answer_questions(data: dict) -> dict:
    """Return an enriched copy, processing each chat independently and in order."""
    _validate_input(data)
    output = deepcopy(data)
    for chat in output["chats"]:
        history = []
        for message in chat["messages"]:
            answer, official_contexts = NO_CONTEXT_MESSAGE, []
            try:
                contexts = retrieve(message["question"], top_k=3)
                candidate = generate_answer(message["question"], contexts, deepcopy(history))
                if not isinstance(candidate, str) or not candidate.strip():
                    raise ValueError("لم يرجع المولد نص إجابة صالحاً.")
                if candidate == NO_CONTEXT_MESSAGE:
                    print(f"امتنع المولد عن الإجابة؛ عدد الأدلة المسترجعة: {len(contexts)}.",
                          file=sys.stderr)
                elif candidate.startswith(("تعذر", "انتهت مهلة", "نموذج Ollama", "حدث خطأ")):
                    answer = ERROR_MESSAGE
                    print(candidate, file=sys.stderr)
                elif candidate != NO_CONTEXT_MESSAGE and re.search(r"[\u0621-\u064a]", candidate):
                    used = _used_contexts(candidate, contexts)
                    if used:
                        answer, official_contexts = candidate, used
                    else:
                        # Never attribute an untraceable response to evidence.
                        print("لم يمكن إسناد الإجابة إلى الأدلة الحالية.", file=sys.stderr)
            except Exception as exc:
                # A failed turn must not discard other messages or expose raw
                # exception details as purported government information.
                answer = ERROR_MESSAGE
                print(f"تعذر معالجة رسالة بسبب خطأ محلي ({type(exc).__name__}): {exc}",
                      file=sys.stderr)
            message["answer"] = answer
            message["contexts"] = official_contexts
            history.append({
                "message_id": deepcopy(message["message_id"]),
                "question": message["question"],
                "answer": answer,
                "contexts": official_contexts.copy(),
            })
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="الإجابة محلياً عن أسئلة الخدمات الحكومية.")
    parser.add_argument("questions", type=Path, help="مسار ملف questions.json")
    parser.add_argument("--documents", type=Path, help="Explicit official TXT document directory")
    args = parser.parse_args(argv)
    questions = args.questions.expanduser().resolve()
    destination = questions.parent / "answers.json"
    reference = questions.parent / "answers_example.json"
    print(f"Input questions: {questions}")
    print(f"Reference example: {reference}" + ("" if reference.is_file() else " (not found)"))
    print(f"Generated answers: {destination}")
    try:
        with questions.open(encoding="utf-8") as stream:
            data = json.load(stream)
        _validate_input(data)
        directory = resolve_document_directory(questions, args.documents)
        print(f"Document directory: {directory}")
        with use_document_directory(directory):
            output = answer_questions(data)
        # The evaluation places the input beside main.py. An explicit path also
        # works: answers.json is written beside that input, independent of cwd.
        if destination.resolve() in (questions, reference.resolve()):
            raise ValueError("يجب ألا يكون ملف الإدخال هو answers.json.")
        with destination.open("w", encoding="utf-8") as stream:
            json.dump(output, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"تعذر قراءة الإدخال أو كتابة الإجابات: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
