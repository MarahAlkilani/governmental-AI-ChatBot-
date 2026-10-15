"""Arabic RTL Streamlit chat UI for the offline government-services RAG app."""
from __future__ import annotations

from typing import Any

import requests
import streamlit as st

from retrieval import retrieve

try:
    from generation import generate_answer
except ModuleNotFoundError as error:
    if error.name != "generation":
        raise
    generate_answer = None


ABSTENTION = "عذراً، المعلومات غير متوفرة في الوثائق."
GENERATOR_UNAVAILABLE = (
    "خدمة توليد الإجابات غير متاحة حالياً؛ لم يتم العثور على وحدة التوليد."
)
OLLAMA_UNAVAILABLE = (
    "تعذر الاتصال بخدمة Ollama حالياً. تأكد من تشغيلها ثم حاول مرة أخرى."
)


def _render_evidence(contexts: list[dict[str, Any]]) -> None:
    if not contexts:
        return
    with st.expander("المصادر والأدلة"):
        for index, context in enumerate(contexts, start=1):
            st.caption(f"{index}. المصدر: {context['source']}")
            st.text(context["text"])


def _render_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_evidence(message.get("contexts", []))


def main() -> None:
    st.set_page_config(
        page_title="المساعد الذكي للخدمات الحكومية",
        page_icon="🇯🇴",
        layout="centered",
    )
    st.markdown(
        """
        <style>
        html, body, [class*="stApp"] {
            direction: rtl;
            text-align: right;
        }
        [data-testid="stChatMessage"] {
            direction: rtl;
            text-align: right;
        }
        [data-testid="stChatInput"] textarea {
            direction: rtl;
            text-align: right;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("المساعد الذكي للخدمات الحكومية 🇯🇴")
    st.caption("اسأل عن خدمات جواز السفر ودفتر العائلة والبطاقة الذكية.")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "turn_history" not in st.session_state:
        st.session_state.turn_history = []

    for message in st.session_state.chat_messages:
        _render_message(message)

    question = st.chat_input("اكتب سؤالك عن الخدمات الحكومية...")
    if not question:
        return

    st.session_state.chat_messages.append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.markdown(question)

    contexts = retrieve(question, top_k=3)
    if not contexts:
        answer = ABSTENTION
    elif generate_answer is None:
        answer = GENERATOR_UNAVAILABLE
    else:
        try:
            with st.spinner("جارٍ إعداد الإجابة..."):
                answer = generate_answer(
                    question,
                    contexts,
                    st.session_state.turn_history,
                )
        except requests.exceptions.RequestException:
            answer = OLLAMA_UNAVAILABLE

    assistant_message = {
        "role": "assistant",
        "content": answer,
        "contexts": contexts,
    }
    st.session_state.chat_messages.append(assistant_message)
    _render_message(assistant_message)

    if contexts and generate_answer is not None and answer != OLLAMA_UNAVAILABLE:
        st.session_state.turn_history.append(
            {
                "question": question,
                "answer": answer,
                "contexts": [context["text"] for context in contexts],
            }
        )


if __name__ == "__main__":
    main()
