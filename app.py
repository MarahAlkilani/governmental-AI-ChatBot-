# -*- coding: utf-8 -*-
"""
app.py - Kashaf (كشّاف) Streamlit UI.   Run:  streamlit run app.py

Theme  : "Kashaf Searchlight" - midnight blue #0F172A with a warm amber #F59E0B glow.
Sidebar: the CSS never hides the header or any sidebar toggle, so it stays collapsible.
Feature: numbered / "أولاً، ثانياً" procedures in an answer become a Visual Procedure
         Timeline, rendered with st.markdown(unsafe_allow_html=True) (no components.v1).
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

# --------------------------------------------------------------------------- #
# Step parsing (pure functions - no Streamlit calls)
# --------------------------------------------------------------------------- #
_ORD_WORD = (
    r"(?:أول|اول|ثاني|ثالث|رابع|خامس|سادس|سابع|ثامن|تاسع|عاشر)"
    r"[ً-ْ]*ا?[ً-ْ]*"
)
ORD_LINE = re.compile(r"^\s*(?:[-•*]\s*)?" + _ORD_WORD + r"\s*[،,:\-–—.]\s*(.+?)\s*$")
ORD_INLINE = re.compile(r"(?<![\w])" + _ORD_WORD + r"\s*[،,:]")
NUM_LINE = re.compile(r"^\s*(?:[-•*]\s*)?(?:\*\*)?(?:\d{1,2}|[٠-٩]{1,2})\s*[\.\)\-–:،]\s*(?:\*\*)?\s*(.+?)\s*$")
_END = ".!؟?۔:؛"


def parse_steps(text: str) -> Optional[Tuple[str, List[str], str]]:
    """Return (intro, steps, outro) when the text holds >= 2 procedural steps, else None."""
    text = (text or "").replace("\r", "")
    if len(ORD_INLINE.findall(text)) >= 2:  # "أولاً، ... ثانياً، ..." written inside one paragraph
        text = ORD_INLINE.sub(lambda m: "\n" + m.group(0), text)
    intro: List[str] = []
    steps: List[str] = []
    outro: List[str] = []
    state = "intro"
    for line in text.split("\n"):
        stripped = line.strip()
        m = NUM_LINE.match(line) or ORD_LINE.match(line)
        if m:
            steps.append(m.group(1).strip())
            state = "steps"
            continue
        if not stripped:
            if state == "steps":
                state = "outro"
            continue
        if state == "intro":
            intro.append(stripped)
        elif state == "steps" and steps and steps[-1][-1:] not in _END:
            steps[-1] += " " + stripped          # wrapped continuation of the previous step
        else:
            outro.append(stripped)
            state = "outro"
    if len(steps) < 2:
        return None
    return " ".join(intro), steps, " ".join(outro)


def _inline(text: str) -> str:
    """Escape for safe HTML and for Streamlit's markdown ($ -> math, ` -> code); keep **bold**."""
    out = html.escape(text, quote=False)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    return out.replace("$", "&#36;").replace("`", "&#96;").replace("*", "&#42;").replace("_", "&#95;")


def timeline_html(intro: str, steps: List[str], outro: str) -> str:
    """Single-line HTML (no blank lines / indentation) so markdown never breaks it."""
    parts = ['<div class="ks-answer" dir="rtl">']
    if intro:
        parts.append(f'<p class="ks-text">{_inline(intro)}</p>')
    parts.append('<div class="ks-timeline">')
    for i, step in enumerate(steps, 1):
        last = " ks-last" if i == len(steps) else ""
        parts.append(
            f'<div class="ks-step{last}" style="animation-delay:{(i - 1) * 70}ms">'
            f'<div class="ks-node">{i}</div>'
            f'<div class="ks-card"><span class="ks-tag">الخطوة {i}</span>{_inline(step)}</div></div>'
        )
    parts.append("</div>")
    if outro:
        parts.append(f'<p class="ks-text ks-outro">{_inline(outro)}</p>')
    parts.append("</div>")
    return "".join(parts)


def plain_html(text: str) -> str:
    body = "<br>".join(_inline(p) for p in (text or "").split("\n") if p.strip())
    return f'<div class="ks-answer" dir="rtl"><p class="ks-text">{body}</p></div>'


def answer_html(text: str) -> str:
    parsed = parse_steps(text)
    return timeline_html(*parsed) if parsed else plain_html(text)


def sources_html(contexts: List[str]) -> str:
    items = "".join(f'<div class="ks-src">{_inline(c)}</div>' for c in contexts)
    return f'<div dir="rtl">{items}</div>'


# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');
:root{
  --ks-bg:#0F172A; --ks-surface:#162036; --ks-surface2:#1E293B; --ks-amber:#F59E0B;
  --ks-amber-dim:#B45309; --ks-text:#E2E8F0; --ks-muted:#94A3B8;
  --ks-line:rgba(245,158,11,.28); --ks-glow:rgba(245,158,11,.30);
}
html, body, .stApp, [data-testid="stAppViewContainer"]{
  background:radial-gradient(1200px 600px at 85% -10%, rgba(245,158,11,.10), transparent 60%), var(--ks-bg) !important;
  color:var(--ks-text); font-family:'Tajawal','Segoe UI',Tahoma,sans-serif;
}
/* Header stays in the layout (it hosts the sidebar toggle) - only made transparent. */
[data-testid="stHeader"]{ background:transparent !important; }
/* Sidebar toggle buttons: explicitly kept visible in every Streamlit version. */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"]{
  visibility:visible !important; opacity:1 !important; display:flex !important;
}
[data-testid="collapsedControl"] button, [data-testid="stSidebarCollapsedControl"] button,
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"] button{
  color:var(--ks-amber) !important;
}
[data-testid="collapsedControl"] svg, [data-testid="stSidebarCollapsedControl"] svg,
[data-testid="stExpandSidebarButton"] svg, [data-testid="stSidebarCollapseButton"] svg{
  fill:var(--ks-amber) !important; color:var(--ks-amber) !important;
}
footer{ visibility:hidden; }
[data-testid="stDecoration"]{ display:none; }
[data-testid="stSidebar"]{ background:var(--ks-surface) !important; border-inline-end:1px solid var(--ks-line); }
[data-testid="stSidebar"] *{ color:var(--ks-text); }
.ks-brand{ font-size:1.9rem; font-weight:700; color:var(--ks-amber); text-shadow:0 0 22px var(--ks-glow); line-height:1.1; }
.ks-sub{ color:var(--ks-muted); font-size:.9rem; margin-bottom:.6rem; }
.ks-hero{ text-align:center; padding:1.2rem 0 .6rem; }
.ks-hero .ks-brand{ font-size:2.6rem; }
.ks-beam{ height:3px; width:220px; margin:.6rem auto 0; border-radius:3px;
  background:linear-gradient(90deg, transparent, var(--ks-amber), transparent); box-shadow:0 0 18px var(--ks-glow); }
.ks-pill{ display:inline-block; padding:.15rem .7rem; border-radius:999px; font-size:.78rem; border:1px solid var(--ks-line); color:var(--ks-muted); }
.ks-ok{ color:#86EFAC; border-color:rgba(134,239,172,.4); } .ks-bad{ color:#FCA5A5; border-color:rgba(252,165,165,.4); }
.stButton > button{ background:var(--ks-surface2); color:var(--ks-text); border:1px solid var(--ks-line); border-radius:12px; text-align:right; }
.stButton > button:hover{ border-color:var(--ks-amber); color:var(--ks-amber); box-shadow:0 0 14px var(--ks-glow); }
[data-testid="stChatMessage"]{ background:var(--ks-surface); border:1px solid rgba(148,163,184,.14); border-radius:16px; padding:.9rem 1rem; }
[data-testid="stChatInput"]{ border-color:var(--ks-line) !important; }
[data-testid="stChatInput"] textarea{ color:var(--ks-text) !important; }
.ks-answer{ direction:rtl; text-align:right; line-height:1.9; font-size:1.05rem; }
.ks-text{ margin:.1rem 0 .6rem; }
.ks-outro{ margin-top:.9rem; color:var(--ks-muted); }
.ks-timeline{ position:relative; margin:.8rem 0 .4rem; padding-inline-start:0; }
.ks-timeline::before{ content:""; position:absolute; top:14px; bottom:14px; right:21px; width:2px;
  background:linear-gradient(180deg, var(--ks-amber), rgba(245,158,11,.12)); box-shadow:0 0 10px var(--ks-glow); }
.ks-step{ position:relative; display:flex; align-items:flex-start; gap:14px; margin-bottom:12px;
  animation:ks-rise .45s ease both; }
.ks-node{ flex:0 0 44px; height:44px; border-radius:50%; display:flex; align-items:center; justify-content:center;
  font-weight:700; font-size:1.05rem; color:#0F172A; background:var(--ks-amber); z-index:1;
  box-shadow:0 0 0 4px var(--ks-bg), 0 0 18px var(--ks-glow); }
.ks-last .ks-node{ background:#FBBF24; box-shadow:0 0 0 4px var(--ks-bg), 0 0 26px rgba(245,158,11,.55); }
.ks-card{ flex:1; background:var(--ks-surface2); border:1px solid var(--ks-line); border-radius:14px; padding:.6rem .95rem; }
.ks-tag{ display:block; font-size:.72rem; letter-spacing:.04em; color:var(--ks-amber); margin-bottom:.1rem; }
.ks-src{ background:var(--ks-surface2); border-inline-start:3px solid var(--ks-amber); border-radius:8px;
  padding:.45rem .7rem; margin:.35rem 0; font-size:.88rem; color:var(--ks-muted); line-height:1.7; }
@keyframes ks-rise{ from{ opacity:0; transform:translateY(8px);} to{ opacity:1; transform:none;} }
@media (prefers-reduced-motion: reduce){ .ks-step{ animation:none; } }
</style>
"""

EXAMPLES = [
    "كم يستغرق تجديد دفتر العائلة؟",
    "شو الأوراق المطلوبة لتجديد جواز السفر؟",
    "عمري 17 وجوازي باقي عليه 3 شهور، كم رسوم التجديد؟",
    "شو خطوات إصدار البطاقة الذكية لأول مرة؟",
    "إذا توفي الأب والأم، مين بصير رب الأسرة؟",
    "كم سعر تذكرة الطيران من عمان إلى دبي؟",
]


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="كشّاف يحمّل الوثائق …")
def get_engine():
    from rag_engine import RAGEngine
    return RAGEngine()


@st.cache_data(ttl=15, show_spinner=False)
def ollama_status() -> bool:
    import requests
    from rag_engine import OLLAMA_HOST
    try:
        s = requests.Session()
        s.trust_env = False
        return s.get(f"{OLLAMA_HOST}/api/tags", timeout=2).ok
    except Exception:  # noqa: BLE001
        return False


def _queue(question: str) -> None:
    st.session_state["pending"] = question


def _reset(engine) -> None:
    st.session_state["messages"] = []
    st.session_state["chat"] = engine.new_session()


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"], avatar="🔦" if msg["role"] == "assistant" else "🧑"):
        if msg["role"] == "user":
            st.markdown(plain_html(msg["content"]), unsafe_allow_html=True)
            return
        st.markdown(answer_html(msg["content"]), unsafe_allow_html=True)
        if msg.get("contexts"):
            with st.expander(f"المصادر ({len(msg['contexts'])})"):
                st.markdown(sources_html(msg["contexts"]), unsafe_allow_html=True)


def main() -> None:
    from rag_engine import EMBED_MODEL, LLM_MODEL

    st.set_page_config(page_title="كشّاف | Kashaf", page_icon="🔦", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    engine = get_engine()
    if "messages" not in st.session_state:
        _reset(engine)

    with st.sidebar:
        st.markdown('<div class="ks-brand">🔦 كشّاف</div><div class="ks-sub">مساعد خدمات الأحوال المدنية والجوازات</div>',
                    unsafe_allow_html=True)
        up = ollama_status()
        st.markdown(f'<span class="ks-pill {"ks-ok" if up else "ks-bad"}">Ollama: {"متصل" if up else "غير متصل"}</span> '
                    f'<span class="ks-pill">{LLM_MODEL} · {EMBED_MODEL}</span>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**جرّب سؤالاً**")
        for i, q in enumerate(EXAMPLES):
            st.button(q, key=f"ex{i}", on_click=_queue, args=(q,), use_container_width=True)
        st.markdown("---")
        st.button("🗑️ محادثة جديدة", on_click=_reset, args=(engine,), use_container_width=True)
        st.caption("الإجابات من الوثائق الرسمية فقط. إن لم تتوفر المعلومة يخبرك كشّاف بذلك.")

    st.markdown('<div class="ks-hero"><div class="ks-brand">كشّاف</div>'
                '<div class="ks-sub">جواز السفر · دفتر العائلة · البطاقة الذكية — إجابات دقيقة من الوثائق الرسمية</div>'
                '<div class="ks-beam"></div></div>', unsafe_allow_html=True)

    for msg in st.session_state["messages"]:
        render_message(msg)

    question = st.chat_input("اكتب سؤالك هنا …") or st.session_state.pop("pending", None)
    if question:
        user_msg = {"role": "user", "content": question}
        st.session_state["messages"].append(user_msg)
        render_message(user_msg)
        with st.spinner("كشّاف يبحث في الوثائق …"):
            try:
                res = st.session_state["chat"].ask(question)
                reply = {"role": "assistant", "content": res["answer"], "contexts": res["contexts"]}
            except Exception as exc:  # noqa: BLE001
                reply = {"role": "assistant", "content": f"حدث خطأ أثناء المعالجة: {exc}", "contexts": []}
        st.session_state["messages"].append(reply)
        render_message(reply)


if __name__ == "__main__":
    main()
