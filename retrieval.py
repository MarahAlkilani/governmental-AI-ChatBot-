"""Offline BM25 retrieval over original Arabic government-service passages."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from rank_bm25 import BM25Plus

from arabic_text import tokenize_arabic
from ingest import DocumentRecord, DocumentSection, load_documents


_SECTION_BOILERPLATE = set(tokenize_arabic("تقديم الخدمة للحصول على الخدمة"))
_STOPWORDS = set(
    tokenize_arabic(
        "ما ماذا من في على الى عن هل هو هي هم هن هذا هذه ذلك تلك كم "
        "و او ثم مع عند كان تكون يتم التي الذي معلومات"
    )
)
def _retrieval_tokens(text: str) -> list[str]:
    tokens = []
    for token in tokenize_arabic(text):
        if token.startswith("ال") and len(token) > 4:
            token = token[2:]
        if token not in _STOPWORDS:
            tokens.append(token)
    return tokens


_TIME_TERMS = set(_retrieval_tokens("وقت مدة متى يستغرق دقيقة دقائق زمن مستغرق"))
_DOCUMENT_TERMS = set(_retrieval_tokens("اوراق مستندات وثائق مطلوبة"))
_FEE_TERMS = set(_retrieval_tokens("رسوم سعر اسعار تكلفة كلفة"))
_CONDITION_TERMS = set(_retrieval_tokens("متطلبات شرط شروط"))
_PROCEDURE_TERMS = set(_retrieval_tokens("خطوات طريقة اجراءات"))
_SERVICE_NAME_WEIGHT = 10
_QUERY_INTENTS = (
    (_TIME_TERMS, ("الزمن", "المستغرق"), "الزمن المستغرق"),
    (_DOCUMENT_TERMS, ("الوثائق", "المطلوبة"), "الوثائق المطلوبة"),
    (_FEE_TERMS, ("الرسوم",), "الرسوم"),
    (_CONDITION_TERMS, ("شروط",), "شروط"),
    (_PROCEDURE_TERMS, ("إجراءات",), "إجراءات"),
)


def _query_tokens(question: str) -> tuple[list[str], set[tuple[str, ...]]]:
    tokens = _retrieval_tokens(question)
    target_sections = set()
    for trigger_terms, expansion, section_title in _QUERY_INTENTS:
        if any(token in trigger_terms for token in tokens):
            tokens.extend(
                token
                for token in _retrieval_tokens(" ".join(expansion))
                if token not in tokens
            )
            target_sections.add(tuple(_retrieval_tokens(section_title)))
    return tokens, target_sections


@dataclass(frozen=True)
class _Passage:
    text: str
    source: str
    section_title: str
    start_line: int
    tokens: list[str]


def _has_content(section: DocumentSection) -> bool:
    lines = section.text.splitlines()
    if not lines:
        return False

    title_tokens = tokenize_arabic(section.title)
    first_line_tokens = tokenize_arabic(lines[0])
    body_tokens = (
        first_line_tokens[len(title_tokens):]
        if first_line_tokens[:len(title_tokens)] == title_tokens
        else first_line_tokens
    )
    body_tokens.extend(
        token
        for line in lines[1:]
        for token in tokenize_arabic(line)
    )
    return any(token not in _SECTION_BOILERPLATE for token in body_tokens)


def _passages(documents: Sequence[DocumentRecord]) -> list[_Passage]:
    passages = []
    for document in documents:
        for section in document.sections:
            if not _has_content(section):
                continue
            service_name = document.service_name or document.service_id
            service_tokens = _retrieval_tokens(service_name)
            passage_tokens = _retrieval_tokens(" ".join((section.title, section.text)))
            tokens = service_tokens * _SERVICE_NAME_WEIGHT + passage_tokens
            if not tokens:
                continue
            passages.append(
                _Passage(
                    text=section.text,
                    source=document.filename,
                    section_title=section.title,
                    start_line=section.start_line,
                    tokens=tokens,
                )
            )
    return passages


def retrieve(question: str, top_k: int = 3) -> list[dict[str, object]]:
    """Return the highest-scoring matching source sections for an Arabic question."""
    if not isinstance(question, str) or not question.strip():
        return []
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        return []

    query_tokens, target_sections = _query_tokens(question)
    if not query_tokens:
        return []

    passages = _passages(load_documents())
    if not passages:
        return []

    bm25 = BM25Plus([passage.tokens for passage in passages])
    scores = bm25.get_scores(query_tokens)
    candidates = [
        (passage, float(score))
        for passage, score in zip(passages, scores)
        if set(query_tokens).intersection(passage.tokens)
        and (
            not target_sections
            or any(
                set(target_section).issubset(
                    _retrieval_tokens(passage.section_title)
                )
                for target_section in target_sections
            )
        )
    ]
    candidates.sort(
        key=lambda result: (
            -result[1],
            result[0].source.casefold(),
            result[0].start_line,
            result[0].section_title,
        )
    )
    return [
        {
            "text": passage.text,
            "source": passage.source,
            "score": score,
        }
        for passage, score in candidates[:top_k]
    ]
