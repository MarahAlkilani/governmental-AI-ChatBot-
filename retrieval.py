"""Offline, service-aware BM25 retrieval over original Arabic service documents."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

from rank_bm25 import BM25Plus

from arabic_text import tokenize_arabic, tokenize_for_retrieval
from ingest import DocumentRecord, DocumentSection, load_documents


_VARIANT_SUFFIXES = ("first_issue", "renewal", "damaged", "lost")
_VARIANT_ALIASES = {
    "first_issue": {"اول", "مره", "جديد", "اولى"},
    "renewal": {"تجديد", "جدد", "مجدد"},
    "lost": {"فقدان"},
    "damaged": {"تالف"},
}
_VARIANT_TERMS = set().union(*_VARIANT_ALIASES.values())
_STOPWORDS = set(
    tokenize_for_retrieval(
        "ما ماذا من في على الى عن هل هو هي هم هن هذا هذه ذلك تلك كم "
        "و او ثم مع عند كان تكون يتم التي الذي معلومات اريد استطيع "
        "ماذا بدي بقدر شو وين فين قديش"
    )
)
_STOPWORDS.discard("مكان")
_SECTION_INTENTS = (
    (
        {"رسوم", "سعر", "اسعار", "تكلفه", "كلفه", "ثمن"},
        {"الرسوم"},
    ),
    (
        {"وقت", "مده", "زمن", "يستغرق", "دقيقه", "دقايق", "متى"},
        {"الزمن"},
    ),
    (
        {"وثايق", "مستندات", "ورق", "بحتاج", "احتاج", "مطلوب"},
        {"الوثايق"},
    ),
    (
        {"شرط", "شروط", "متطلبات", "لازم"},
        {"الشروط"},
    ),
    (
        {"اجراءات", "خطوات", "طريقه", "كيف"},
        {"الاجراءات"},
    ),
    (
        {"مكان", "اماكن", "اين", "موقع"},
        {"الاماكن"},
    ),
    (
        {
            "ملاحظه",
            "ملاحظات",
            "استثناء",
            "استثناءات",
            "تعهد",
            "كفاله",
        },
        {"الملاحظات"},
    ),
)
_SECTION_ALIASES = {
    "الفئه": {"فئه"},
    "الاماكن": {"اماكن", "مكان"},
    "الشروط": {"شرط", "شروط"},
    "الوثايق": {"وثايق", "مستندات", "ورق"},
    "الاجراءات": {"اجراءات"},
    "الزمن": {"زمن", "مده"},
    "الشركاء": {"شركاء"},
    "الملاحظات": {"ملاحظات", "ملاحظه"},
    "الرسوم": {"رسوم"},
}
_TITLE_NAVIGATION = set(tokenize_arabic("الرئيسية الخدمات دليل الخدمات شارك"))
_SERVICE_PROFILE_STOPWORDS = {"بدل"}
_CLAUSE_SPLIT = re.compile(r"[,،;؛؟?!.\n]+")
_SECTION_BOILERPLATE = set(tokenize_arabic("تقديم الخدمة للحصول على الخدمة"))


@dataclass(frozen=True)
class _Passage:
    text: str
    source: str
    service_group: str
    variant: str | None
    section_key: str
    start_line: int
    tokens: list[str]


def _retrieval_tokens(text: str) -> list[str]:
    tokens = []
    for token in tokenize_for_retrieval(text):
        if token.startswith("ال") and len(token) > 4:
            token = token[2:]
        if token not in _STOPWORDS:
            tokens.append(token)
    return tokens


def _service_parts(document: DocumentRecord) -> tuple[str, str | None]:
    for suffix in _VARIANT_SUFFIXES:
        marker = f"_{suffix}"
        if document.service_id.endswith(marker):
            return document.service_id[:-len(marker)], suffix
    return document.service_id, None


def _service_profiles(
    documents: Sequence[DocumentRecord],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    grouped: dict[str, list[DocumentRecord]] = defaultdict(list)
    variants: dict[str, set[str]] = defaultdict(set)
    for document in documents:
        group, variant = _service_parts(document)
        grouped[group].append(document)
        if variant is not None:
            variants[group].add(variant)

    profiles = {}
    for group, records in grouped.items():
        title_counts: Counter[str] = Counter()
        for record in records:
            title = record.service_name or record.service_id
            title_tokens = set(_retrieval_tokens(title))
            title_tokens.difference_update(_TITLE_NAVIGATION)
            title_tokens.difference_update(_VARIANT_TERMS)
            title_tokens.difference_update(_SERVICE_PROFILE_STOPWORDS)
            title_counts.update(title_tokens)
        threshold = max(1, (len(records) + 1) // 2)
        profiles[group] = {
            token for token, count in title_counts.items() if count >= threshold
        }
    return profiles, variants


def _matched_services(
    tokens: Sequence[str],
    profiles: dict[str, set[str]],
) -> set[str]:
    query_terms = set(tokens)
    overlaps = {
        group: query_terms.intersection(profile)
        for group, profile in profiles.items()
        if query_terms.intersection(profile)
    }
    selected = set()
    for group, overlap in overlaps.items():
        unique_terms = overlap.difference(
            set().union(
                *(profile for other, profile in profiles.items() if other != group)
            )
        )
        if unique_terms or len(overlap) >= 2:
            selected.add(group)
    return selected


def _detected_variants(tokens: Sequence[str]) -> set[str]:
    query_terms = set(tokens)
    return {
        variant
        for variant, aliases in _VARIANT_ALIASES.items()
        if query_terms.intersection(aliases)
    }


def _query_constraints(
    question: str,
    profiles: dict[str, set[str]],
    variants_by_service: dict[str, set[str]],
) -> tuple[set[str], dict[str, set[str]], set[str]]:
    service_targets: set[str] = set()
    variant_targets: dict[str, set[str]] = defaultdict(set)
    section_targets: set[str] = set()
    question_tokens = set(_retrieval_tokens(question))
    for triggers, sections in _SECTION_INTENTS:
        if question_tokens.intersection(triggers):
            section_targets.update(sections)

    clauses = [clause for clause in _CLAUSE_SPLIT.split(question) if clause.strip()]
    if not clauses:
        clauses = [question]
    for clause in clauses:
        tokens = _retrieval_tokens(clause)
        services = _matched_services(tokens, profiles)
        service_targets.update(services)
        detected = _detected_variants(tokens)
        if detected:
            for service in services:
                available = variants_by_service.get(service, set())
                matching = detected.intersection(available)
                if matching:
                    variant_targets[service].update(matching)

    return service_targets, variant_targets, section_targets


def _section_key(title: str) -> str:
    tokens = set(_retrieval_tokens(title))
    for section, aliases in _SECTION_ALIASES.items():
        if tokens.intersection(aliases):
            return section
    return ""


def _has_content(section: DocumentSection) -> bool:
    body = section.text
    lines = body.splitlines()
    if not lines:
        return False
    title_tokens = tokenize_arabic(section.title)
    first_line_tokens = tokenize_arabic(lines[0])
    content_tokens = (
        first_line_tokens[len(title_tokens):]
        if first_line_tokens[:len(title_tokens)] == title_tokens
        else first_line_tokens
    )
    content_tokens.extend(
        token
        for line in lines[1:]
        for token in tokenize_arabic(line)
    )
    return any(token not in _SECTION_BOILERPLATE for token in content_tokens)


def _make_passages(documents: Sequence[DocumentRecord]) -> list[_Passage]:
    passages = []
    for document in documents:
        group, variant = _service_parts(document)
        service_name = document.service_name or document.service_id
        for section in document.sections:
            if not _has_content(section):
                continue
            indexed_tokens = _retrieval_tokens(
                f"{service_name} {service_name} {section.title} {section.text}"
            )
            if indexed_tokens:
                passages.append(
                    _Passage(
                        text=section.text,
                        source=document.filename,
                        service_group=group,
                        variant=variant,
                        section_key=_section_key(section.title),
                        start_line=section.start_line,
                        tokens=indexed_tokens,
                    )
                )
    return passages


def _select_results(
    candidates: list[tuple[_Passage, float]],
    service_targets: set[str],
    top_k: int,
) -> list[tuple[_Passage, float]]:
    chosen = []
    chosen_passages: set[tuple[str, int]] = set()
    for service in sorted(service_targets):
        match = next(
            (item for item in candidates if item[0].service_group == service),
            None,
        )
        if match is not None:
            chosen.append(match)
            chosen_passages.add((match[0].source, match[0].start_line))
    chosen.sort(key=lambda item: -item[1])
    if len(chosen) >= top_k:
        return chosen[:top_k]

    chosen.extend(
        item
        for item in candidates
        if (item[0].source, item[0].start_line) not in chosen_passages
    )
    return sorted(chosen[:top_k], key=lambda item: -item[1])


def retrieve(question: str, top_k: int = 3) -> list[dict[str, object]]:
    """Return relevant exact source sections, prioritizing requested services."""
    if not isinstance(question, str) or not question.strip():
        return []
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        return []

    documents = load_documents()
    if not documents:
        return []
    profiles, variants_by_service = _service_profiles(documents)
    service_targets, variant_targets, section_targets = _query_constraints(
        question,
        profiles,
        variants_by_service,
    )
    if not service_targets:
        return []
    query_tokens = _retrieval_tokens(question)
    if not query_tokens:
        return []

    passages = _make_passages(documents)
    if not passages:
        return []
    bm25 = BM25Plus([passage.tokens for passage in passages])
    scores = bm25.get_scores(query_tokens)
    query_terms = set(query_tokens)
    candidates = []
    for passage, score in zip(passages, scores):
        if not query_terms.intersection(passage.tokens):
            continue
        if service_targets and passage.service_group not in service_targets:
            continue
        requested_variants = variant_targets.get(passage.service_group)
        if requested_variants and passage.variant not in requested_variants:
            continue
        if section_targets and passage.section_key not in section_targets:
            continue
        candidates.append((passage, float(score)))

    candidates.sort(
        key=lambda item: (
            -item[1],
            item[0].source.casefold(),
            item[0].start_line,
        )
    )
    results = _select_results(candidates, service_targets, top_k)
    return [
        {"text": passage.text, "source": passage.source, "score": score}
        for passage, score in results
    ]
