#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_engine.py - Kashaf (كشّاف) RAG engine for Jordan civil-status services.

Pipeline
  1. Load + clean the 12 Arabic service files (menu junk removed, facts untouched).
  2. Chunk by FIELD (documents / fees / notes / ...), every chunk prefixed with its
     service title, so look-alike services can never be mixed up.
     Fields go through RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200,
     separators=["\\n\\n", "\\n", ".", " "]); every field in the data is < 1200 chars, so
     numbered steps are never cut in half.
  3. Understand the question: document (passport / family book / smart ID),
     situation (first issue / renewal / lost / damaged), intent (fees, documents, ...),
     with dialect normalisation and conversation memory for follow-ups.
  4. Retrieve: rule-based service routing + bge-m3 embeddings + lexical score.
  5. Generate with llama3 (Ollama) under a strict "context only" prompt.
  6. Anti-hallucination: if the context cannot answer -> exact refusal sentence.

Only hard dependencies: numpy + requests. langchain-text-splitters is used when
installed, with a built-in line-preserving fallback.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import requests

try:  # preferred splitter
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
    except Exception:
        RecursiveCharacterTextSplitter = None  # type: ignore

# --------------------------------------------------------------------------- #
# Configuration (override with environment variables)
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
CACHE_DIR = BASE_DIR / ".cache"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = "http://" + OLLAMA_HOST
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
NUM_CTX = int(os.environ.get("NUM_CTX", "8192"))
NUM_PREDICT = int(os.environ.get("NUM_PREDICT", "450"))
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "240"))
MAX_CTX_CHARS = int(os.environ.get("MAX_CTX_CHARS", "6500"))
CHUNK_SIZE = 1200   # characters
CHUNK_OVERLAP = 200
SEPARATORS = ["\n\n", "\n", ".", " "]  # a field is split only if it exceeds CHUNK_SIZE (none do in this data)

REFUSAL = "عذراً، المعلومات غير متوفرة في الوثائق."
REFUSAL_MARKER = "غير متوفرة في الوثائق"

# --------------------------------------------------------------------------- #
# Service catalogue (file stem -> document, situation, Arabic title)
# --------------------------------------------------------------------------- #
SITUATIONS = ["first", "renewal", "lost", "damaged"]
DOC_ORDER = ["passport", "family", "id"]
SERVICES: Dict[str, Tuple[str, str, str]] = {
    "passport_first_issue": ("passport", "first", "جواز سفر لأول مرة"),
    "passport_renewal": ("passport", "renewal", "تجديد جواز السفر"),
    "passport_lost": ("passport", "lost", "جواز سفر بدل فاقد"),
    "passport_damaged": ("passport", "damaged", "جواز سفر بدل تالف"),
    "family_book_first_issue": ("family", "first", "دفتر عائلة لأول مرة"),
    "family_book_renewal": ("family", "renewal", "تجديد دفتر العائلة"),
    "family_book_lost": ("family", "lost", "دفتر عائلة بدل فاقد"),
    "family_book_damaged": ("family", "damaged", "دفتر عائلة بدل تالف"),
    "id_card_first_issue": ("id", "first", "إصدار البطاقة الذكية لأول مرة"),
    "id_card_renewal": ("id", "renewal", "تجديد البطاقة الذكية"),
    "id_card_lost": ("id", "lost", "بطاقة ذكية بدل فاقد"),
    "id_card_damaged": ("id", "damaged", "بطاقة ذكية بدل تالف"),
}
DOC_NAMES = {"passport": "جواز السفر", "family": "دفتر العائلة", "id": "البطاقة الذكية"}
SIT_NAMES = {"first": "لأول مرة", "renewal": "التجديد", "lost": "بدل فاقد", "damaged": "بدل تالف"}


def service_id(doc: str, sit: str) -> str:
    for sid, (d, s, _t) in SERVICES.items():
        if d == doc and s == sit:
            return sid
    raise KeyError((doc, sit))


# --------------------------------------------------------------------------- #
# Arabic text helpers
# --------------------------------------------------------------------------- #
_DIAC = re.compile(r"[ً-ٰٟـ]")
_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي"})  # 1:1 (length preserving)
_NORM = str.maketrans(
    {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ئ": "ي", "ؤ": "و",
     "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4", "٥": "5", "٦": "6", "٧": "7",
     "٨": "8", "٩": "9"}
)


def norm(text: str) -> str:
    """Aggressive normalisation used only for matching (never for output)."""
    t = _DIAC.sub("", text or "").translate(_NORM).lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def stem(tok: str) -> str:
    for p in ("وال", "بال", "كال", "فال", "لل", "ال", "و", "ب", "ل", "ف"):
        if tok.startswith(p) and len(tok) - len(p) >= 3:
            tok = tok[len(p):]
            break
    if len(tok) > 4 and (tok.endswith("ات") or tok.endswith("ون") or tok.endswith("ين")):
        tok = tok[:-2]
    elif len(tok) > 3 and tok[-1] in "هي":
        tok = tok[:-1]
    return tok


_SYN = [
    (re.compile(r"\b(?:توفي|توفت|متوفي|متوف|متوفيه|مات|ماتت|وفاه|وفات|وفاتها|وفاته)\b"), "وفاه"),
    (re.compile(r"\b(?:ال)?(?:استعجال|مستعجل|مستعجله|عاجل|مستعجلين)\b"), "استعجال"),
    (re.compile(r"\b(?:ضاع|ضايع|ضايعه|ضاعت|فقد|فقدت|فقدان|مفقود|فاقد)\b"), "فقدان"),
    (re.compile(r"\b(?:تلف|تالف|تالفه|تلفت|اهترا|اهتري|مهتري)\b"), "تلف"),
]


def tokens(text: str) -> List[str]:
    t = norm(text)
    for rx, repl in _SYN:
        t = rx.sub(repl, t)
    return [stem(x) for x in t.split() if len(x) >= 2]


# --------------------------------------------------------------------------- #
# Loading, cleaning, chunking
# --------------------------------------------------------------------------- #
# (field key, label shown in contexts, regex on length-preserving-folded line start)
FIELD_SPECS = [
    ("audience", "الفئة المستفيدة", r"^الفئة المستفيدة"),
    ("where", "أماكن تقديم الخدمة", r"^اماكن تقديم الخدمة"),
    ("conditions", "شروط تقديم الخدمة", r"^شروط(?:\s+تقديم\s+الخدمة)?"),
    ("documents", "الوثائق المطلوبة", r"^الوثائق المطلوبة(?:\s+للحصول\s+عل[يى]\s+الخدمة)?"),
    ("procedure", "إجراءات تقديم الخدمة", r"^اجراءات(?:\s+تقديم\s+الخدمة)?"),
    ("time", "الزمن المستغرق لتقديم الخدمة", r"^الزمن المستغرق(?:\s+لتقديم\s+الخدمة)?"),
    ("partners", "شركاء الخدمة", r"^شركاء(?:\s+الخدمة)?"),
    ("notes", "ملاحظات", r"^ملاحظات"),
    ("fees", "الرسوم", r"^الرسوم"),
]
_FIELD_RES = [(k, lab, re.compile(rx + r"(?=\s|$)")) for k, lab, rx in FIELD_SPECS]
FIELD_LABEL = {k: lab for k, lab, _ in FIELD_SPECS}


@dataclass
class Chunk:
    sid: str          # service id, e.g. "passport_lost"
    doc: str
    sit: str
    field: str        # field key
    text: str         # flat text (what we return as "contexts")
    llm_text: str     # same text with line breaks kept (what the LLM sees)


def _clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.replace("\t", " ")).strip()


def parse_service_file(path: Path) -> Dict[str, List[str]]:
    """Return {field_key: [clean lines]}; website menu lines before the first field are dropped."""
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    fields: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for raw_line in raw.splitlines():
        folded = raw_line.strip().translate(_FOLD)
        hit = None
        for key, _lab, rx in _FIELD_RES:
            m = rx.match(folded)
            if m:
                hit = (key, m.end())
                break
        if hit:
            current = hit[0]
            fields.setdefault(current, [])
            rest = _clean_line(raw_line.strip()[hit[1]:])
            if rest:
                fields[current].append(rest)
            continue
        if current is None:
            continue  # menu / breadcrumb / title lines
        line = _clean_line(raw_line)
        if line:
            fields[current].append(line)
    return fields


def _split_text(text: str) -> List[str]:
    """Split on paragraph/line boundaries first, so numbered steps stay whole."""
    if len(text) <= CHUNK_SIZE:
        return [text]
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=SEPARATORS, keep_separator=False
        )
        parts = [p.strip() for p in splitter.split_text(text) if p.strip()]
        if parts:
            return parts
    parts, buf = [], ""  # fallback: greedy line packing
    for line in text.split("\n"):
        if buf and len(buf) + len(line) + 1 > CHUNK_SIZE:
            parts.append(buf)
            buf = line
        else:
            buf = f"{buf}\n{line}" if buf else line
    if buf:
        parts.append(buf)
    return parts


def load_chunks(data_dir: Path = DATA_DIR) -> List[Chunk]:
    chunks: List[Chunk] = []
    for sid, (doc, sit, title) in SERVICES.items():
        path = data_dir / f"{sid}.txt"
        if not path.exists():
            print(f"[rag_engine] WARNING: missing data file {path}", file=sys.stderr)
            continue
        fields = parse_service_file(path)
        for key, label, _rx in _FIELD_RES:
            lines = fields.get(key) or []
            if not lines:
                continue
            body = "\n".join(lines)
            if key == "notes" and norm(body) in {"لا يوجد", "لايوجد"}:
                continue
            for piece in _split_text(body):
                flat = re.sub(r"\s*\n\s*", " ", piece).strip()
                head = f"{title} - {label}: "
                chunks.append(Chunk(sid, doc, sit, key, head + flat, head + "\n" + piece))
    if not chunks:
        raise RuntimeError(f"No service files found in {data_dir}")
    return chunks


# --------------------------------------------------------------------------- #
# Question understanding
# --------------------------------------------------------------------------- #
def _R(pattern: str) -> "re.Pattern[str]":
    return re.compile(pattern)


_NOISE_ID = _R(r"بطاق\w*\s+(?:احصاء\w*\s+)?(?:ال)?جسور")  # "جسور" bridge cards are not the smart ID
DOC_PATTERNS = {
    "passport": _R(r"جواز|باسبور|بسبور|passport"),
    "family": _R(r"دفتر(?!\s+(?:ال)?خدمه)|رب\s*(?:ال)?اسر|قيد\s*(?:ال)?عايل"),
    "id": _R(r"بطاق|هويه|هويت|ذكيه|الاحوال المدنيه|\bid\b"),
}
SIT_PATTERNS = {
    "renewal": _R(r"تجديد|(?<!\w)(?:و|ب|ل|ف)?[اتي]?جدد|منتهي|منتهيه|انتهي|انتهت|(?<!\w)(?:و|ب|ل|ف)?(?:ال)?[a-z]*خلص|(?<!\w)(?:ال)?خالص|expired|renew"),
    "lost": _R(r"(?<!\w)(?:و|ب|ل|ف)?(?:ال)?(?:فاقد|فقد|ضاع|ضايع|ضيع|مفقود|سرق|مسروق)\w*"),
    "damaged": _R(r"(?<!\w)(?:و|ب|ل|ف)?(?:ال)?(?:تالف|تلف|اهتر|مهتر|ممزق|تمزق|مقطوع|متضرر|تضرر|خربان|خربت|مبلول|مكسور)\w*"),
    "first": _R(r"اول مره|لاول|للمره الاولي|(?<!\w)(?:و|ب|ل|ف)?(?:ال)?جديد\w*"),
}
SIT_ISSUE = _R(r"اصدار|استخراج")  # "issue" -> first, only when no other situation

INTENT_PATTERNS = {
    "fees": _R(r"رسوم|رسم\b|ادفع|بدفع|يدفع|دفع|سعر|تكلف|كلف|بكم|مبلغ|دينار|غرامه|كفاله|تعهد|اغلي|ارخص|فلوس|اجره|قديش\s+(?:ب|ت|ي)?(?:دفع|كلف)"),
    "time": _R(r"يستغرق|تستغرق|بستغرق|(?<!\w)(?:و|ب|ل|ف)?(?:ال)?(?:مده|وقت|زمن|دقيقه|ساعه)|(?<!\w)(?:ي|ب|ت)?(?:ياخد|اخد)|كم يوم|بيجهز|(?<!\w)تجهز"),
    "documents": _R(r"اوراق|ورق|وثائق|وثيقه|مطلوب|يلزم|بلزم|احتاج|بحتاج|اجيب|احضر|احضار|مستندات|صوره|صور|بدي اجيب|شو بدي"),
    "where": _R(r"(?<!\w)(?:و|ب|ل|ف)?(?:وين|اين|مكان|اماكن|مكتب|مكاتب|سفار|قنصل|برا|بره|خارج)"),
    "procedure": _R(r"خطوات|خطوه|اجراءات|اجراء|كيف|طريقه|مراحل"),
    "open": _R(r"شو بعمل|شو بسوي|ايش بعمل|ما العمل|شو الحل|شو بصير"),
    "conditions": _R(r"شروط|شرط|يحق|بحق|يجوز|مسموح|ممنوع|عمري|عمر|قاصر|وصايه|ولايه|ولي|موافقه|زوج|زوجه|مرتي|مراتي|ابني|ابنتي|بنتي|\bاب\b|\bام\b|توفي|متوفي|اكبر|رب الاسره"),
    "partners": _R(r"شركاء|شريك|جهات|الجهه"),
}
INTENT_FIELDS = {
    "fees": ["fees", "notes"],
    "time": ["time", "notes"],
    "documents": ["documents", "notes"],
    "where": ["where"],
    "procedure": ["procedure", "notes"],
    "conditions": ["conditions", "audience", "notes"],
    "open": ["documents", "procedure", "fees", "where", "notes"],
    "partners": ["partners"],
}
FIELD_PRIORITY = ["fees", "time", "documents", "conditions", "where", "procedure", "audience", "partners", "notes"]

STOP = set(map(stem, norm(
    "كم شو ايش وين كيف هل ما ماذا متي لماذا من مين في علي الي عن مع او ان اذا لو هذا هذه هاي هيك "
    "بدي بدك بقدر اقدر ممكن يمكن يعني طيب لو سمحت مرحبا مرحباً شكرا هو هي انا انت نحن ده هاد هادا ليش ايه "
    "رسوم سعر تكلفه مده وقت اوراق وثائق مطلوب المطلوب مطلوبه يلزم لازم الخدمه خدمه الطلب طلب "
    "تقديم اقدم بعمل بسوي اعمل عمل احتاج بحتاج بلزمني بلزم كل بعد قبل عند هل قديش كام اللي الذي التي "
    ""
).split()))


def _find_all(patterns: Dict[str, "re.Pattern[str]"], text: str) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for key, rx in patterns.items():
        for m in rx.finditer(text):
            out.append((m.start(), key))
    return sorted(out)


@dataclass
class Understanding:
    pairs: Dict[str, Set[str]]       # doc -> situations (empty set = unknown)
    intents: Set[str]
    docs_found: bool
    sits_found: Set[str]
    inherited: bool = False


def understand(question: str, prev: Optional[Dict[str, Set[str]]] = None,
               prev_intents: Optional[Set[str]] = None) -> Understanding:
    t = _NOISE_ID.sub(" ", norm(question))
    doc_hits = _find_all(DOC_PATTERNS, t)
    sit_hits = _find_all(SIT_PATTERNS, t)
    if not sit_hits or all(s == "first" for _p, s in sit_hits):
        if SIT_ISSUE.search(t) and not any(s != "first" for _p, s in sit_hits):
            m = SIT_ISSUE.search(t)
            sit_hits.append((m.start(), "first"))  # type: ignore[union-attr]
    # A lone "جديد" next to a renewal/lost/damaged word is not "first issue"
    if any(s in ("renewal", "lost", "damaged") for _p, s in sit_hits):
        sit_hits = [(p, s) for p, s in sit_hits if s != "first" or re.search(r"اول مره|لاول|للمره الاولي", t)]

    # first mention of each document
    docs: Dict[str, int] = {}
    for pos, d in doc_hits:
        docs.setdefault(d, pos)
    # incidental "ID card" (e.g. "البطاقة الشخصية" as a required paper) next to another document
    if "id" in docs and len(docs) > 1 and re.search(r"بطاق\w*\s+(?:ال)?شخصي|هويت\w*\s+(?:ال)?شخصي", t):
        docs.pop("id")

    pairs: Dict[str, Set[str]] = {d: set() for d in sorted(docs, key=docs.get)}  # type: ignore[arg-type]
    orphan: Set[str] = set()
    for pos, s in sit_hits:
        if not docs:
            orphan.add(s)
            continue
        nearest = min(docs, key=lambda d: (abs(docs[d] - pos), docs[d]))
        pairs[nearest].add(s)
    sits_found = {s for _p, s in sit_hits}

    intents = {k for k, rx in INTENT_PATTERNS.items() if rx.search(t)}

    inherited = False
    prev = prev or {}
    if not pairs and prev:                       # follow-up: "وكم بتكلف؟"
        pairs = {d: (set(orphan) if orphan else set(s)) for d, s in prev.items()}
        inherited = True
    elif pairs and prev:                         # new document mentioned without a situation
        prev_sits = set().union(*prev.values()) if prev else set()
        for d, s in pairs.items():
            if not s:
                s |= orphan or (set(prev.get(d, set())) or prev_sits)
                inherited = inherited or bool(s)
    elif pairs and orphan:
        for s in pairs.values():
            if not s:
                s |= orphan
    # short follow-up that only switches document/service ("طيب والبطاقة الذكية؟"): keep the topic asked before
    if not intents and prev_intents and (inherited or docs) and len(t.split()) <= 8:
        intents = set(prev_intents)
    return Understanding(pairs, intents, bool(docs), sits_found, inherited)


def services_for(pairs: Dict[str, Set[str]]) -> List[str]:
    out: List[str] = []
    for doc, sits in pairs.items():
        for sit in (sorted(sits, key=SITUATIONS.index) if sits else SITUATIONS):
            out.append(service_id(doc, sit))
    return out[:8]


# --------------------------------------------------------------------------- #
# Ollama client
# --------------------------------------------------------------------------- #
class Ollama:
    def __init__(self, host: str = OLLAMA_HOST):
        self.host = host
        self.http = requests.Session()
        self.http.trust_env = False  # never route localhost through a corporate proxy

    def embed(self, texts: List[str], model: str = EMBED_MODEL) -> np.ndarray:
        try:
            r = self.http.post(f"{self.host}/api/embed", json={"model": model, "input": texts}, timeout=300)
            r.raise_for_status()
            vecs = r.json()["embeddings"]
        except Exception:  # older Ollama: one call per text
            vecs = []
            for t in texts:
                r = self.http.post(f"{self.host}/api/embeddings", json={"model": model, "prompt": t}, timeout=300)
                r.raise_for_status()
                vecs.append(r.json()["embedding"])
        arr = np.asarray(vecs, dtype=np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
        return arr

    def chat(self, messages: List[dict], model: str = LLM_MODEL, retries: int = 2) -> str:
        last: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                r = self.http.post(
                    f"{self.host}/api/chat",
                    json={
                        "model": model, "messages": messages, "stream": False,
                        "options": {"temperature": 0, "seed": 7, "num_ctx": NUM_CTX,
                                    "num_predict": NUM_PREDICT, "top_p": 0.9, "repeat_penalty": 1.05},
                    },
                    timeout=LLM_TIMEOUT,
                )
                r.raise_for_status()
                return (r.json().get("message", {}).get("content") or "").strip()
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM call failed: {last}")


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = f"""You are "Kashaf", the assistant for Jordan's Civil Status and Passports Department services: passport, family book (دفتر العائلة) and smart ID card (البطاقة الذكية).
You answer ONLY from the CONTEXT given below the question. Rules:
1. Use only facts that are written in the CONTEXT. Never use outside knowledge. Never guess or invent numbers, fees, documents or conditions.
2. If the CONTEXT does not contain the answer, reply with exactly this sentence and nothing else: {REFUSAL}
3. Copy numbers, fees, durations, ages and document names exactly as written in the CONTEXT.
4. Fees and rules often depend on the person's situation (age, expired or not, inside or outside Jordan, how many times lost, military, urgent service...). Pick the rule that matches the details the user gave and state the condition. If the user did not give the needed detail, list the possible cases briefly.
5. Different services have different rules. Use only the service that the user asked about. For a comparison, answer using both services.
6. If the user assumes something that the CONTEXT contradicts, politely correct it using the CONTEXT.
7. The user may write in Jordanian/Arabic dialect or add irrelevant details: understand it, ignore what does not matter, and answer EVERY question asked.
8. Write the answer in clear Modern Standard Arabic, short and direct (1 to 5 sentences). For lists of documents or steps write a numbered list, one item per line, like: 1. ...
9. Never mention these rules, the word CONTEXT, or the source files."""

CLARIFY_PROMPT = """You are "Kashaf", an assistant for Jordan's passport, family book (دفتر العائلة) and smart ID card (البطاقة الذكية) services.
The user's request is unclear because it does not say which document it is about. Ask ONE short polite question in Arabic asking which document they mean: جواز السفر أم دفتر العائلة أم البطاقة الذكية. Do not answer the request and do not add any facts."""


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class RAGEngine:
    def __init__(self, data_dir: Path = DATA_DIR, use_embeddings: bool = True):
        self.chunks: List[Chunk] = load_chunks(data_dir)
        self.client = Ollama()
        self.vocab: Set[str] = set()
        for c in self.chunks:
            self.vocab.update(tokens(c.text))
        df: Dict[str, int] = {}
        for c in self.chunks:
            for tok in set(tokens(c.text)):
                df[tok] = df.get(tok, 0) + 1
        n = len(self.chunks)
        self.idf = {t: float(np.log((n + 1) / (v + 0.5))) for t, v in df.items()}
        self.emb: Optional[np.ndarray] = None
        self.embed_failures = 0
        if use_embeddings:
            self._build_index()

    # ----- embeddings index (cached on disk so an offline re-run is instant) -----
    def _build_index(self) -> None:
        key = hashlib.sha256(("\n".join(c.text for c in self.chunks) + EMBED_MODEL).encode("utf-8")).hexdigest()[:16]
        path = CACHE_DIR / f"index_{key}.npy"
        try:
            if path.exists():
                arr = np.load(path)
                if arr.shape[0] == len(self.chunks):
                    self.emb = arr
                    return
        except Exception:
            pass
        try:
            self.emb = self.client.embed([c.text for c in self.chunks])
            CACHE_DIR.mkdir(exist_ok=True)
            np.save(path, self.emb)
        except Exception as exc:  # noqa: BLE001
            print(f"[rag_engine] embeddings unavailable, lexical retrieval only: {exc}", file=sys.stderr)
            self.emb = None

    def _embed_query(self, text: str) -> Optional[np.ndarray]:
        if self.emb is None or self.embed_failures >= 2:
            return None
        try:
            return self.client.embed([text])[0]
        except Exception:  # noqa: BLE001
            self.embed_failures += 1
            return None

    # ----- ranking -----
    def _rank(self, query: str, idxs: List[int]) -> List[int]:
        q_tokens = [t for t in tokens(query) if t not in STOP]
        lex = np.zeros(len(idxs), dtype=np.float32)
        for j, i in enumerate(idxs):
            ct = set(tokens(self.chunks[i].text))
            lex[j] = sum(self.idf.get(t, 0.0) for t in q_tokens if t in ct)
        sem = np.zeros(len(idxs), dtype=np.float32)
        qv = self._embed_query(query)
        if qv is not None and self.emb is not None:
            sem = self.emb[idxs] @ qv

        def mm(x: np.ndarray) -> np.ndarray:
            return (x - x.min()) / (x.max() - x.min() + 1e-9) if len(x) > 1 else np.ones_like(x)

        score = (mm(lex) + mm(sem)) if qv is not None else mm(lex)
        return [idxs[j] for j in np.argsort(-score, kind="stable")]

    def _select(self, question: str, sids: List[str], intents: Set[str]) -> List[Chunk]:
        by_sid: Dict[str, List[int]] = {}
        for i, c in enumerate(self.chunks):
            by_sid.setdefault(c.sid, []).append(i)
        name_tokens = {stem(w) for sid in sids for w in norm(SERVICES[sid][2]).split()}
        name_tokens |= {stem(w) for w in norm("جواز سفر باسبور دفتر عائله اسره بطاقه ذكيه هويه تجديد فقدان تلف").split()}
        q_tokens = {t for t in tokens(question) if t not in STOP and t not in name_tokens and not t.isdigit()}

        def overlaps(c: Chunk) -> bool:  # secondary fields only when they mention something asked
            return bool(q_tokens & set(tokens(c.text.split(": ", 1)[1])))

        specific = intents - {"conditions"}
        picked: List[Chunk] = []
        if specific and sids:
            wanted: List[str] = []
            for it in sorted(intents):
                for f in INTENT_FIELDS[it]:
                    if f not in wanted:
                        wanted.append(f)
            wanted.sort(key=FIELD_PRIORITY.index)
            primary = set()
            for it in specific:
                primary.add(INTENT_FIELDS[it][0])
            if "conditions" in intents:
                primary |= {"conditions", "audience"}
            if "open" in specific:
                primary |= set(INTENT_FIELDS["open"])
            for f in wanted:               # field-major: every service gets its key field first
                for sid in sids:
                    for i in by_sid.get(sid, []):
                        c = self.chunks[i]
                        if c.field != f:
                            continue
                        always = f in primary or (f == "notes" and "fees" in intents)
                        if always or overlaps(c):
                            picked.append(c)
        else:
            idxs = [i for sid in sids for i in by_sid.get(sid, [])] if sids else list(range(len(self.chunks)))
            k = 4 if not sids else (6 if len(sids) == 1 else 5)
            picked = [self.chunks[i] for i in self._rank(question, idxs)[:k]]
            if "conditions" in intents and len(sids) == 1:
                for i in by_sid.get(sids[0], []):
                    c = self.chunks[i]
                    if c.field in ("conditions", "audience") and c not in picked:
                        picked.append(c)

        out, used = [], 0
        for c in picked:                   # respect the context budget
            if used + len(c.llm_text) > MAX_CTX_CHARS and out:
                continue
            out.append(c)
            used += len(c.llm_text)
        return out

    # ----- gate for questions that name no document -----
    def _coverage(self, question: str, extra_skip: Set[str]) -> Tuple[int, float]:
        toks = [t for t in tokens(question) if t not in STOP and t not in extra_skip and not t.isdigit()]
        if not toks:
            return 0, 1.0
        hit = sum(1 for t in toks if t in self.vocab)
        return len(toks), hit / len(toks)

    # ----- public API -----
    def new_session(self) -> "ChatSession":
        return ChatSession(self)

    def answer(self, question: str, prev: Optional[Dict[str, Set[str]]] = None,
               history: Optional[List[Tuple[str, str]]] = None,
               prev_intents: Optional[Set[str]] = None) -> dict:
        question = (question or "").strip()
        if not question:
            return self._result(REFUSAL, [], "empty", {})
        u = understand(question, prev, prev_intents)
        sids = services_for(u.pairs)

        if not sids:  # no document named (and nothing to inherit)
            generic_skip = {stem(w) for w in norm("تجديد اجدد جدد فاقد ضاع ضايع تالف اصدار اطلع جديد اول مره خلص منتهي").split()}
            n, cov = self._coverage(question, generic_skip)
            if n == 0 and (u.sits_found or u.intents):
                return self._clarify(question)
            if n == 0 or cov < 0.5:
                return self._result(REFUSAL, [], "out_of_scope", {})
            # topic words exist in the corpus (e.g. "الاستعجال"): search globally
        chunks = self._select(question, sids, u.intents)
        if not chunks:
            return self._result(REFUSAL, [], "no_context", u.pairs)
        return self._generate(question, chunks, sids, u, history or [])

    # ----- generation -----
    def _clarify(self, question: str) -> dict:
        try:
            text = self.client.chat([{"role": "system", "content": CLARIFY_PROMPT},
                                     {"role": "user", "content": question}])
            if not text or REFUSAL_MARKER in text:
                raise ValueError("bad clarification")
        except Exception:  # noqa: BLE001
            text = "هل تقصد جواز السفر، أم دفتر العائلة، أم البطاقة الذكية؟"
        return self._result(text, [], "clarify", {})

    def _generate(self, question: str, chunks: List[Chunk], sids: List[str],
                  u: Understanding, history: List[Tuple[str, str]]) -> dict:
        ctx_block = "\n\n".join(f"[{i + 1}] {c.llm_text}" for i, c in enumerate(chunks))
        titles = "، ".join(dict.fromkeys(SERVICES[s][2] for s in sids)) if sids else "غير محددة"
        hist = ""
        if history:
            hist = "\nPrevious turns in this chat (for understanding follow-ups only):\n" + "\n".join(
                f"- Q: {q[:200]}\n  A: {a[:250]}" for q, a in history[-2:]) + "\n"
        user = (f"CONTEXT:\n{ctx_block}\n\nService(s) the user is asking about: {titles}\n{hist}"
                f"\nUSER QUESTION: {question}\n\nAnswer in Arabic using only the CONTEXT.")
        contexts = list(dict.fromkeys(c.text for c in chunks))
        try:
            raw = self.client.chat([{"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user", "content": user}])
        except Exception as exc:  # noqa: BLE001  - model unreachable: stay grounded, never crash
            print(f"[rag_engine] {exc}", file=sys.stderr)
            return self._result(contexts[0], contexts[:1], "llm_error", u.pairs)
        answer = self._postprocess(raw)
        if not answer or REFUSAL_MARKER in answer:
            return self._result(REFUSAL, [], "refused", u.pairs)
        res = self._result(answer, contexts, "answered", u.pairs)
        res["intents"] = sorted(u.intents)
        return res

    @staticmethod
    def _postprocess(text: str) -> str:
        text = re.sub(r"^\s*(?:Answer|الإجابة|الجواب)\s*[:：]\s*", "", text.strip(), flags=re.I)
        text = text.replace("\r", "")
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    @staticmethod
    def _result(answer: str, contexts: List[str], mode: str, pairs: Dict[str, Set[str]]) -> dict:
        return {"answer": answer, "contexts": contexts, "mode": mode,
                "pairs": {d: sorted(s) for d, s in pairs.items()}, "intents": []}


class ChatSession:
    """One user = one session. Remembers which service the chat is about."""

    def __init__(self, engine: RAGEngine):
        self.engine = engine
        self.state: Dict[str, Set[str]] = {}
        self.intents: Set[str] = set()
        self.history: List[Tuple[str, str]] = []

    def ask(self, question: str) -> dict:
        res = self.engine.answer(question, self.state, self.history, self.intents)
        if res["pairs"]:
            self.state = {d: set(s) for d, s in res["pairs"].items()}
            if res.get("intents"):
                self.intents = set(res["intents"])
        self.history.append((question, res["answer"]))
        return res


# --------------------------------------------------------------------------- #
# Batch runner (used by main.py and evaluate.py)
# --------------------------------------------------------------------------- #
def _atomic_write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def run_questions(questions_path: str, answers_path: str = "answers.json",
                  engine: Optional[RAGEngine] = None,
                  on_message: Optional[Callable[[dict, dict, float], None]] = None) -> dict:
    """Read questions, answer every message (never raises per message), write answers_path."""
    qpath, apath = Path(questions_path), Path(answers_path)
    data = json.loads(qpath.read_text(encoding="utf-8-sig"))
    chats = data["chats"] if isinstance(data, dict) else data
    try:
        engine = engine or RAGEngine()
    except Exception as exc:  # noqa: BLE001
        print(f"[rag_engine] engine failed to start: {exc}", file=sys.stderr)
        engine = None
    for chat in chats:
        session = engine.new_session() if engine else None
        for msg in chat.get("messages", []):
            t0 = time.time()
            try:
                if session is None:
                    raise RuntimeError("engine unavailable")
                res = session.ask(str(msg.get("question", "")))
            except Exception as exc:  # noqa: BLE001
                print(f"[rag_engine] {msg.get('message_id')}: {exc}", file=sys.stderr)
                res = {"answer": "error", "contexts": [], "mode": "error", "pairs": {}}
            msg["answer"] = res["answer"]
            msg["contexts"] = res["contexts"]
            if on_message:
                try:
                    on_message(msg, res, time.time() - t0)
                except Exception:  # noqa: BLE001
                    pass
        _atomic_write(apath, data)  # a partial file beats no file
    _atomic_write(apath, data)
    return data


if __name__ == "__main__":
    eng = RAGEngine()
    sess = eng.new_session()
    for line in sys.stdin:
        if line.strip():
            print(json.dumps(sess.ask(line.strip()), ensure_ascii=False, indent=2))
