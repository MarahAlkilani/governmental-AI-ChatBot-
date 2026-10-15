"""Arabic text normalization and tokenization for retrieval matching."""
from __future__ import annotations

import re
import unicodedata
from typing import List


_NORMALIZATION = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ة": "ه",
        "ؤ": "و",
        "ئ": "ي",
        **dict(zip("٠١٢٣٤٥٦٧٨٩", "0123456789")),
        **dict(zip("۰۱۲۳۴۵۶۷۸۹", "0123456789")),
    }
)
_TATWEEL = "\u0640"
_PUNCTUATION = re.compile(r"[^\w\s]|_", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_RETRIEVAL_SYNONYMS = {
    "اين": "مكان",
    "باسبور": "جواز",
    "باسبوراتي": "جواز",
    "بقدر": "استطيع",
    "وبقدر": "استطيع",
    "بتجدد": "تجديد",
    "بجدد": "تجديد",
    "تجدد": "تجديد",
    "اجدد": "تجديد",
    "اجددها": "تجديد",
    "جدد": "تجديد",
    "جوازات": "جواز",
    "خربان": "تالف",
    "خربانه": "تالف",
    "خربت": "تالف",
    "ضايع": "فقدان",
    "ضايعه": "فقدان",
    "ضاع": "فقدان",
    "ضاعت": "فقدان",
    "فاقد": "فقدان",
    "فقد": "فقدان",
    "فقدت": "فقدان",
    "فين": "مكان",
    "وين": "مكان",
    "لاول": "اول",
    "مفقود": "فقدان",
    "مفقوده": "فقدان",
    "مكسور": "تالف",
    "مكسوره": "تالف",
    "اوراق": "وثائق",
    "بحتاج": "احتاج",
    "هويه": "بطاقه",
    "هوية": "بطاقه",
    "ضايعين": "فقدان",
    "خرب": "تالف",
    "شو": "ماذا",
    "ايش": "ماذا",
    "اش": "ماذا",
    "قديش": "كم",
    "الاوراق": "وثايق",
    "الوثايق": "وثايق",
    "اوراق": "وثايق",
    "لتجديد": "تجديد",
    "تجديده": "تجديد",
    "تجديدها": "تجديد",
    "لاصدار": "اصدار",
    "وتجديد": "تجديد",
    "وتجدد": "تجديد",
    "تالفه": "تالف",
    "التالفه": "تالف",
    "متهالك": "تالف",
    "مهترئ": "تالف",
    "الضايع": "فقدان",
    "الضايعه": "فقدان",
    "الفاقد": "فقدان",
    "الضياع": "فقدان",
    "الخراب": "تالف",
    "الخربان": "تالف",
    "الخربانه": "تالف",
}
_SAFE_CONJUNCTION_PREFIXES = {
    "اجراءات",
    "بطاقه",
    "دفتر",
    "جواز",
    "دقيقه",
    "رسوم",
    "سفر",
    "شروط",
    "عائله",
    "فقدان",
    "مكان",
    "مطلوبه",
    "مره",
    "وثايق",
    "اول",
    "تجديد",
    "تالف",
}


def normalize_arabic(text: str) -> str:
    """Normalize text for matching without changing the caller's source text."""
    without_marks = "".join(
        character
        for character in text
        if character != _TATWEEL and unicodedata.category(character) != "Mn"
    )
    normalized = without_marks.translate(_NORMALIZATION)
    normalized = _PUNCTUATION.sub(" ", normalized)
    return _WHITESPACE.sub(" ", normalized).strip()


def tokenize_arabic(text: str) -> List[str]:
    """Return normalized Arabic, Latin, and numeric tokens for retrieval."""
    return _TOKEN.findall(normalize_arabic(text))


def tokenize_for_retrieval(text: str) -> List[str]:
    """Return normalized tokens with common service and colloquial variants folded."""
    tokens = []
    for token in tokenize_arabic(text):
        canonical = _RETRIEVAL_SYNONYMS.get(token)
        if canonical is None:
            for prefix in ("وال", "بال", "كال", "فال", "لل", "ال"):
                if token.startswith(prefix) and len(token) - len(prefix) >= 3:
                    canonical = _RETRIEVAL_SYNONYMS.get(
                        token[len(prefix):],
                        token[len(prefix):],
                    )
                    break
            if canonical is None and token.startswith("و") and len(token) > 3:
                remainder = token[1:]
                if remainder in _SAFE_CONJUNCTION_PREFIXES:
                    canonical = _RETRIEVAL_SYNONYMS.get(remainder, remainder)
        tokens.append(canonical if canonical is not None else token)
    return tokens
