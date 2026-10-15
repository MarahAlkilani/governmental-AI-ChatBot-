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
