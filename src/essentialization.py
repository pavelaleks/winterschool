"""
Эссенциализация: паттерны "The X are ...", "All X ...", "by nature", "tend to" и т.д.
Возвращает is_essentializing, essentialization_pattern, essentialization_span.
"""

import re
from typing import List, Dict, Any, Optional, Tuple

# Паттерны: (regex, имя_паттерна)
ESSENTIALIZATION_PATTERNS = [
    (re.compile(r"\bthe\s+(\w+)\s+are\s+[\w\s,]+", re.IGNORECASE), "the_X_are"),
    (re.compile(r"\b(\w+)\s+are\s+[\w\s,]+", re.IGNORECASE), "X_are"),
    (re.compile(r"\ball\s+(\w+)\s+", re.IGNORECASE), "all_X"),
    (re.compile(r"\b(\w+)\s+as\s+a\s+race\b", re.IGNORECASE), "X_as_race"),
    (re.compile(r"\b(\w+)\s+as\s+a\s+people\b", re.IGNORECASE), "X_as_people"),
    (re.compile(r"\b(\w+)\s+as\s+a\s+tribe\b", re.IGNORECASE), "X_as_tribe"),
    (re.compile(r"\b(\w+)\s+by\s+nature\b", re.IGNORECASE), "by_nature"),
    (re.compile(r"\b(\w+)\s+tend\s+to\s+", re.IGNORECASE), "tend_to"),
    (re.compile(r"\b(\w+)\s+are\s+prone\s+to\s+", re.IGNORECASE), "prone_to"),
    (re.compile(r"\b(\w+)\s+are\s+apt\s+to\s+", re.IGNORECASE), "apt_to"),
    (re.compile(r"\binvariably\s+[\w\s]*\b(\w+)\b", re.IGNORECASE), "invariably"),
    (re.compile(r"\balways\s+[\w\s]*\b(\w+)\b", re.IGNORECASE), "always_near"),
    (re.compile(r"\b(\w+)\s+,\s*who\s+", re.IGNORECASE), "X_who"),
]


def detect_essentialization(
    sentence_text: str,
    ethnonym_variants: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Проверяет предложение на эссенциализирующие конструкции.
    ethnonym_variants: список вариантов написания этнонима (для совпадения с группой в regex).
    Возвращает: is_essentializing (bool), essentialization_pattern (str или ""), essentialization_span (str).
    """
    text = sentence_text or ""
    if not text.strip():
        return {
            "is_essentializing": False,
            "essentialization_pattern": "",
            "essentialization_span": "",
        }

    text_lower = text.lower()
    found_pattern = ""
    found_span = ""

    for pattern_re, pattern_name in ESSENTIALIZATION_PATTERNS:
        m = pattern_re.search(text)
        if not m:
            continue
        span_text = m.group(0).strip()
        # Если заданы варианты этнонима — проверяем, что захваченное слово совпадает
        if ethnonym_variants:
            captured = m.group(1).lower() if m.lastindex >= 1 else ""
            if captured not in [v.lower() for v in ethnonym_variants]:
                continue
        if len(span_text) > len(found_span):
            found_span = span_text[:120]
            found_pattern = pattern_name

    return {
        "is_essentializing": bool(found_pattern),
        "essentialization_pattern": found_pattern,
        "essentialization_span": found_span,
    }


def detect_essentialization_in_sentence(
    sentence_text: str,
    ethnos_norm: str,
    all_ethnonym_forms: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """
    Удобная обёртка: по ethnos_norm подставляем варианты из словаря этнонимов.
    all_ethnonym_forms: {canonical: [variant1, variant2, ...]}.
    """
    variants = None
    if all_ethnonym_forms and ethnos_norm in all_ethnonym_forms:
        variants = all_ethnonym_forms[ethnos_norm]
    return detect_essentialization(sentence_text, ethnonym_variants=variants)
