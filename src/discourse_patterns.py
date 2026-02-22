"""
Дискурсные паттерны: эссенциализация и экзотизация.
"""

import re
from typing import List, Dict, Any
from collections import defaultdict


# Паттерны эссенциализации
ESSENTIALIZATION_PATTERNS = [
    re.compile(r"\bthe\s+(\w+)\s+are\s+", re.IGNORECASE),   # The X are
    re.compile(r"\b(\w+)\s+are\s+", re.IGNORECASE),          # X are (в начале клаузы)
    re.compile(r"\ball\s+(\w+)\s+", re.IGNORECASE),         # All X
    re.compile(r"\b(\w+)\s+as\s+a\s+race\b", re.IGNORECASE),
    re.compile(r"\b(\w+)\s+by\s+nature\b", re.IGNORECASE),
]

# Маркеры экзотизации
EXOTICIZATION_MARKERS = [
    "strange", "curious", "peculiar", "primitive", "savage",
    "wild", "picturesque", "oriental", "exotic", "quaint",
    "unusual", "remarkable", "singular", "odd", "fantastic",
    "mysterious", "romantic", "colourful", "colorful", "striking",
]
EXOTIC_SET = set(EXOTICIZATION_MARKERS)


def get_ethnonym_set_from_piro(piro_records: List[Dict]) -> set:
    """Множество канонических этнонимов из PIRO."""
    return set(r["P"] for r in piro_records)


def count_exoticization(
    piro_records: List[Dict],
) -> Dict[str, int]:
    """Распределение маркеров экзотизации по этносам (из контекстов PIRO)."""
    from collections import Counter
    dist = defaultdict(Counter)
    for r in piro_records:
        eth = r["P"]
        context_lower = (r.get("context") or "").lower()
        words = set(re.findall(r"[a-z]+", context_lower))
        for w in EXOTIC_SET:
            if w in words:
                dist[eth][w] += 1
    return {eth: dict(c) for eth, c in dist.items()}


def get_essentialization_table(
    doc_list: List[Dict],
    ethnonym_patterns: List[tuple],
) -> Dict[str, int]:
    """
    Итоговая таблица: этнос -> количество эссенциализирующих конструкций.
    Используем точный поиск конструкций с этнонимами.
    """
    counts = defaultdict(int)
    for doc in doc_list:
        for sent in doc["sentences"]:
            text = sent.get("text", "")
            text_lower = text.lower()
            for canonical, pattern in ethnonym_patterns:
                counted = False
                for m in pattern.finditer(text):
                    if counted:
                        break
                    eth_match = m.group(1).lower()
                    if f"the {eth_match} are" in text_lower or f" {eth_match} are " in text_lower:
                        counts[canonical] += 1
                        counted = True
                    elif f"all {eth_match}" in text_lower or f"all {eth_match} " in text_lower:
                        counts[canonical] += 1
                        counted = True
                    elif f"{eth_match} as a race" in text_lower or f"{eth_match} by nature" in text_lower:
                        counts[canonical] += 1
                        counted = True
    return dict(counts)


def get_essentialization_examples(
    piro_records: List[Dict],
    min_per_ethnos: int = 5,
) -> Dict[str, List[Dict[str, str]]]:
    """
    Примеры эссенциализации по этносам: минимум min_per_ethnos уникальных предложений на этнос.
    Дедупликация по нормализованному тексту предложения. Возвращает {ethnos: [{sentence_text, pattern, source_pointer}, ...]}.
    """
    seen_texts: Dict[str, set] = defaultdict(set)
    out: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in piro_records:
        if not r.get("is_essentializing"):
            continue
        eth = r.get("P") or r.get("ethnos_norm", "")
        if not eth:
            continue
        text = (r.get("sentence_text") or r.get("essentialization_span") or "").strip()
        if not text:
            continue
        norm = " ".join(text.lower().split())
        if norm in seen_texts[eth]:
            continue
        seen_texts[eth].add(norm)
        fn = r.get("file_name") or (r.get("O_metadata") or {}).get("file", "")
        sid = r.get("sent_idx") if r.get("sent_idx") is not None else (r.get("O_metadata") or {}).get("sentence_index", "")
        source_pointer = f"{fn}#{sid}"
        out[eth].append({
            "sentence_text": text[:500],
            "pattern": r.get("essentialization_pattern", ""),
            "source_pointer": source_pointer,
        })
        if len(out[eth]) >= min_per_ethnos:
            pass
    return dict(out)
