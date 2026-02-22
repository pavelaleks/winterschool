"""
Извлечение упоминаний этнонимов из текста с учётом исторических вариантов написания.
Использует словарь resources/ethnonyms.yml и строит регулярные выражения.
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import yaml


def load_ethnonyms() -> Dict[str, List[str]]:
    """Загружает ethnonyms.yml: каноническое имя -> список вариантов."""
    path = Path(__file__).resolve().parent.parent / "resources" / "ethnonyms.yml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def build_ethnonym_patterns(ethnonyms: Dict[str, List[str]]) -> List[Tuple[str, re.Pattern]]:
    """
    Строит для каждого канонического этнонима regex, учитывающий все варианты.
    Returns: [(canonical_name, compiled_pattern), ...]
    """
    result = []
    for canonical, variants in ethnonyms.items():
        if not variants:
            continue
        # Уникальные варианты, экранируем для regex, сортируем по длине (длинные первыми)
        escaped = sorted(set(re.escape(v) for v in variants), key=len, reverse=True)
        pattern = r"\b(" + "|".join(escaped) + r")\b"
        result.append((canonical, re.compile(pattern, re.IGNORECASE)))
    return result


def extract_ethnonym_mentions(
    text: str,
    patterns: List[Tuple[str, re.Pattern]],
) -> List[Dict[str, Any]]:
    """
    Находит все вхождения этнонимов в тексте.
    Returns: список dict с ключами: ethnonym (canonical), match_text, start, end.
    """
    mentions = []
    for canonical, pattern in patterns:
        for m in pattern.finditer(text):
            mentions.append({
                "ethnonym": canonical,
                "match_text": m.group(1),
                "start": m.start(),
                "end": m.end(),
            })
    return sorted(mentions, key=lambda x: x["start"])


def map_mentions_to_sentences(
    mentions: List[Dict[str, Any]],
    sentences: List[Dict[str, Any]],
    doc_start_offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Для каждого упоминания определяет номер предложения (в рамках документа).
    sentences: список предложений с start_char, end_char (относительно документа).
    doc_start_offset: сдвиг начала документа в общем тексте (обычно 0).
    Returns: копии упоминаний с добавленным ключом sentence_index.
    """
    result = []
    for m in mentions:
        s_start = m["start"] - doc_start_offset
        sent_idx = None
        for i, sent in enumerate(sentences):
            if sent["start_char"] <= s_start < sent["end_char"]:
                sent_idx = i
                break
        out = dict(m)
        out["sentence_index"] = sent_idx
        result.append(out)
    return result


def get_canonical_lemma_set(ethnonyms: Optional[Dict[str, List[str]]] = None) -> set:
    """
    Множество канонических имён (лемм) для поиска по леммам.
    В ethnonyms.yml каноническое имя = лемма (kalmuk, kirghiz, ...); варианты (kalmucks, Kirghizes) приводятся к ней.
    """
    ethnonyms = ethnonyms or load_ethnonyms()
    return set(k.lower().strip() for k in ethnonyms.keys() if k)


def extract_ethnonym_mentions_from_sentences(
    sentences: List[Dict[str, Any]],
    ethnonyms: Optional[Dict[str, List[str]]] = None,
    lemma_to_canonical: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Находит упоминания этнонимов по леммам токенов (вместо regex по сырому тексту).
    sentences: список предложений с token_objects; у каждого токена — text, lemma.
    Возвращает список dict: ethnonym (canonical), match_text (словоформа в тексте), sentence_index, start, end (0 для совместимости).
    """
    ethnonyms = ethnonyms or load_ethnonyms()
    if lemma_to_canonical is None:
        lemma_to_canonical = {canon.lower().strip(): canon for canon in ethnonyms}
    canonical_lower = set(lemma_to_canonical.keys())
    mentions = []
    for sent_idx, sent in enumerate(sentences):
        tokens = sent.get("token_objects", []) or sent.get("tokens", [])
        if not tokens:
            continue
        for t in tokens:
            if isinstance(t, dict):
                lemma = (t.get("lemma") or t.get("text", "")).lower().strip()
                word = t.get("text", "")
            else:
                lemma = str(t).lower().strip()
                word = str(t)
            if lemma in canonical_lower:
                canon = lemma_to_canonical.get(lemma, lemma)
                mentions.append({
                    "ethnonym": canon,
                    "match_text": word,
                    "sentence_index": sent_idx,
                    "start": 0,
                    "end": 0,
                })
    return sorted(mentions, key=lambda x: (x["sentence_index"], x.get("start", 0)))


def get_ethnonym_patterns_cached():
    """Загружает словарь и строит паттерны один раз (для использования в пайплайне)."""
    ethnonyms = load_ethnonyms()
    return build_ethnonym_patterns(ethnonyms)
