"""
Извлечение эпитетов: приоритет dependency-based (amod, copula, appos), fallback — окно ±5 токенов.
Стоп-лист, метод извлечения и confidence (0–1).
"""

import re
from typing import List, Dict, Any, Tuple

# Стоп-лист эпитетов (исключаем)
EPITHET_STOPLIST = {
    "other", "many", "several", "some", "such", "great", "little",
    "first", "last", "few", "one", "two", "three", "next", "same", "different",
    "certain", "various", "numerous", "most", "more", "less", "much", "all",
    "both", "either", "neither", "no", "any", "every", "each",
    "very", "quite", "rather", "almost", "enough", "too",
}


def _is_stop(word: str) -> bool:
    w = word.lower().strip()
    if w in EPITHET_STOPLIST:
        return True
    if w.isdigit() or (len(w) <= 1):
        return True
    return False


def extract_epithets_dependency(
    sentence: Dict[str, Any],
    ethnonym_match_text: str,
) -> Tuple[List[str], str, float]:
    """
    Извлекает эпитеты по зависимостям spaCy.
    Возвращает (epithets_list, method, confidence).
    method: amod | copula | appos | fallback_window
    confidence: 0.85 (amod/copula/appos), 0.4 (fallback).
    """
    tokens = sentence.get("token_objects", [])
    epithets = []
    method = "fallback_window"
    confidence = 0.4

    if not tokens:
        # Fallback по окну слов из sentence_text
        text = sentence.get("text", "")
        words = text.split()
        eth_lower = ethnonym_match_text.lower()
        for i, w in enumerate(words):
            if w.lower() == eth_lower:
                start = max(0, i - 5)
                end = min(len(words), i + 6)
                for j in range(start, end):
                    if j != i and words[j].lower() not in (eth_lower, "the", "a", "an") and not _is_stop(words[j]):
                        epithets.append(words[j])
                return (list(dict.fromkeys(epithets)), "fallback_window", 0.4)
        return ([], "fallback_window", 0.4)

    # Найти индекс токена этнонима
    eth_idx = None
    for i, t in enumerate(tokens):
        if t.get("text", "").lower() == ethnonym_match_text.lower():
            eth_idx = i
            break
    if eth_idx is None:
        text = sentence.get("text", "")
        words = text.split()
        eth_lower = ethnonym_match_text.lower()
        for i, w in enumerate(words):
            if w.lower() == eth_lower:
                start = max(0, i - 5)
                end = min(len(words), i + 6)
                for j in range(start, end):
                    if j != i and not _is_stop(words[j]):
                        epithets.append(words[j])
                return (list(dict.fromkeys(epithets)), "fallback_window", 0.4)
        return ([], "fallback_window", 0.4)

    # 1) amod: adjective, зависящий от головы-этнонима
    for t in tokens:
        head_i = t.get("head_i")
        dep = t.get("dep", "")
        if head_i == eth_idx and dep == "amod":
            w = t.get("text", "")
            if not _is_stop(w):
                epithets.append(w)
                method = "amod"
                confidence = 0.85

    # 2) copula: ethnos nsubj + cop + acomp/attr
    # Этноним — nsubj при глаголе cop (is/are), acomp или attr — прилагательное при cop
    nsubj_eth = False
    for t in tokens:
        if t.get("i") == eth_idx and t.get("dep") in ("nsubj", "nsubjpass"):
            nsubj_eth = True
            break
    if nsubj_eth:
        for t in tokens:
            dep = t.get("dep", "")
            head_i = t.get("head_i")
            # Проверяем, что головой является copula (ROOT с lemma be)
            head_tok = next((x for x in tokens if x.get("i") == head_i), None)
            if head_tok and head_tok.get("lemma", "").lower() in ("be", "is", "are", "was", "were"):
                if dep in ("acomp", "attr") and t.get("pos") == "ADJ":
                    w = t.get("text", "")
                    if not _is_stop(w):
                        epithets.append(w)
                        method = "copula"
                        confidence = 0.85

    # 3) appos: приложение к этнониму + прилагательное рядом
    for t in tokens:
        head_i = t.get("head_i")
        dep = t.get("dep", "")
        if head_i == eth_idx and dep == "appos":
            w = t.get("text", "")
            if t.get("pos") == "ADJ" and not _is_stop(w):
                epithets.append(w)
                if method == "fallback_window":
                    method = "appos"
                    confidence = 0.85
            # Или соседний ADJ к appos
            for t2 in tokens:
                if t2.get("head_i") == t.get("i") and t2.get("pos") == "ADJ":
                    w2 = t2.get("text", "")
                    if not _is_stop(w2):
                        epithets.append(w2)
                        method = "appos"
                        confidence = 0.85

    if epithets:
        epithets = list(dict.fromkeys(epithets))
        return (epithets, method, confidence)

    # 4) Fallback: окно ±5 токенов, только ADJ
    start = max(0, eth_idx - 5)
    end = min(len(tokens), eth_idx + 6)
    for j in range(start, end):
        if j == eth_idx:
            continue
        t = tokens[j]
        if t.get("pos") == "ADJ":
            w = t.get("text", "")
            if not _is_stop(w):
                epithets.append(w)
    if epithets:
        epithets = list(dict.fromkeys(epithets))
        return (epithets, "fallback_window", 0.4)

    return ([], "fallback_window", 0.4)


def extract_epithets_for_mention(
    sentence: Dict[str, Any],
    ethnonym_match_text: str,
) -> Dict[str, Any]:
    """
    Возвращает dict: epithets_list, epithet_extraction_method, epithet_confidence.
    """
    epithets, method, conf = extract_epithets_dependency(sentence, ethnonym_match_text)
    return {
        "epithets": epithets,
        "epithet_extraction_method": method,
        "epithet_confidence": conf,
    }
