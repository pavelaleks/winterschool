"""
PIRO-модуль: для каждого упоминания этнонима — контекст ±4 предложения,
эпитеты, R (тип репрезентации), O (O_situation + O_metadata).
Поддержка построения из mentions_df с новыми полями (R_confidence, O_confidence, essentialization).
"""

import re
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from .ethnonym_extractor import (
    load_ethnonyms,
    build_ethnonym_patterns,
    extract_ethnonym_mentions,
    map_mentions_to_sentences,
)
from .situation_classifier import (
    load_situation_domains,
    get_context_text,
    classify_situation,
    classify_situation_full,
)
from .epithet_extractor import extract_epithets_for_mention
from .representation_classifier import classify_representation
from .essentialization import detect_essentialization_in_sentence


def load_sentiment_lexicon() -> Dict[str, List[str]]:
    """Загружает sentiment_lexicon.yml."""
    path = Path(__file__).resolve().parent.parent / "resources" / "sentiment_lexicon.yml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _classify_representation(epithets: List[str], context_lower: str, lexicon: Dict[str, List[str]]) -> str:
    """
    Классификация R: negative / exotic / positive / neutral.
    Учитываем эпитеты и слова из контекста по лексикону.
    При конфликте приоритет: negative > exotic > positive > neutral.
    """
    neg_set = set(w.lower() for w in (lexicon.get("negative") or []))
    exo_set = set(w.lower() for w in (lexicon.get("exotic") or []))
    pos_set = set(w.lower() for w in (lexicon.get("positive") or []))
    neu_set = set(w.lower() for w in (lexicon.get("neutral") or []))

    tokens = set(re.findall(r"[a-z]+", context_lower))
    all_words = tokens | set(e.lower() for e in epithets)

    has_neg = bool(all_words & neg_set)
    has_exo = bool(all_words & exo_set)
    has_pos = bool(all_words & pos_set)
    has_neu = bool(all_words & neu_set)

    if has_neg:
        return "negative"
    if has_exo:
        return "exotic"
    if has_pos:
        return "positive"
    if has_neu:
        return "neutral"
    return "neutral"


def extract_epithets(
    sentence: Dict,
    ethnonym_match_text: str,
    window_words: int = 5,
) -> List[str]:
    """
    Извлекает эпитеты: прилагательные в радиусе ±window_words от этнонима,
    плюс атрибутивные/предикативные конструкции ("X are Y").
    """
    tokens = sentence.get("token_objects", [])
    if not tokens:
        text = sentence.get("text", "")
        words = text.split()
        pos_tags = []
        for i, w in enumerate(words):
            if w.lower() == ethnonym_match_text.lower():
                start = max(0, i - window_words)
                end = min(len(words), i + window_words + 1)
                adj = [words[j] for j in range(start, end) if words[j].lower() != ethnonym_match_text.lower()]
                return list(set(adj))
        return []

    # Найти индекс токена этнонима
    eth_idx = None
    for i, t in enumerate(tokens):
        if t["text"].lower() == ethnonym_match_text.lower():
            eth_idx = i
            break
    if eth_idx is None:
        return []

    adjectives = []
    start = max(0, eth_idx - window_words)
    end = min(len(tokens), eth_idx + window_words + 1)
    for j in range(start, end):
        if tokens[j]["pos"] == "ADJ":
            adjectives.append(tokens[j]["text"])
        if tokens[j]["pos"] == "ADV" and j + 1 < len(tokens) and tokens[j + 1]["pos"] == "ADJ":
            adjectives.append(tokens[j]["text"] + " " + tokens[j + 1]["text"])

    # Предикативные "X are Y" / "The X are Y"
    sent_text = sentence.get("text", "")
    eth_lower = ethnonym_match_text.lower()
    # "Yakuts are hardy" -> hardy; "The Tungus are wild" -> wild
    for m in re.finditer(r"(?:the\s+)?(\w+)\s+are\s+(\w+)", sent_text, re.IGNORECASE):
        if m.group(1).lower() == eth_lower and m.group(2).lower() not in ("the", "a", "an"):
            adjectives.append(m.group(2))
    for m in re.finditer(r"(?:the\s+)?(\w+)\s+is\s+(\w+)", sent_text, re.IGNORECASE):
        if m.group(1).lower() == eth_lower and m.group(2).lower() not in ("the", "a", "an"):
            adjectives.append(m.group(2))
    return list(set(adjectives))


def run_piro_on_document(
    doc: Dict[str, Any],
    patterns: List[tuple],
    sentiment_lexicon: Dict[str, List[str]],
    situation_domains: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """
    Запускает PIRO для одного документа (из preprocess: filename, sentences, raw_text).
    Возвращает список записей PIRO для каждого упоминания этнонима.
    """
    filename = doc["filename"]
    sentences = doc["sentences"]
    raw_text = doc["raw_text"]

    mentions = extract_ethnonym_mentions(raw_text, patterns)
    mentions = map_mentions_to_sentences(mentions, sentences, doc_start_offset=0)

    results = []
    for m in mentions:
        sent_idx = m.get("sentence_index")
        if sent_idx is None:
            continue
        context_text = get_context_text(sentences, sent_idx, window=4)
        O_situation = classify_situation(context_text, situation_domains)
        O_metadata = {"file": filename, "sentence_index": sent_idx}

        center_sent = sentences[sent_idx]
        epithets = extract_epithets(center_sent, m["match_text"], window_words=5)
        R = _classify_representation(epithets, context_text.lower(), sentiment_lexicon)

        results.append({
            "P": m["ethnonym"],
            "epithets": epithets,
            "R": R,
            "O_situation": O_situation,
            "O_metadata": O_metadata,
            "context": context_text,
            "mention": m["match_text"],
        })
    return results


def run_piro_on_corpus(corpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Запускает PIRO по всему корпусу (результат preprocess_corpus).
    Возвращает плоский список всех PIRO-записей.
    """
    ethnonyms = load_ethnonyms()
    patterns = build_ethnonym_patterns(ethnonyms)
    sentiment_lexicon = load_sentiment_lexicon()
    situation_domains = load_situation_domains()

    all_piro = []
    for doc in corpus:
        all_piro.extend(
            run_piro_on_document(doc, patterns, sentiment_lexicon, situation_domains)
        )
    return all_piro


def build_piro_from_mentions_df(
    mentions_df: pd.DataFrame,
    ethnonym_forms: Optional[Dict[str, List[str]]] = None,
    use_lemmas: bool = False,
) -> List[Dict[str, Any]]:
    """
    Строит список PIRO-записей из DataFrame упоминаний (после noise_filter).
    use_lemmas: при True классификация R и O идёт по леммам контекста (и лексиконы приводятся к леммам).
    """
    ethnonym_forms = ethnonym_forms or load_ethnonyms()
    if use_lemmas:
        from .situation_classifier import load_situation_domains
        situation_domains = load_situation_domains(lemmatize=True)
    else:
        situation_domains = None
    records = []
    for _, row in mentions_df.iterrows():
        sentence = {
            "text": row.get("sentence_text", ""),
            "token_objects": row.get("token_objects", []) or [],
        }
        ethnos_raw = row.get("ethnos_raw", "")
        ethnos_norm = row.get("ethnos_norm", "")
        context_text = row.get("context_text", "")

        token_objects = sentence.get("token_objects") or []
        context_lemmas = None
        if use_lemmas and token_objects:
            context_lemmas = [
                (t.get("lemma") or t.get("text", "") or "").lower().strip()
                for t in token_objects
            ]

        epithet_result = extract_epithets_for_mention(sentence, ethnos_raw)
        epithets = epithet_result.get("epithets", [])
        epithet_method = epithet_result.get("epithet_extraction_method", "fallback_window")
        epithet_conf = epithet_result.get("epithet_confidence", 0.4)

        R_result = classify_representation(
            context_text,
            epithets,
            epithet_confidence=epithet_conf,
            context_lemmas=context_lemmas,
            use_lemma_lexicons=use_lemmas,
        )
        O_result = classify_situation_full(
            context_text,
            domains=situation_domains,
            context_lemmas=context_lemmas,
        )
        ess = detect_essentialization_in_sentence(
            row.get("sentence_text", ""),
            ethnos_norm,
            all_ethnonym_forms=ethnonym_forms,
        )

        token_count = len(sentence.get("token_objects") or [])

        rec = {
            "mention_id": row.get("mention_id"),
            "file_name": row.get("file_name", ""),
            "doc_id": row.get("doc_id"),
            "sent_idx": row.get("sent_idx"),
            "doc_position_percent": row.get("doc_position_percent"),
            "ethnos_raw": ethnos_raw,
            "P": ethnos_norm,
            "ethnos_norm": ethnos_norm,
            "sentence_text": row.get("sentence_text", ""),
            "context_text": context_text,
            "context": context_text,
            "epithets": epithets,
            "epithet_extraction_method": epithet_method,
            "epithet_confidence": epithet_conf,
            "R": R_result.get("R", "neutral"),
            "R_scores": R_result.get("R_scores", {}),
            "R_confidence": R_result.get("R_confidence", 0.3),
            "O_situation": O_result.get("O_situation", "unknown"),
            "O_scores": O_result.get("O_scores", {}),
            "O_confidence": O_result.get("O_confidence", 0.2),
            "O_metadata": {"file": row.get("file_name", ""), "sentence_index": row.get("sent_idx")},
            "is_essentializing": ess.get("is_essentializing", False),
            "essentialization_pattern": ess.get("essentialization_pattern", ""),
            "essentialization_span": ess.get("essentialization_span", ""),
            "is_probable_header_footer": row.get("is_probable_header_footer", False),
            "is_noise": row.get("is_noise", False),
            "noise_reason": row.get("noise_reason", ""),
            "position_bucket": row.get("position_bucket", ""),
            "position_bucket_pct": row.get("position_bucket_pct"),
            "is_position_header_footer": row.get("is_position_header_footer", False),
            "is_cross_doc_header_footer": row.get("is_cross_doc_header_footer", False),
            "token_count": token_count,
        }
        records.append(rec)
    return records
