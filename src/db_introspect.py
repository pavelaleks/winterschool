"""
Интроспекция SQLite корпуса и построение единого DataFrame упоминаний (mentions_df).
Определяет структуру БД, загружает документы и предложения, извлекает этнонимы и контексты.
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd

from .corpus_db import DEFAULT_DB_PATH, load_corpus
from .ethnonym_extractor import (
    load_ethnonyms,
    build_ethnonym_patterns,
    extract_ethnonym_mentions,
    extract_ethnonym_mentions_from_sentences,
    map_mentions_to_sentences,
)
from .situation_classifier import get_context_text


def get_db_schema(db_path: Optional[Path] = None) -> Dict[str, List[str]]:
    """Возвращает схему БД: {имя_таблицы: [колонки]}."""
    db_path = Path(db_path or DEFAULT_DB_PATH)
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(str(db_path))
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [row[0] for row in cur.fetchall()]
    schema = {}
    for t in tables:
        cur = conn.execute(f"PRAGMA table_info({t})")
        schema[t] = [row[1] for row in cur.fetchall()]
    conn.close()
    return schema


def _column_map_sentences(row_factory: Any) -> Dict[str, str]:
    """Маппинг имён колонок таблицы предложений на стандартные (sentence_index, text, ...)."""
    # Стандартные имена в corpus_db: sentence_index, text, start_char, end_char, tokens_json, ...
    aliases = {
        "sentence_index": ["sentence_index", "sent_idx", "sent_index", "idx"],
        "text": ["text", "sentence_text", "sentence", "content"],
        "doc_id": ["doc_id", "document_id", "docid"],
    }
    return {}  # Используем load_corpus, который уже знает схему


def build_mentions_df_from_corpus(
    corpus: List[Dict[str, Any]],
    patterns: Optional[List[Tuple[str, Any]]] = None,
    use_lemmas: bool = False,
) -> pd.DataFrame:
    """
    Строит единый DataFrame упоминаний из предобработанного корпуса.
    corpus: список dict с ключами filename, sentences, raw_text (как из load_corpus / preprocess).
    use_lemmas: если True, этнонимы ищутся по леммам токенов (token_objects[].lemma), иначе — regex по сырому тексту.
    """
    ethnonyms = load_ethnonyms()
    if patterns is None:
        patterns = build_ethnonym_patterns(ethnonyms)

    rows = []
    mention_id = 0
    for doc_idx, doc in enumerate(corpus):
        filename = doc.get("filename", "")
        sentences = doc.get("sentences", [])
        raw_text = doc.get("raw_text", "")
        doc_sent_count = len(sentences)

        if use_lemmas:
            mentions = extract_ethnonym_mentions_from_sentences(sentences, ethnonyms=ethnonyms)
        else:
            mentions = extract_ethnonym_mentions(raw_text, patterns)
            mentions = map_mentions_to_sentences(mentions, sentences, doc_start_offset=0)

        for m in mentions:
            sent_idx = m.get("sentence_index")
            if sent_idx is None:
                continue
            context_text = get_context_text(sentences, sent_idx, window=4)
            sentence_text = sentences[sent_idx].get("text", "") if sent_idx < len(sentences) else ""

            doc_position_percent = (
                (sent_idx / doc_sent_count * 100) if doc_sent_count else 0.0
            )

            mention_id += 1
            rows.append({
                "mention_id": mention_id,
                "doc_id": doc_idx,
                "file_name": filename,
                "sent_idx": sent_idx,
                "sentence_text": sentence_text,
                "context_text": context_text,
                "ethnos_raw": m.get("match_text", ""),
                "ethnos_norm": m.get("ethnonym", ""),
                "doc_sent_count": doc_sent_count,
                "doc_position_percent": round(doc_position_percent, 2),
                "sentence_tokens": sentences[sent_idx].get("tokens", []) if sent_idx < len(sentences) else [],
                "token_objects": sentences[sent_idx].get("token_objects", []) if sent_idx < len(sentences) else [],
            })

    if not rows:
        return pd.DataFrame(columns=[
            "mention_id", "doc_id", "file_name", "sent_idx", "sentence_text",
            "context_text", "ethnos_raw", "ethnos_norm", "doc_sent_count", "doc_position_percent",
        ])

    return pd.DataFrame(rows)


def load_mentions_df(
    db_path: Optional[Path] = None,
    corpus: Optional[List[Dict[str, Any]]] = None,
    use_lemmas: bool = False,
) -> pd.DataFrame:
    """
    Загружает корпус из SQLite (или принимает готовый corpus) и строит mentions_df.
    use_lemmas передаётся в build_mentions_df_from_corpus (по умолчанию False — поиск по словоформам).
    """
    if corpus is None:
        corpus = load_corpus(db_path)
    if not corpus:
        return pd.DataFrame()

    return build_mentions_df_from_corpus(corpus, use_lemmas=use_lemmas)


def introspect_and_load(
    db_path: Optional[Path] = None,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """
    Интроспекция БД и загрузка mentions_df.
    Возвращает (schema, mentions_df).
    """
    db_path = Path(db_path or DEFAULT_DB_PATH)
    schema = get_db_schema(db_path)
    mentions_df = load_mentions_df(db_path=db_path)
    return schema, mentions_df
