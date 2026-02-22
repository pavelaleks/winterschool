"""
Кэш предобработанного корпуса в SQLite.
Сохраняет результат предобработки (документы + предложения с токенами) для быстрого повторного запуска.
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "output" / "corpus.db"


def _get_connection(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            raw_text TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sentences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL,
            sentence_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            start_char INTEGER,
            end_char INTEGER,
            tokens_json TEXT,
            pos_tags_json TEXT,
            deps_json TEXT,
            token_objects_json TEXT,
            FOREIGN KEY (doc_id) REFERENCES documents(id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sentences_doc_id ON sentences(doc_id)")
    conn.commit()


def save_corpus(corpus: List[Dict[str, Any]], db_path: Optional[Path] = None) -> str:
    """
    Сохраняет предобработанный корпус в SQLite.
    corpus: список dict с ключами filename, sentences, raw_text (как из preprocess_corpus).
    Возвращает путь к файлу БД.
    """
    db_path = db_path or DEFAULT_DB_PATH
    conn = _get_connection(db_path)
    _create_tables(conn)
    conn.execute("DELETE FROM sentences")
    conn.execute("DELETE FROM documents")
    for doc in corpus:
        cur = conn.execute(
            "INSERT INTO documents (filename, raw_text) VALUES (?, ?)",
            (doc.get("filename", ""), doc.get("raw_text", "")),
        )
        doc_id = cur.lastrowid
        for sent in doc.get("sentences", []):
            conn.execute(
                """INSERT INTO sentences (doc_id, sentence_index, text, start_char, end_char,
                   tokens_json, pos_tags_json, deps_json, token_objects_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    doc_id,
                    sent.get("sentence_index", 0),
                    sent.get("text", ""),
                    sent.get("start_char"),
                    sent.get("end_char"),
                    json.dumps(sent.get("tokens", [])),
                    json.dumps(sent.get("pos_tags", [])),
                    json.dumps(sent.get("deps", [])),
                    json.dumps(sent.get("token_objects", [])),
                ),
            )
    conn.commit()
    conn.close()
    return str(db_path)


def load_corpus(db_path: Optional[Path] = None) -> Optional[List[Dict[str, Any]]]:
    """
    Загружает корпус из SQLite. Возвращает тот же формат, что и preprocess_corpus.
    Если БД нет или пуста — возвращает None.
    """
    db_path = Path(db_path or DEFAULT_DB_PATH)
    if not db_path.exists():
        return None
    conn = _get_connection(db_path)
    try:
        docs_rows = conn.execute("SELECT id, filename, raw_text FROM documents ORDER BY id").fetchall()
        if not docs_rows:
            return None
        corpus = []
        for row in docs_rows:
            doc_id, filename, raw_text = row["id"], row["filename"], row["raw_text"] or ""
            sent_rows = conn.execute(
                "SELECT sentence_index, text, start_char, end_char, tokens_json, pos_tags_json, deps_json, token_objects_json FROM sentences WHERE doc_id = ? ORDER BY sentence_index",
                (doc_id,),
            ).fetchall()
            sentences = []
            for s in sent_rows:
                sentences.append({
                    "sentence_index": s["sentence_index"],
                    "text": s["text"] or "",
                    "start_char": s["start_char"],
                    "end_char": s["end_char"],
                    "tokens": json.loads(s["tokens_json"] or "[]"),
                    "pos_tags": json.loads(s["pos_tags_json"] or "[]"),
                    "deps": json.loads(s["deps_json"] or "[]"),
                    "token_objects": json.loads(s["token_objects_json"] or "[]"),
                })
            corpus.append({
                "filename": filename,
                "sentences": sentences,
                "raw_text": raw_text,
            })
        return corpus
    except (sqlite3.OperationalError, json.JSONDecodeError):
        return None
    finally:
        conn.close()
