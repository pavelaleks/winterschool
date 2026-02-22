"""
Вспомогательные функции для scripts/translator_demo.py:
интроспекция БД, загрузка mentions_df, паттерны, лексиконы, шум-фильтр.
"""

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# --- Интроспекция БД ---

ETHNOS_COL_PATTERNS = ["ethnos", "ethnicity", "group", "people", "tribe", "ethnos_norm", "ethnos_raw"]
SENTENCE_COL_PATTERNS = ["sentence", "sent", "text", "context", "sentence_text", "context_text", "content"]
FILE_COL_PATTERNS = ["file", "doc", "source", "file_name", "filename", "doc_id", "document_id"]
SENT_IDX_PATTERNS = ["sent_idx", "sentence_index", "sent_index", "idx", "sentence_index"]


def _col_matches(col_lower: str, patterns: List[str]) -> bool:
    return any(p in col_lower for p in patterns)


def introspect_db(conn: sqlite3.Connection) -> Dict[str, Any]:
    """
    Определяет структуру БД и кандидата на таблицу mentions/contexts.
    Возвращает mapping для загрузки mentions_df.
    """
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cur.fetchall()]
    if not tables:
        raise RuntimeError("В БД нет таблиц. Проверьте путь к corpus.db.")

    table_columns: Dict[str, List[str]] = {}
    for t in tables:
        cur = conn.execute(f"PRAGMA table_info({t})")
        table_columns[t] = [row[1] for row in cur.fetchall()]

    # Ищем одну таблицу с колонками ethnos + sentence/text + file/doc
    for tname, cols in table_columns.items():
        c_lower = [c.lower() for c in cols]
        has_ethnos = any(_col_matches(c, ETHNOS_COL_PATTERNS) for c in c_lower)
        has_sent = any(_col_matches(c, SENTENCE_COL_PATTERNS) for c in c_lower)
        has_file = any(_col_matches(c, FILE_COL_PATTERNS) for c in c_lower)
        if has_ethnos and has_sent:
            col_map = _build_col_map(cols, c_lower)
            if col_map.get("ethnos") and col_map.get("sentence"):
                return {
                    "mentions_table": tname,
                    "schema": "single_table",
                    "cols": col_map,
                    "table_columns": {tname: cols},
                }

    # Fallback: documents + sentences (как в corpus_db)
    if "documents" in table_columns and "sentences" in table_columns:
        doc_cols = table_columns["documents"]
        sent_cols = table_columns["sentences"]
        doc_lower = [c.lower() for c in doc_cols]
        sent_lower = [c.lower() for c in sent_cols]
        return {
            "mentions_table": None,
            "schema": "documents_sentences",
            "cols": {
                "doc_id": next((c for c in sent_cols if "doc" in c.lower() and c.lower() != "document_id"), "doc_id"),
                "filename": next((c for c in doc_cols if "filename" in c.lower() or "file" in c.lower()), doc_cols[0] if doc_cols else None),
                "sentence_index": next((c for c in sent_cols if "sentence_index" in c.lower() or "sent_idx" in c.lower() or c.lower() == "idx"), "sentence_index"),
                "text": next((c for c in sent_cols if c.lower() in ("text", "sentence", "content")), "text"),
            },
            "table_columns": table_columns,
        }

    msg = (
        "Не найдена таблица с колонками для упоминаний (ethnos/ethnicity/group + sentence/text + file/doc). "
        f"Доступные таблицы: {list(table_columns.keys())}. "
        "Для формата corpus.db ожидаются таблицы 'documents' и 'sentences'."
    )
    raise RuntimeError(msg)


def _build_col_map(cols: List[str], c_lower: List[str]) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {"ethnos": None, "sentence": None, "context": None, "file": None, "sent_idx": None}
    for i, c in enumerate(c_lower):
        if not out["ethnos"] and _col_matches(c, ETHNOS_COL_PATTERNS):
            out["ethnos"] = cols[i]
        if not out["sentence"] and _col_matches(c, SENTENCE_COL_PATTERNS):
            if "context" in c:
                out["context"] = cols[i]
            else:
                out["sentence"] = cols[i]
        if not out["file"] and _col_matches(c, FILE_COL_PATTERNS):
            out["file"] = cols[i]
        if not out["sent_idx"] and _col_matches(c, SENT_IDX_PATTERNS):
            out["sent_idx"] = cols[i]
    if out["context"] is None and out["sentence"]:
        out["context"] = out["sentence"]
    return out


def load_ethnonyms_yml(resources_dir: Optional[Path] = None) -> List[Tuple[str, str]]:
    """Загружает (variant_lower, canonical) из resources/ethnonyms.yml."""
    if resources_dir is None:
        resources_dir = Path(__file__).resolve().parent.parent / "resources"
    path = resources_dir / "ethnonyms.yml"
    if not path.exists():
        return []
    try:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not data or not isinstance(data, dict):
        return []
    pairs: List[Tuple[str, str]] = []
    for canonical, variants in data.items():
        if not isinstance(variants, list):
            continue
        for v in variants:
            if isinstance(v, str) and v.strip():
                pairs.append((v.strip().lower(), canonical))
    return pairs


def load_mentions_from_conn(
    conn: sqlite3.Connection,
    mapping: Dict[str, Any],
    ethnonyms_pairs: List[Tuple[str, str]],
    project_root: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Строит DataFrame с колонками file_name, sent_idx, sentence_text, context_text, ethnos_norm.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    if mapping.get("schema") == "single_table" and mapping.get("mentions_table"):
        tname = mapping["mentions_table"]
        cols = mapping["cols"]
        ethnos_col = cols.get("ethnos") or "ethnos_norm"
        sent_col = cols.get("sentence") or "sentence_text"
        ctx_col = cols.get("context") or cols.get("sentence")
        file_col = cols.get("file") or "file_name"
        idx_col = cols.get("sent_idx") or "sent_idx"
        sel = [c for c in [file_col, idx_col, sent_col, ctx_col, ethnos_col] if c]
        sel = list(dict.fromkeys(sel))
        query = f"SELECT {', '.join(sel)} FROM {tname}"
        df = pd.read_sql_query(query, conn)
        rename = {}
        if file_col and file_col != "file_name":
            rename[file_col] = "file_name"
        if idx_col and idx_col != "sent_idx":
            rename[idx_col] = "sent_idx"
        if sent_col and sent_col != "sentence_text":
            rename[sent_col] = "sentence_text"
        if ctx_col and ctx_col != "context_text":
            rename[ctx_col] = "context_text"
        if ethnos_col and ethnos_col != "ethnos_norm":
            rename[ethnos_col] = "ethnos_norm"
        df = df.rename(columns=rename)
        if "context_text" not in df.columns and "sentence_text" in df.columns:
            df["context_text"] = df["sentence_text"]
        return df

    if mapping.get("schema") == "documents_sentences":
        # JOIN documents + sentences, затем ищем этнонимы в text
        doc_id_col = mapping["cols"].get("doc_id") or "doc_id"
        filename_col = mapping["cols"].get("filename") or "filename"
        sent_idx_col = mapping["cols"].get("sentence_index") or "sentence_index"
        text_col = mapping["cols"].get("text") or "text"
        q = (
            f"SELECT d.{filename_col} AS file_name, s.{sent_idx_col} AS sent_idx, s.{text_col} AS sentence_text "
            f"FROM sentences s JOIN documents d ON d.id = s.{doc_id_col} ORDER BY d.id, s.{sent_idx_col}"
        )
        try:
            df_full = pd.read_sql_query(q, conn)
        except sqlite3.OperationalError:
            q2 = (
                f"SELECT documents.{filename_col} AS file_name, sentences.{sent_idx_col} AS sent_idx, "
                f"sentences.{text_col} AS sentence_text FROM sentences JOIN documents ON documents.id = sentences.{doc_id_col}"
            )
            df_full = pd.read_sql_query(q2, conn)
        df_full["context_text"] = df_full["sentence_text"].fillna("")
        df_full["ethnos_norm"] = ""

        if not ethnonyms_pairs:
            ethnonyms_pairs = load_ethnonyms_yml(project_root / "resources")

        # Уникальные канонические имена и все варианты для regex
        canon_to_variants: Dict[str, List[str]] = {}
        for var, can in ethnonyms_pairs:
            canon_to_variants.setdefault(can, []).append(re.escape(var))
        pattern_by_canonical: Dict[str, re.Pattern] = {}
        for can, vars_list in canon_to_variants.items():
            pattern_by_canonical[can] = re.compile(r"\b(?:" + "|".join(vars_list) + r")\b", re.I)

        rows = []
        for _, row in df_full.iterrows():
            text = (row.get("sentence_text") or "") if isinstance(row.get("sentence_text"), str) else ""
            for canon, pat in pattern_by_canonical.items():
                if pat.search(text):
                    rows.append({
                        "file_name": row.get("file_name"),
                        "sent_idx": row.get("sent_idx"),
                        "sentence_text": text,
                        "context_text": row.get("context_text") or text,
                        "ethnos_norm": canon,
                    })
                    break
        if not rows:
            return pd.DataFrame(columns=["file_name", "sent_idx", "sentence_text", "context_text", "ethnos_norm"])
        return pd.DataFrame(rows)

    return pd.DataFrame(columns=["file_name", "sent_idx", "sentence_text", "context_text", "ethnos_norm"])


def add_context_from_adjacent(conn: sqlite3.Connection, df: pd.DataFrame, mapping: Dict[str, Any]) -> pd.DataFrame:
    """Для schema documents_sentences добавляет context_text как ±2 предложения."""
    if mapping.get("schema") != "documents_sentences" or "context_text" in df.columns and df["context_text"].str.len().gt(50).any():
        return df
    doc_id_col = mapping["cols"].get("doc_id") or "doc_id"
    text_col = mapping["cols"].get("text") or "text"
    sent_idx_col = mapping["cols"].get("sentence_index") or "sentence_index"
    q = f"SELECT doc_id, {sent_idx_col}, {text_col} FROM sentences ORDER BY doc_id, {sent_idx_col}"
    try:
        all_sents = pd.read_sql_query(q, conn)
    except Exception:
        return df
    all_sents.columns = ["doc_id", "sent_idx", "text"]
    contexts = []
    for (fname, sidx), group in df.groupby(["file_name", "sent_idx"]):
        doc_sents = all_sents[all_sents["doc_id"] == all_sents["doc_id"].iloc[0]] if len(all_sents) else all_sents
        if "file_name" in df.columns and "doc_id" not in df.columns:
            doc_sents = all_sents
        idx = all_sents[(all_sents["sent_idx"] == sidx)].index
        if len(idx) == 0:
            contexts.append((fname, sidx, ""))
            continue
        i = all_sents.index.get_loc(idx[0])
        start = max(0, i - 2)
        end = min(len(all_sents), i + 3)
        ctx = " ".join(all_sents.iloc[start:end]["text"].astype(str).tolist())
        contexts.append((fname, sidx, ctx))
    ctx_df = pd.DataFrame(contexts, columns=["file_name", "sent_idx", "context_text"])
    df = df.drop(columns=["context_text"], errors="ignore")
    df = df.merge(ctx_df, on=["file_name", "sent_idx"], how="left")
    if "context_text" not in df.columns:
        df["context_text"] = df["sentence_text"]
    return df


# --- Шум для демо ---

NOISE_TOC_PATTERN = re.compile(r"\b(contents?|index|chapter|page\s*\d+|part\s+[ivxlcdm]+)\b", re.I)
LIST_LIKE_PATTERN = re.compile(r",\s*\d+\s*$")  # "..., 66"


def is_noise_demo(sentence_text: str) -> bool:
    """True если строка похожа на колонтитул/оглавление/мусор."""
    if not sentence_text or not isinstance(sentence_text, str):
        return True
    s = sentence_text.strip()
    if len(s) < 20:
        return True
    words = s.split()
    if len(words) < 5:
        return True
    digit_ratio = sum(1 for c in s if c.isdigit()) / max(len(s), 1)
    if digit_ratio > 0.3:
        return True
    punct_ratio = sum(1 for c in s if c in ".,;:!?") / max(len(s), 1)
    if punct_ratio > 0.25:
        return True
    if NOISE_TOC_PATTERN.search(s):
        return True
    if LIST_LIKE_PATTERN.search(s):
        return True
    return False


# --- Паттерны эссенциализации ---

ESSENTIALIZATION_PATTERNS = [
    (r"\bthe\s+(\w+)\s+are\b", "the X are"),
    (r"\b(\w+)\s+are\s+", "X are"),
    (r"\ball\s+(\w+)\s+", "all X"),
    (r"\b(\w+)\s+were\s+known\s+to\b", "X were known to"),
    (r"\bby\s+nature\b", "by nature"),
    (r"\bas\s+a\s+(?:race|people|tribe)\b", "as a race/people/tribe"),
]


def detect_essentialization_pattern(text: str) -> Optional[Tuple[str, str]]:
    """Возвращает (pattern_name, matched_phrase) или None."""
    if not text:
        return None
    text_lower = text.lower()
    for regex, name in ESSENTIALIZATION_PATTERNS:
        m = re.search(regex, text_lower, re.I)
        if m:
            return (name, m.group(0))
    return None


# --- Лексиконы (в коде) ---

LANDSCAPE_LEXICON_LEMMAS = [
    "steppe", "taiga", "forest", "tundra", "wilderness", "barren", "desolate", "vast", "endless",
    "river", "snow", "frost", "savage", "bleak", "remote", "primitive",
    "wild", "mountain", "ice", "cold", "dreary", "uncultivated",
]

PITFALLS_LEMMAS = [
    "savage", "primitive", "simple", "childlike", "barbarous", "backward", "uncivilized",
    "superstitious", "filthy", "lazy", "cunning", "cruel", "wild",
]

IMPERIAL_AGENT_LEMMAS = [
    "russians", "government", "administration", "officials", "cossacks", "soldiers",
    "russian", "authority", "authorities", "colonists", "empire",
]


def load_landscape_lexicon_file(resources_dir: Optional[Path] = None) -> List[str]:
    """Дополняет LANDSCAPE_LEXICON_LEMMAS из resources/landscape_lexicon.txt если есть."""
    if resources_dir is None:
        resources_dir = Path(__file__).resolve().parent.parent / "resources"
    path = resources_dir / "landscape_lexicon.txt"
    if not path.exists():
        return list(LANDSCAPE_LEXICON_LEMMAS)
    extra = [line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return list(dict.fromkeys(LANDSCAPE_LEXICON_LEMMAS + extra))


def get_pitfalls_list() -> List[str]:
    return list(PITFALLS_LEMMAS)


def get_imperial_agent_list() -> List[str]:
    return list(IMPERIAL_AGENT_LEMMAS)


# --- Подсказки по переводу (rule-based) ---

PITFALL_SUGGESTIONS: Dict[str, str] = {
    "savage": "варианты: дикарь/дикий/варварский — различать уровень стигматизации; savage (nature) = суровый/дикий (о природе).",
    "primitive": "примитивный/первобытный/простой — коннотация развития vs нейтральное описание.",
    "simple": "простой/нехитрый/наивный — возможна патронизирующая тональность.",
    "childlike": "детский/по-детски/наивный — часто инфантилизация.",
    "barbarous": "варварский/жестокий/нецивилизованный.",
    "backward": "отсталый/запоздалый — осторожно с оценочностью.",
    "uncivilized": "нецивилизованный/дикий — контраст с «цивилизацией».",
    "superstitious": "суеверный — обычно нейтрально, контекст важен.",
    "filthy": "грязный/скверный — может быть стигматизация.",
    "lazy": "ленивый — стереотип, проверить контекст.",
    "cunning": "хитрый/лукавый — позитив/негатив по контексту.",
    "cruel": "жестокий — описание поступков vs обобщение.",
    "wild": "дикий/необузданный — о людях vs о природе.",
}


def get_pitfall_suggestion(word: str) -> str:
    w = word.lower()
    return PITFALL_SUGGESTIONS.get(w, f"Слово «{word}» — проверить коннотацию в контексте и варианты перевода.")
