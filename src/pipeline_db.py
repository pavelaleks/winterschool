"""
Кэш результатов пайплайна аналитики в SQLite.
Позволяет загрузить raw_df, clean_df, piro_*, keyness, norm_*, сети, derived_indices
без пересчёта (флаг --from-db в main.py).
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "output" / "pipeline.db"
PIPELINE_VERSION = 1


def _get_connection(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_meta (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            version INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_artifacts (
            name TEXT PRIMARY KEY,
            data_json TEXT NOT NULL
        )
    """)
    conn.commit()


def save_pipeline(
    corpus: List[Dict[str, Any]],
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    piro_raw: List[Dict],
    piro_clean: List[Dict],
    norm_ethnos: Dict,
    norm_R: Dict,
    norm_O: Dict,
    keyness_tables: Dict[str, pd.DataFrame],
    essentialization_table: Dict[str, int],
    essentialization_examples: Dict[str, List[Dict]],
    interaction_edges: List[Dict],
    comention_raw: Dict,
    comention_npmi: Optional[Dict] = None,
    robust_interaction_edges: Optional[List[Dict]] = None,
    interaction_node_metrics: Optional[List[Dict]] = None,
    interaction_node_metrics_robust: Optional[List[Dict]] = None,
    interaction_graph_metrics: Optional[Dict] = None,
    interaction_graph_metrics_robust: Optional[Dict] = None,
    comention_window_sensitivity: Optional[Dict] = None,
    interaction_edge_stability: Optional[List[Dict]] = None,
    derived_indices: Optional[Dict] = None,
    db_path: Optional[Path] = None,
) -> str:
    """
    Сохраняет результаты пайплайна (шаги 4–8b) в SQLite.
    Возвращает путь к БД.
    """
    from datetime import datetime
    db_path = db_path or DEFAULT_DB_PATH
    conn = _get_connection(db_path)
    _create_tables(conn)

    def _put(name: str, obj: Any) -> None:
        if obj is None:
            return
        if isinstance(obj, pd.DataFrame):
            data = obj.to_json(orient="split", date_format="iso", force_ascii=False)
        else:
            data = json.dumps(obj, ensure_ascii=False, default=str)
        conn.execute(
            "INSERT OR REPLACE INTO pipeline_artifacts (name, data_json) VALUES (?, ?)",
            (name, data),
        )

    _put("corpus", corpus)
    _put("raw_df", raw_df)
    _put("clean_df", clean_df)
    _put("piro_raw", piro_raw)
    _put("piro_clean", piro_clean)
    _put("norm_ethnos", norm_ethnos)
    _put("norm_R", norm_R)
    _put("norm_O", norm_O)
    _put("essentialization_table", essentialization_table)
    _put("essentialization_examples", essentialization_examples)
    _put("interaction_edges", interaction_edges)
    _put("comention_raw", comention_raw)
    if comention_npmi is not None:
        _put("comention_npmi", comention_npmi)
    if robust_interaction_edges is not None:
        _put("robust_interaction_edges", robust_interaction_edges)
    if interaction_node_metrics is not None:
        _put("interaction_node_metrics", interaction_node_metrics)
    if interaction_node_metrics_robust is not None:
        _put("interaction_node_metrics_robust", interaction_node_metrics_robust)
    if interaction_graph_metrics is not None:
        _put("interaction_graph_metrics", interaction_graph_metrics)
    if interaction_graph_metrics_robust is not None:
        _put("interaction_graph_metrics_robust", interaction_graph_metrics_robust)
    if comention_window_sensitivity is not None:
        _put("comention_window_sensitivity", comention_window_sensitivity)
    if interaction_edge_stability is not None:
        _put("interaction_edge_stability", interaction_edge_stability)
    if derived_indices is not None:
        _put("derived_indices", derived_indices)

    # keyness_tables: dict of name -> DataFrame; store as dict of name -> list of rows
    keyness_serializable = {}
    for k, df in (keyness_tables or {}).items():
        if df is not None and not df.empty:
            keyness_serializable[k] = df.to_dict(orient="records")
    _put("keyness_tables", keyness_serializable)

    conn.execute(
        "INSERT OR REPLACE INTO pipeline_meta (id, version, updated_at) VALUES (1, ?, ?)",
        (PIPELINE_VERSION, datetime.utcnow().isoformat() + "Z"),
    )
    conn.commit()
    conn.close()
    return str(db_path)


def load_pipeline(db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    Загружает кэш пайплайна из SQLite.
    Возвращает словарь с ключами: corpus, raw_df, clean_df, piro_raw, piro_clean,
    norm_ethnos, norm_R, norm_O, keyness_tables, essentialization_table,
    essentialization_examples, interaction_edges, comention_raw, derived_indices.
    Если БД нет или нет обязательных артефактов — возвращает None.
    """
    db_path = Path(db_path or DEFAULT_DB_PATH)
    if not db_path.exists():
        return None
    conn = _get_connection(db_path)
    try:
        row = conn.execute("SELECT version, updated_at FROM pipeline_meta WHERE id = 1").fetchone()
        if not row or row["version"] != PIPELINE_VERSION:
            return None

        def _get(name: str) -> Any:
            r = conn.execute("SELECT data_json FROM pipeline_artifacts WHERE name = ?", (name,)).fetchone()
            if not r:
                return None
            data = r["data_json"]
            if name in ("raw_df", "clean_df"):
                return pd.read_json(data, orient="split") if data else None
            if name == "keyness_tables":
                d = json.loads(data) if data else {}
                return {k: pd.DataFrame(v) for k, v in d.items() if v}
            return json.loads(data) if data else None

        corpus = _get("corpus")
        raw_df = _get("raw_df")
        clean_df = _get("clean_df")
        if corpus is None or raw_df is None or clean_df is None:
            return None

        out = {
            "corpus": corpus,
            "raw_df": raw_df,
            "clean_df": clean_df,
            "piro_raw": _get("piro_raw") or [],
            "piro_clean": _get("piro_clean") or [],
            "norm_ethnos": _get("norm_ethnos") or {},
            "norm_R": _get("norm_R") or {},
            "norm_O": _get("norm_O") or {},
            "keyness_tables": _get("keyness_tables") or {},
            "essentialization_table": _get("essentialization_table") or {},
            "essentialization_examples": _get("essentialization_examples") or {},
            "interaction_edges": _get("interaction_edges") or [],
            "comention_raw": _get("comention_raw") or {},
            "comention_npmi": _get("comention_npmi") or {},
            "robust_interaction_edges": _get("robust_interaction_edges") or [],
            "interaction_node_metrics": _get("interaction_node_metrics") or [],
            "interaction_node_metrics_robust": _get("interaction_node_metrics_robust") or [],
            "interaction_graph_metrics": _get("interaction_graph_metrics") or {},
            "interaction_graph_metrics_robust": _get("interaction_graph_metrics_robust") or {},
            "comention_window_sensitivity": _get("comention_window_sensitivity") or {},
            "interaction_edge_stability": _get("interaction_edge_stability") or [],
            "derived_indices": _get("derived_indices"),
        }
        return out
    except (sqlite3.OperationalError, json.JSONDecodeError, Exception):
        return None
    finally:
        conn.close()
