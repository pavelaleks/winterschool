"""
Экспорт полной базы PIRO в output/piro_full_database.xlsx:
листы PIRO_clean, PIRO_raw, Summary_stats, Keyness_tables, Validation.
Колонки с global_confidence_score и всеми полями для ручной валидации.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd


def global_confidence_score(row: Dict) -> float:
    """
    +0.35 epithet dependency-based (method amod/copula/appos)
    +0.20 R_confidence >= 0.6
    +0.20 O_confidence >= 0.6
    +0.15 is_essentializing
    +0.10 interaction detected
    cap at 1.0
    """
    s = 0.0
    method = (row.get("epithet_extraction_method") or "").lower()
    if method in ("amod", "copula", "appos"):
        s += 0.35
    if (row.get("R_confidence") or 0) >= 0.6:
        s += 0.20
    if (row.get("O_confidence") or 0) >= 0.6:
        s += 0.20
    if row.get("is_essentializing"):
        s += 0.15
    if row.get("interaction_type"):
        s += 0.10
    return min(1.0, round(s, 2))


def piro_records_to_dataframe(
    records: List[Dict],
    include_global_confidence: bool = True,
) -> pd.DataFrame:
    """Преобразует список PIRO-записей в DataFrame с нужными колонками."""
    rows = []
    for r in records:
        row = {
            "mention_id": r.get("mention_id"),
            "file_name": r.get("file_name") or (r.get("O_metadata") or {}).get("file", ""),
            "doc_id": r.get("doc_id"),
            "sent_idx": r.get("sent_idx") or (r.get("O_metadata") or {}).get("sentence_index"),
            "doc_position_percent": r.get("doc_position_percent"),
            "ethnos_raw": r.get("ethnos_raw") or r.get("mention", ""),
            "ethnos_norm": r.get("P") or r.get("ethnos_norm", ""),
            "sentence_text": r.get("sentence_text") or "",
            "context_text": r.get("context_text") or r.get("context", ""),
            "epithets": str(r.get("epithets", []))[:500],
            "epithet_extraction_method": r.get("epithet_extraction_method", ""),
            "R": r.get("R", ""),
            "R_scores": str(r.get("R_scores", "")),
            "R_confidence": r.get("R_confidence"),
            "O_situation": r.get("O_situation", ""),
            "O_scores": str(r.get("O_scores", "")),
            "O_confidence": r.get("O_confidence"),
            "is_essentializing": r.get("is_essentializing", False),
            "essentialization_pattern": r.get("essentialization_pattern", ""),
            "interaction_type": r.get("interaction_type", ""),
            "is_probable_header_footer": r.get("is_probable_header_footer", False),
            "token_count": r.get("token_count"),
        }
        if include_global_confidence:
            row["global_confidence_score"] = global_confidence_score(row)
        rows.append(row)
    return pd.DataFrame(rows)


def summary_stats_df(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    keyness_paths: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Лист Summary_stats: краткая сводка."""
    keyness_paths = keyness_paths or {}
    rows = [
        {"metric": "mentions_raw", "value": len(raw_df)},
        {"metric": "mentions_clean", "value": len(clean_df)},
        {"metric": "noise_removed", "value": len(raw_df) - len(clean_df)},
        {"metric": "keyness_files", "value": "; ".join(keyness_paths.keys())},
    ]
    return pd.DataFrame(rows)


def export_piro_full_database(
    piro_raw: List[Dict],
    piro_clean: List[Dict],
    keyness_tables: Optional[Dict[str, pd.DataFrame]] = None,
    output_path: Optional[Path] = None,
) -> str:
    """
    Создаёт output/piro_full_database.xlsx с листами:
    PIRO_clean, PIRO_raw, Summary_stats, Keyness_tables (врезка/ссылка), Validation (пустой для заметок).
    """
    output_path = output_path or Path(__file__).resolve().parent.parent / "output" / "piro_full_database.xlsx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_raw = piro_records_to_dataframe(piro_raw, include_global_confidence=True)
    df_clean = piro_records_to_dataframe(piro_clean, include_global_confidence=True)
    summary = summary_stats_df(df_raw, df_clean)
    keyness_tables = keyness_tables or {}
    validation_df = pd.DataFrame(columns=["mention_id", "validator_notes", "validated_R", "validated_O"])

    with pd.ExcelWriter(output_path, engine="openpyxl") as w:
        df_clean.to_excel(w, sheet_name="PIRO_clean", index=False)
        df_raw.to_excel(w, sheet_name="PIRO_raw", index=False)
        summary.to_excel(w, sheet_name="Summary_stats", index=False)
        for name, kdf in list(keyness_tables.items())[:5]:
            sheet_name = name[:31]
            kdf.head(50).to_excel(w, sheet_name=sheet_name, index=False)
        validation_df.to_excel(w, sheet_name="Validation", index=False)

    return str(output_path)
