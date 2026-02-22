"""
Evidence pack: для ручной валидации — топ контекстов по категориям (negative, exotic, positive, uncertain, essentialization).
Сохраняет output/evidence_pack.xlsx с листами и примерами.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd


def _shorten(s: str, max_len: int = 400) -> str:
    if not s or not isinstance(s, str):
        return ""
    s = s.strip()
    return s[:max_len] + "..." if len(s) > max_len else s


def build_evidence_dfs(
    piro_records: List[Dict],
    top_n: int = 10,
) -> Dict[str, pd.DataFrame]:
    """
    Строит по категории DataFrame с колонками:
    ethnos, R/O, confidence, sentence_text, context_text (укороченный), file_name, sent_idx, doc_position_percent.
    Категории: NEGATIVE_examples, EXOTIC_examples, POSITIVE_examples, UNCERTAIN_examples, ESSENTIALIZATION_examples.
    """
    # Приводим к единому формату полей
    def row_to_dict(r: Dict) -> Dict:
        return {
            "ethnos": r.get("P") or r.get("ethnos_norm", ""),
            "R": r.get("R", ""),
            "O_situation": r.get("O_situation", ""),
            "confidence": r.get("R_confidence") or r.get("O_confidence") or 0,
            "sentence_text": r.get("sentence_text") or r.get("sentence", ""),
            "context_text": _shorten(r.get("context_text") or r.get("context", ""), 400),
            "file_name": r.get("file_name") or (r.get("O_metadata") or {}).get("file", ""),
            "sent_idx": r.get("sent_idx") or (r.get("O_metadata") or {}).get("sentence_index", ""),
            "doc_position_percent": r.get("doc_position_percent"),
        }

    by_neg = [row_to_dict(r) for r in piro_records if r.get("R") == "negative"]
    by_neg.sort(key=lambda x: -(x.get("confidence") or 0))
    by_exo = [row_to_dict(r) for r in piro_records if r.get("R") == "exotic"]
    by_exo.sort(key=lambda x: -(x.get("confidence") or 0))
    by_pos = [row_to_dict(r) for r in piro_records if r.get("R") == "positive"]
    by_pos.sort(key=lambda x: -(x.get("confidence") or 0))
    by_unc = [row_to_dict(r) for r in piro_records if r.get("R") == "uncertain"]
    by_unc.sort(key=lambda x: -(x.get("confidence") or 0))
    by_ess = [row_to_dict(r) for r in piro_records if r.get("is_essentializing")]
    by_ess.sort(key=lambda x: -(x.get("confidence") or 0))

    def to_df(rows: List[Dict], n: int) -> pd.DataFrame:
        return pd.DataFrame(rows[:n]) if rows else pd.DataFrame()

    return {
        "NEGATIVE_examples": to_df(by_neg, top_n),
        "EXOTIC_examples": to_df(by_exo, top_n),
        "POSITIVE_examples": to_df(by_pos, top_n),
        "UNCERTAIN_examples": to_df(by_unc, top_n),
        "ESSENTIALIZATION_examples": to_df(by_ess, top_n),
    }


def save_evidence_pack(
    piro_records: List[Dict],
    output_path: Optional[Path] = None,
    top_n: int = 10,
) -> str:
    """Сохраняет evidence_pack.xlsx с листами по категориям."""
    output_path = output_path or Path(__file__).resolve().parent.parent / "output" / "evidence_pack.xlsx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dfs = build_evidence_dfs(piro_records, top_n=top_n)
    with pd.ExcelWriter(output_path, engine="openpyxl") as w:
        for sheet_name, df in dfs.items():
            if not df.empty:
                df.to_excel(w, sheet_name=sheet_name[:31], index=False)
            else:
                pd.DataFrame(columns=["ethnos", "R", "confidence", "sentence_text", "context_text", "file_name", "sent_idx", "doc_position_percent"]).to_excel(w, sheet_name=sheet_name[:31], index=False)
    return str(output_path)
