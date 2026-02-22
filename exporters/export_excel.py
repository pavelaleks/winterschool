"""
Экспорт полного Evidence Pack в output/piro_fragments.xlsx.
Одна строка = одно упоминание; колонки для ручной разметки и source_pointer для верификации.
O в PIRO = Occasion/ситуация; source_pointer = file_name#sent_idx — указатель на источник.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd


def _source_pointer(row: Dict) -> str:
    """Строгий указатель для верификации: file_name#sent_idx."""
    fn = row.get("file_name") or (row.get("O_metadata") or {}).get("file", "")
    sid = row.get("sent_idx")
    if sid is None:
        sid = (row.get("O_metadata") or {}).get("sentence_index", "")
    return f"{fn}#{sid}"


def piro_fragments_to_dataframe(piro_records: List[Dict]) -> pd.DataFrame:
    """
    Строит DataFrame для piro_fragments.xlsx: все поля PIRO + source_pointer + колонки для ручной правки.
    """
    rows = []
    for r in piro_records:
        row = {
            "mention_id": r.get("mention_id"),
            "file_name": r.get("file_name") or (r.get("O_metadata") or {}).get("file", ""),
            "doc_id": r.get("doc_id"),
            "sent_idx": r.get("sent_idx") if r.get("sent_idx") is not None else (r.get("O_metadata") or {}).get("sentence_index"),
            "doc_position_percent": r.get("doc_position_percent"),
            "ethnos_raw": r.get("ethnos_raw") or r.get("mention", ""),
            "ethnos_norm": r.get("ethnos_norm") or r.get("P", ""),
            "P": r.get("P") or r.get("ethnos_norm", ""),
            "sentence_text": r.get("sentence_text") or "",
            "context_text": r.get("context_text") or r.get("context", ""),
            "R": r.get("R", ""),
            "R_scores": str(r.get("R_scores", ""))[:500] if r.get("R_scores") else "",
            "R_confidence": r.get("R_confidence"),
            "O_situation": r.get("O_situation", ""),
            "O_scores": str(r.get("O_scores", ""))[:500] if r.get("O_scores") else "",
            "O_confidence": r.get("O_confidence"),
            "epithets": ", ".join(r.get("epithets", [])) if isinstance(r.get("epithets"), list) else str(r.get("epithets", ""))[:500],
            "is_essentializing": r.get("is_essentializing", False),
            "essentialization_pattern": r.get("essentialization_pattern", ""),
            "is_probable_header_footer": r.get("is_probable_header_footer", False),
            "is_noise": r.get("is_noise", False),
            "position_bucket": r.get("position_bucket", ""),
            "position_bucket_pct": r.get("position_bucket_pct"),
            "is_position_header_footer": r.get("is_position_header_footer", False),
            "is_cross_doc_header_footer": r.get("is_cross_doc_header_footer", False),
            "manual_keep": "",
            "manual_note": "",
            "manual_R": "",
            "manual_O": "",
            "source_pointer": _source_pointer(r),
            "noise_reason": r.get("noise_reason", ""),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def export_piro_fragments(
    piro_raw: List[Dict],
    output_path: Optional[Path] = None,
) -> str:
    """
    Создаёт output/piro_fragments.xlsx — полный Evidence Pack (1 строка = 1 упоминание).
    piro_raw должен быть построен из raw_df (все упоминания, с флагами is_noise, is_probable_header_footer).
    """
    output_path = output_path or Path(__file__).resolve().parent.parent / "output" / "piro_fragments.xlsx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = piro_fragments_to_dataframe(piro_raw)
    df.to_excel(output_path, sheet_name="PIRO_fragments", index=False, engine="openpyxl")
    return str(output_path)


def export_evidence_base(
    piro_raw: List[Dict],
    piro_clean: List[Dict],
    output_path: Optional[Path] = None,
) -> str:
    """
    Создаёт output/evidence_base.xlsx — база для верификации и статьи.
    Лист 1 «evidence_clean»: всё, что идёт в таблицы отчёта (очищенная выборка), с source_pointer.
    Лист 2 «excluded_noise»: исключённые записи (шум) с noise_reason для аудита.
    """
    output_path = output_path or Path(__file__).resolve().parent.parent / "output" / "evidence_base.xlsx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_clean = piro_fragments_to_dataframe(piro_clean)
    excluded = [r for r in piro_raw if r.get("is_noise")]
    df_excluded = piro_fragments_to_dataframe(excluded) if excluded else pd.DataFrame()

    with pd.ExcelWriter(output_path, engine="openpyxl") as w:
        df_clean.to_excel(w, sheet_name="evidence_clean", index=False)
        if not df_excluded.empty:
            df_excluded.to_excel(w, sheet_name="excluded_noise", index=False)
    return str(output_path)
