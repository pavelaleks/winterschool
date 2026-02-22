#!/usr/bin/env python3
"""
Автономный скрипт для мастер-класса переводчиков.
Работает поверх готового SQLite-корпуса (output/corpus.db).
Формирует «переводческий пакет»: эссенциализация, грамматика власти, экзотизация через ландшафт,
опасные оценочные слова, мини-кейсы для интерактива.
"""

import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Добавляем корень проекта в path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.translator_demo_helpers import (
    detect_essentialization_pattern,
    get_imperial_agent_list,
    get_pitfall_suggestion,
    get_pitfalls_list,
    introspect_db,
    is_noise_demo,
    load_ethnonyms_yml,
    load_landscape_lexicon_file,
    load_mentions_from_conn,
)


def _log(msg: str, log_path: Optional[Path] = None) -> None:
    print(msg)
    if log_path:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass


def _ensure_columns(df: pd.DataFrame, required: List[str], fill: Any = "") -> pd.DataFrame:
    for c in required:
        if c not in df.columns:
            df[c] = fill
    return df


# --------------- Module A: Essentialization ---------------

def module_a_essentialization(
    df: pd.DataFrame,
    out_dir: Path,
    max_examples: int,
    include_noise: bool,
    log_path: Optional[Path],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Паттерны эссенциализации. Листы: examples_clean, examples_raw, stats_by_ethnos."""
    text_col = "context_text" if "context_text" in df.columns else "sentence_text"
    rows_clean = []
    rows_raw = []
    for _, row in df.iterrows():
        text = row.get(text_col) or row.get("sentence_text") or ""
        if not text or not isinstance(text, str):
            continue
        pat_result = detect_essentialization_pattern(text)
        if not pat_result:
            continue
        pattern_name, _ = pat_result
        ethnos = row.get("ethnos_norm") or ""
        rec = {
            "ethnos": ethnos,
            "pattern": pattern_name,
            "sentence_text": row.get("sentence_text") or text,
            "context_text": row.get("context_text") or text,
            "file_name": row.get("file_name") or "",
            "sent_idx": row.get("sent_idx"),
            "notes_for_translator": "",
            "translation_variant_literal": "",
            "translation_variant_neutralizing": "",
        }
        is_noise = is_noise_demo(text)
        if not is_noise:
            rows_clean.append(rec)
        if include_noise or not is_noise:
            rows_raw.append(rec)
        if len(rows_clean) >= max_examples and len(rows_raw) >= max_examples:
            break
    df_clean = pd.DataFrame(rows_clean[:max_examples] if not include_noise else rows_clean)
    df_raw = pd.DataFrame(rows_raw[:max_examples])
    stats = df_clean.groupby(["ethnos", "pattern"]).size().reset_index(name="count") if not df_clean.empty else pd.DataFrame(columns=["ethnos", "pattern", "count"])
    out_path = out_dir / "essentialization_examples.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        _ensure_columns(df_clean, ["ethnos", "pattern", "sentence_text", "context_text", "file_name", "sent_idx", "notes_for_translator", "translation_variant_literal", "translation_variant_neutralizing"]).to_excel(w, sheet_name="examples_clean", index=False)
        _ensure_columns(df_raw, list(df_clean.columns)).to_excel(w, sheet_name="examples_raw", index=False)
        stats.to_excel(w, sheet_name="stats_by_ethnos", index=False)
    _log(f"  Module A: essentialization_examples.xlsx (clean={len(df_clean)}, raw={len(df_raw)})", log_path)
    return df_clean, df_raw, stats


# --------------- Module B: Agency (Grammar of Power) ---------------

def _get_nlp(model_name: str):
    try:
        import spacy
        return spacy.load(model_name)
    except OSError:
        return None


def _extract_svo(nlp, text: str, ethnos_lemmas: set, imperial_lemmas: set) -> List[Dict[str, Any]]:
    """Извлекает субъект-глагол-объект; классифицирует субъект/объект как ETHNOS, IMPERIAL_AGENT, OTHER."""
    if not text or not nlp:
        return []
    doc = nlp(text[:100000])
    results = []
    for sent in doc.sents:
        subj, verb_lemma, obj = "", "", ""
        for tok in sent:
            if tok.dep_ == "nsubj" or tok.dep_ == "nsubjpass":
                subj = " ".join(t.text for t in tok.subtree).lower()
            if tok.dep_ == "ROOT" and tok.pos_ == "VERB":
                verb_lemma = tok.lemma_.lower()
            if tok.dep_ in ("dobj", "attr", "oprd"):
                obj = " ".join(t.text for t in tok.subtree).lower()
        if not (subj or verb_lemma):
            continue
        subj_type = "ETHNOS" if any(e in subj for e in ethnos_lemmas) else ("IMPERIAL_AGENT" if any(i in subj for i in imperial_lemmas) else "OTHER")
        obj_type = "ETHNOS" if any(e in obj for e in ethnos_lemmas) else ("IMPERIAL_AGENT" if any(i in obj for i in imperial_lemmas) else "OTHER")
        results.append({"subject": subj, "lemma_verb": verb_lemma, "object": obj, "subject_type": subj_type, "object_type": obj_type})
    return results


def module_b_agency(
    df: pd.DataFrame,
    out_dir: Path,
    max_examples: int,
    nlp: Any,
    spacy_model: str,
    log_path: Optional[Path],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Грамматика власти: SVO, ETHNOS/IMPERIAL_AGENT. agency_edges.csv, agency_examples.xlsx, agency_stats.json."""
    if nlp is None:
        _log("  Module B: spaCy не загружен — пропуск. Установите: python -m spacy download " + spacy_model, log_path)
        return pd.DataFrame(), pd.DataFrame(), {}
    imperial = set(get_imperial_agent_list())
    ethnos_lemmas = set(df["ethnos_norm"].dropna().unique().tolist()) if "ethnos_norm" in df.columns else set()
    ethnos_lemmas = {e.lower() for e in ethnos_lemmas if isinstance(e, str)}
    edges = []
    examples_rows = []
    for _, row in df.iterrows():
        text = row.get("context_text") or row.get("sentence_text") or ""
        if not text or not isinstance(text, str):
            continue
        if not any(i in text.lower() for i in imperial):
            continue
        svo_list = _extract_svo(nlp, text, ethnos_lemmas, imperial)
        for svo in svo_list:
            if svo["subject_type"] != "OTHER" or svo["object_type"] != "OTHER":
                edges.append({
                    "file_name": row.get("file_name"),
                    "sent_idx": row.get("sent_idx"),
                    "subject": svo["subject"],
                    "subject_type": svo["subject_type"],
                    "lemma_verb": svo["lemma_verb"],
                    "object": svo["object"],
                    "object_type": svo["object_type"],
                })
        if svo_list:
            examples_rows.append({
                "file_name": row.get("file_name"),
                "sent_idx": row.get("sent_idx"),
                "sentence_text": row.get("sentence_text") or text,
                "context_text": text,
                "ethnos_norm": row.get("ethnos_norm"),
                "translation_attention": "",
            })
        if len(edges) >= max_examples * 2 and len(examples_rows) >= max_examples:
            break
    edges_df = pd.DataFrame(edges[:max_examples * 2])
    examples_df = pd.DataFrame(examples_rows[:max_examples])
    # Статистика
    stats: Dict[str, Any] = {"ethnos_as_subject": 0, "ethnos_as_object": 0, "imperial_as_subject": 0, "imperial_as_object": 0, "total_edges": len(edges_df)}
    for _, r in edges_df.iterrows():
        if r.get("subject_type") == "ETHNOS":
            stats["ethnos_as_subject"] += 1
        if r.get("object_type") == "ETHNOS":
            stats["ethnos_as_object"] += 1
        if r.get("subject_type") == "IMPERIAL_AGENT":
            stats["imperial_as_subject"] += 1
        if r.get("object_type") == "IMPERIAL_AGENT":
            stats["imperial_as_object"] += 1
    total = max(stats["total_edges"], 1)
    stats["ethnos_subject_ratio"] = round(stats["ethnos_as_subject"] / total, 4)
    stats["ethnos_object_ratio"] = round(stats["ethnos_as_object"] / total, 4)
    stats["imperial_subject_ratio"] = round(stats["imperial_as_subject"] / total, 4)
    stats["imperial_object_ratio"] = round(stats["imperial_as_object"] / total, 4)
    (out_dir / "agency_edges.csv").write_text(edges_df.to_csv(index=False), encoding="utf-8")
    with pd.ExcelWriter(out_dir / "agency_examples.xlsx", engine="openpyxl") as w:
        _ensure_columns(examples_df, ["file_name", "sent_idx", "sentence_text", "context_text", "ethnos_norm", "translation_attention"]).to_excel(w, sheet_name="examples", index=False)
    (out_dir / "agency_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"  Module B: agency_edges={len(edges_df)}, agency_examples={len(examples_df)}, stats сохранены", log_path)
    return edges_df, examples_df, stats


# --------------- Module C: Landscape exoticization ---------------

def _tokenize_lower(text: str) -> set:
    return set(re.findall(r"[a-z]+", (text or "").lower()))


def module_c_landscape(
    df: pd.DataFrame,
    out_dir: Path,
    max_examples: int,
    resources_dir: Path,
    log_path: Optional[Path],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Экзотизация через лексику природы/пространства."""
    lexicon = set(load_landscape_lexicon_file(resources_dir))
    text_col = "context_text" if "context_text" in df.columns else "sentence_text"
    rows = []
    cooccur: Dict[Tuple[str, str], int] = {}
    for _, row in df.iterrows():
        text = row.get(text_col) or row.get("sentence_text") or ""
        if not text:
            continue
        words = _tokenize_lower(text)
        hit = lexicon & words
        if not hit:
            continue
        ethnos = row.get("ethnos_norm") or ""
        rows.append({
            "ethnos": ethnos,
            "sentence_text": row.get("sentence_text") or text,
            "context_text": row.get("context_text") or text,
            "file_name": row.get("file_name"),
            "sent_idx": row.get("sent_idx"),
            "landscape_terms": ", ".join(sorted(hit)),
        })
        for term in hit:
            cooccur[(ethnos, term)] = cooccur.get((ethnos, term), 0) + 1
        if len(rows) >= max_examples:
            break
    df_out = pd.DataFrame(rows[:max_examples])
    cooccur_df = pd.DataFrame([{"ethnos": k[0], "lemma": k[1], "count": v} for k, v in sorted(cooccur.items(), key=lambda x: -x[1])])
    with pd.ExcelWriter(out_dir / "landscape_exoticization.xlsx", engine="openpyxl") as w:
        df_out.to_excel(w, sheet_name="examples", index=False)
    cooccur_df.to_csv(out_dir / "landscape_cooccurrence.csv", index=False, encoding="utf-8")
    _log(f"  Module C: landscape_exoticization.xlsx ({len(df_out)}), landscape_cooccurrence.csv", log_path)
    return df_out, cooccur_df


# --------------- Module D: Translation pitfalls ---------------

def module_d_pitfalls(
    df: pd.DataFrame,
    out_dir: Path,
    max_examples: int,
    log_path: Optional[Path],
) -> pd.DataFrame:
    """Опасные оценочные слова с контекстами и подсказками перевода."""
    pitfalls = set(get_pitfalls_list())
    text_col = "context_text" if "context_text" in df.columns else "sentence_text"
    rows = []
    for _, row in df.iterrows():
        text = row.get(text_col) or row.get("sentence_text") or ""
        if not text:
            continue
        words_lower = (text or "").lower().split()
        words_set = set(re.sub(r"\W", "", w) for w in words_lower)
        hit = [p for p in pitfalls if p in words_set or any(p in w for w in words_lower)]
        if not hit:
            continue
        for word in hit[:3]:
            rows.append({
                "ethnos": row.get("ethnos_norm") or "",
                "pitfall_word": word,
                "sentence_text": row.get("sentence_text") or text,
                "context_text": row.get("context_text") or text,
                "file_name": row.get("file_name"),
                "sent_idx": row.get("sent_idx"),
                "suggested_translation_options": get_pitfall_suggestion(word),
                "comment_for_discussion": "",
            })
        if len(rows) >= max_examples:
            break
    out_df = pd.DataFrame(rows[:max_examples])
    with pd.ExcelWriter(out_dir / "pitfalls.xlsx", engine="openpyxl") as w:
        out_df.to_excel(w, sheet_name="pitfalls", index=False)
    _log(f"  Module D: pitfalls.xlsx ({len(out_df)})", log_path)
    return out_df


# --------------- Module E: Workshop mini-cases ---------------

def _questions_for_theme(theme: str) -> List[str]:
    if theme == "essentialization":
        return ["Какие обобщающие формулы вы видите?", "Как передать в переводе без эссенциализации?"]
    if theme == "agency":
        return ["Кто субъект действия? Кто объект?", "Как изменить залог/порядок в переводе?"]
    if theme == "pitfall":
        return ["Какая коннотация у выделенного слова?", "Какие варианты перевода возможны и чем они отличаются?"]
    if theme == "landscape":
        return ["Как лексика природы связана с описанием народа?", "Как смягчить экзотизацию в переводе?"]
    return ["Обсудите варианты перевода."]


def module_e_workshop_cases(
    df_clean: pd.DataFrame,
    essentialization_rows: List[Dict],
    agency_examples_df: pd.DataFrame,
    pitfalls_df: pd.DataFrame,
    landscape_df: pd.DataFrame,
    out_dir: Path,
    log_path: Optional[Path],
) -> pd.DataFrame:
    """15–20 мини-кейсов: 5 essentialization, 5 agency, 5 pitfalls, 5 landscape."""
    cases = []
    case_id = 0
    # 5 essentialization
    for _, row in (pd.DataFrame(essentialization_rows).head(5) if essentialization_rows else pd.DataFrame()).iterrows():
        case_id += 1
        text = row.get("context_text") or row.get("sentence_text") or ""
        pat = detect_essentialization_pattern(text)
        markers = [pat[0]] if pat else []
        cases.append({
            "case_id": case_id,
            "theme": "essentialization",
            "sentence_text": row.get("sentence_text") or text,
            "context_text": text,
            "detected_markers": ", ".join(markers),
            "questions_for_translators": " | ".join(_questions_for_theme("essentialization")),
            "translation_blank": "",
        })
    # 5 agency
    for _, row in (agency_examples_df.head(5) if not agency_examples_df.empty else pd.DataFrame()).iterrows():
        case_id += 1
        cases.append({
            "case_id": case_id,
            "theme": "agency",
            "sentence_text": row.get("sentence_text") or "",
            "context_text": row.get("context_text") or "",
            "detected_markers": "SVO, agency",
            "questions_for_translators": " | ".join(_questions_for_theme("agency")),
            "translation_blank": "",
        })
    # 5 pitfalls
    for _, row in (pitfalls_df.head(5) if not pitfalls_df.empty else pd.DataFrame()).iterrows():
        case_id += 1
        cases.append({
            "case_id": case_id,
            "theme": "pitfall",
            "sentence_text": row.get("sentence_text") or "",
            "context_text": row.get("context_text") or "",
            "detected_markers": row.get("pitfall_word", ""),
            "questions_for_translators": " | ".join(_questions_for_theme("pitfall")),
            "translation_blank": "",
        })
    # 5 landscape
    for _, row in (landscape_df.head(5) if not landscape_df.empty else pd.DataFrame()).iterrows():
        case_id += 1
        cases.append({
            "case_id": case_id,
            "theme": "landscape",
            "sentence_text": row.get("sentence_text") or "",
            "context_text": row.get("context_text") or "",
            "detected_markers": row.get("landscape_terms", ""),
            "questions_for_translators": " | ".join(_questions_for_theme("landscape")),
            "translation_blank": "",
        })
    out_df = pd.DataFrame(cases)
    with pd.ExcelWriter(out_dir / "workshop_cases.xlsx", engine="openpyxl") as w:
        out_df.to_excel(w, sheet_name="cases", index=False)
    _log(f"  Module E: workshop_cases.xlsx ({len(out_df)} кейсов)", log_path)
    return out_df


# --------------- Main ---------------

def run(
    db_path: Path,
    out_dir: Path,
    max_examples: int = 500,
    include_noise: bool = False,
    spacy_model: str = "en_core_web_sm",
) -> Dict[str, Any]:
    log_path = out_dir / "logs.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs.txt").write_text("", encoding="utf-8")
    _log(f"Translator demo started at {datetime.now().isoformat()}", log_path)
    _log(f"  db_path={db_path}, out_dir={out_dir}, max_examples={max_examples}", log_path)

    if not db_path.exists():
        raise FileNotFoundError(f"База не найдена: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        mapping = introspect_db(conn)
        _log(f"  Найдена схема: {mapping.get('schema')}, mentions_table={mapping.get('mentions_table')}", log_path)
    except RuntimeError as e:
        conn.close()
        raise

    project_root = PROJECT_ROOT
    ethnonyms_pairs = load_ethnonyms_yml(project_root / "resources")
    mentions_df = load_mentions_from_conn(conn, mapping, ethnonyms_pairs, project_root)
    conn.close()
    if "context_text" not in mentions_df.columns or mentions_df["context_text"].isna().all():
        mentions_df["context_text"] = mentions_df.get("sentence_text", "")

    if mentions_df.empty:
        _log("  Нет упоминаний (mentions_df пуст). Проверьте ethnonyms и тексты.", log_path)
        summary = {"db_path": str(db_path), "mentions_table": mapping.get("mentions_table"), "cols": mapping.get("cols"), "n_raw": 0, "n_clean": 0, "modules": {}, "timestamp": datetime.now().isoformat()}
        (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    mentions_df["is_noise_demo"] = mentions_df.apply(lambda r: is_noise_demo(r.get("context_text") or r.get("sentence_text") or ""), axis=1)
    df_raw = mentions_df
    df_clean = mentions_df[~mentions_df["is_noise_demo"]].copy() if not include_noise else mentions_df.copy()
    n_raw, n_clean = len(df_raw), len(df_clean)
    _log(f"  Упоминаний: raw={n_raw}, clean={n_clean}", log_path)

    # Module A
    df_ess_clean, df_ess_raw, stats_ess = module_a_essentialization(df_raw, out_dir, max_examples, include_noise, log_path)

    # Module B
    nlp = _get_nlp(spacy_model)
    if nlp is None:
        _log("  Внимание: для модуля B установите spaCy: python -m spacy download " + spacy_model, log_path)
    edges_df, agency_examples_df, agency_stats = module_b_agency(df_clean, out_dir, max_examples, nlp, spacy_model, log_path)

    # Module C
    landscape_df, cooccur_df = module_c_landscape(df_clean, out_dir, max_examples, project_root / "resources", log_path)

    # Module D
    pitfalls_df = module_d_pitfalls(df_clean, out_dir, max_examples, log_path)

    # Module E
    ess_rows = df_ess_clean.to_dict("records") if not df_ess_clean.empty else []
    workshop_df = module_e_workshop_cases(df_clean, ess_rows, agency_examples_df, pitfalls_df, landscape_df, out_dir, log_path)

    summary = {
        "db_path": str(db_path),
        "mentions_table": mapping.get("mentions_table"),
        "cols": mapping.get("cols"),
        "n_raw": n_raw,
        "n_clean": n_clean,
        "n_essentialization_clean": len(df_ess_clean),
        "n_essentialization_raw": len(df_ess_raw),
        "n_agency_edges": len(edges_df),
        "n_agency_examples": len(agency_examples_df),
        "n_landscape": len(landscape_df),
        "n_pitfalls": len(pitfalls_df),
        "n_workshop_cases": len(workshop_df),
        "timestamp": datetime.now().isoformat(),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _log("  run_summary.json записан.", log_path)
    _log("Translator demo finished.", log_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Переводческий пакет для мастер-класса (поверх corpus.db)")
    parser.add_argument("--db", type=Path, default=None, help="Путь к SQLite (по умолчанию: output/corpus.db)")
    parser.add_argument("--out", type=Path, default=None, help="Выходная папка (по умолчанию: output/translator_demo)")
    parser.add_argument("--max-examples", type=int, default=500, help="Ограничение примеров на модуль")
    parser.add_argument("--include-noise", action="store_true", help="Включать шумные строки в выгрузку")
    parser.add_argument("--spacy-model", type=str, default="en_core_web_sm", help="Модель spaCy для модуля B")
    args = parser.parse_args()

    db_path = args.db or (PROJECT_ROOT / "output" / "corpus.db")
    out_dir = args.out or (PROJECT_ROOT / "output" / "translator_demo")
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    try:
        run(db_path=db_path, out_dir=out_dir, max_examples=args.max_examples, include_noise=args.include_noise, spacy_model=args.spacy_model)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
