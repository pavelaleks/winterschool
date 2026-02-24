"""
Главный скрипт: пайплайн анализа травелогов о Сибири (DH, исследовательская платформа).
По умолчанию: core анализ + интерактивный HTML-отчёт (output/report.html). PDF только с --report-pdf.

Два основных режима:
  1) Полный академический отчёт с LLM (мемо, синтез, учёт токенов в отчёте):
     python main.py --run-llm
     или с кэшем корпуса: python main.py --from-cache --run-llm
     или всё сразу: python main.py --full
  2) Аналитика по уже готовой базе (без пересчёта, без LLM):
     python main.py --from-db
     Требует предварительно созданный output/pipeline.db (шаг 1).

Флаги:
  --from-cache         загрузить корпус из SQLite (output/corpus.db)
  --from-db            загрузить корпус + всю аналитику из output/pipeline.db (не пересчитывать шаги 4–8b, без LLM)
  --export-excel       экспорт piro_full_database.xlsx
  --build-evidence-pack  evidence_pack.xlsx и встроить в HTML
  --run-embeddings     кластеризация и UMAP для валидации
  --run-llm            аналитический модуль LLM: мемо по блокам, синтез, учёт токенов в отчёте
  --run-deepseek       то же что --run-llm (совместимость)
  --report-only        только собрать report.html из pipeline.db и output/* (без прогона)
  --report-pdf         дополнительно сгенерировать краткий PDF
  --full               всё по очереди (включая embeddings, LLM, report-pdf)
  --ocr-symspell       усиленная очистка OCR
  --jobs N             число процессов spaCy (по умолчанию 1)
  --documents NAME     анализ только по выбранным документам (подстрока имени файла, напр. michie; несколько через пробел или запятую)
  --use-lemmas         поиск этнонимов и совпадения лексиконов R/O по леммам (spaCy); иначе по словоформам
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import List, Optional, Dict, Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR = PROJECT_ROOT / "output"

# Минимальная длина фрагмента и паттерны, по которым отбрасываем TOC/header-подобные строки в доказательной выборке
EVIDENCE_MIN_LEN = 25
EVIDENCE_TOC_HEADER_RE = None  # lazy compile


def _is_likely_content_sentence(r: Dict[str, Any]) -> bool:
    """Отбираем предложения, подходящие как доказательства: не оглавление, не колонтитул, не слишком короткие."""
    sent = (r.get("sentence_text") or "").strip()
    if len(sent) < EVIDENCE_MIN_LEN:
        return False
    global EVIDENCE_TOC_HEADER_RE
    if EVIDENCE_TOC_HEADER_RE is None:
        import re
        EVIDENCE_TOC_HEADER_RE = re.compile(
            r"^\s*(Chapter|Page|Part|Book\s+[IVXLCDM\d]+|\d+\s*\.\s*[A-Z]|[IVXLCDM]+\s*\.)\s",
            re.IGNORECASE,
        )
    if EVIDENCE_TOC_HEADER_RE.match(sent):
        return False
    # Слишком много цифр/римских — подозрительно
    digits = sum(c.isdigit() for c in sent)
    if digits > len(sent) // 2:
        return False
    return True


def _extract_document_metadata(corpus: Optional[List[Dict]]) -> List[Dict[str, Any]]:
    """
    Парсит author/year/title из filename в формате Author_Year_Title.txt (best-effort).
    """
    import re
    out: List[Dict[str, Any]] = []
    for d in (corpus or []):
        fn = (d.get("filename") or d.get("file_name") or "").strip()
        if not fn:
            continue
        stem = Path(fn).stem
        parts = stem.split("_")
        year = None
        yi = None
        for i, p in enumerate(parts):
            if re.fullmatch(r"\d{4}", p):
                year = int(p)
                yi = i
                break
        if yi is not None:
            author = " ".join(parts[:yi]).replace("  ", " ").strip(" _-")
            title = " ".join(parts[yi + 1:]).replace("  ", " ").strip(" _-")
        else:
            author = ""
            title = stem.replace("_", " ").strip()
        out.append({
            "filename": fn,
            "author": author,
            "year": year,
            "title": title,
        })
    return out


def build_run_passport(
    corpus: List[Dict],
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    input_documents: Optional[List[str]] = None,
    noise_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Единый паспорт прогона: дата, входные файлы, объёмы, параметры фильтра шума."""
    num_docs = len(corpus) if corpus else 0
    num_sents = sum(len(d.get("sentences", [])) for d in (corpus or []))
    n_raw = len(raw_df) if raw_df is not None else 0
    n_clean = len(clean_df) if clean_df is not None else 0
    n_noise = n_raw - n_clean
    noise_pct = round(100 * n_noise / n_raw, 1) if n_raw else 0
    doc_names = input_documents or []
    if not doc_names and corpus:
        doc_names = [d.get("filename", d.get("file_name", "")) for d in corpus if d]
    return {
        "run_ts": datetime.now().isoformat(),
        "input_documents": doc_names,
        "num_docs": num_docs,
        "num_sents": num_sents,
        "n_raw": n_raw,
        "n_clean": n_clean,
        "n_noise": n_noise,
        "noise_pct": noise_pct,
        "noise_filter": noise_params or {},
    }


from tqdm import tqdm

from src.loader import load_texts, get_texts_dir, filter_corpus_by_docs
from src.ocr_cleaner import clean_corpus
from src.preprocess import preprocess_corpus, _get_nlp
from src.corpus_db import save_corpus as save_corpus_db, load_corpus as load_corpus_db
from src.db_introspect import build_mentions_df_from_corpus
from src.noise_filter import run_noise_filter
from src.piro import run_piro_on_corpus, build_piro_from_mentions_df
from src.keyness import run_keyness_visualizations, keyness_by_representation, keyness_by_ethnos
from src.normalization import normalized_stats_ethnos, normalized_stats_R, normalized_stats_O
from src.relations import run_relations_pipeline
from src.visualization import run_all_visualizations
from src.ethnonym_extractor import get_ethnonym_patterns_cached
from src.discourse_patterns import get_essentialization_table, get_essentialization_examples
from src.report import build_report_from_pipeline
from src.html_report import build_html_report


def _run_report_only() -> None:
    """
    Собирает report.html из уже сохранённых данных (pipeline.db, output/derived/, output/llm_memos/).
    Без прогона пайплайна и без вызовов LLM. Требует существующий output/pipeline.db.
    """
    from src.pipeline_db import load_pipeline, DEFAULT_DB_PATH
    print("Режим: только сборка отчёта (данные из pipeline.db и файлов).")
    pipeline_data = load_pipeline()
    if not pipeline_data:
        if not DEFAULT_DB_PATH.exists():
            print("   Файл output/pipeline.db не найден. Сначала выполните полный прогон: python main.py")
        else:
            print("   Кэш pipeline пуст или устарел. Выполните: python main.py (без --report-only)")
        return
    corpus = pipeline_data["corpus"]
    raw_df = pipeline_data["raw_df"]
    clean_df = pipeline_data["clean_df"]
    piro_raw = pipeline_data.get("piro_raw") or []
    piro_clean = pipeline_data.get("piro_clean") or []
    norm_ethnos = pipeline_data.get("norm_ethnos") or {}
    norm_R = pipeline_data.get("norm_R") or {}
    norm_O = pipeline_data.get("norm_O") or {}
    keyness_tables_combined = pipeline_data.get("keyness_tables") or {}
    essentialization_table = pipeline_data.get("essentialization_table") or {}
    essentialization_examples = pipeline_data.get("essentialization_examples") or {}
    interaction_edges = pipeline_data.get("interaction_edges") or []
    comention_raw = pipeline_data.get("comention_raw") or {}
    derived_indices = pipeline_data.get("derived_indices")
    n_raw, n_clean = len(raw_df), len(clean_df)
    print(f"   Загружено из pipeline.db: {len(corpus)} док., {len(piro_clean)} упоминаний PIRO.")

    # Derived-данные из файлов (если есть)
    derived_profiles_records = None
    derived_stats_tests = None
    derived_correlations_records = None
    derived_clusters_records = None
    ethnic_profiles_path = OUTPUT_DIR / "tables" / "ethnic_profiles.csv"
    if ethnic_profiles_path.exists():
        try:
            pr = pd.read_csv(ethnic_profiles_path, encoding="utf-8")
            if not pr.empty:
                derived_profiles_records = pr.fillna("").to_dict(orient="records")
        except Exception as e:
            print(f"   Предупреждение: не удалось прочитать {ethnic_profiles_path.name}: {e}")
    stats_path = OUTPUT_DIR / "derived" / "stats_tests.json"
    if stats_path.exists():
        try:
            derived_stats_tests = json.loads(stats_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"   Предупреждение: не удалось прочитать {stats_path.name}: {e}")
    corr_path = OUTPUT_DIR / "derived" / "correlations.csv"
    if corr_path.exists():
        try:
            derived_correlations_records = pd.read_csv(corr_path, encoding="utf-8").to_dict(orient="records")
        except Exception as e:
            print(f"   Предупреждение: не удалось прочитать {corr_path.name}: {e}")
    clusters_path = OUTPUT_DIR / "derived" / "ethnos_clusters.csv"
    if clusters_path.exists():
        try:
            derived_clusters_records = pd.read_csv(clusters_path, encoding="utf-8").to_dict(orient="records")
        except Exception as e:
            print(f"   Предупреждение: не удалось прочитать {clusters_path.name}: {e}")
    if not derived_indices and (OUTPUT_DIR / "derived" / "derived_indices.json").exists():
        try:
            derived_indices = json.loads((OUTPUT_DIR / "derived" / "derived_indices.json").read_text(encoding="utf-8"))
        except Exception as e:
            print(f"   Предупреждение: не удалось прочитать derived/derived_indices.json: {e}")
    if not derived_indices and (OUTPUT_DIR / "derived_indices.json").exists():
        try:
            derived_indices = json.loads((OUTPUT_DIR / "derived_indices.json").read_text(encoding="utf-8"))
        except Exception as e:
            print(f"   Предупреждение: не удалось прочитать derived_indices.json: {e}")

    # Мемо и синтез из output/llm_memos
    report_memos = {}
    try:
        from llm.memo_engine import load_all_memos, get_rule_only_memo, get_fallback_memo
        loaded = load_all_memos(OUTPUT_DIR / "llm_memos")
        report_memos["representations"] = loaded.get("representation") or get_fallback_memo("representations")
        report_memos["situations"] = loaded.get("situation") or get_fallback_memo("situations")
        report_memos["keyness"] = loaded.get("keyness") or get_fallback_memo("keyness")
        report_memos["essentialization"] = loaded.get("essentialization") or get_fallback_memo("essentialization")
        report_memos["networks"] = loaded.get("network") or get_fallback_memo("networks")
        report_memos["embedding"] = loaded.get("embeddings") or get_fallback_memo("embedding")
        report_memos["corpus"] = get_rule_only_memo("corpus", {"raw": n_raw, "clean": n_clean})
        report_memos["evidence"] = get_rule_only_memo("evidence", {})
        report_memos["limits"] = get_rule_only_memo("limits", {})
    except Exception as e:
        print(f"   Загрузка мемо: {e}")
    synthesis_data = None
    path_syn = OUTPUT_DIR / "llm_memos" / "synthesis.json"
    if path_syn.exists():
        try:
            synthesis_data = json.loads(path_syn.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"   Предупреждение: не удалось прочитать synthesis.json: {e}")
    token_usage = None
    if (OUTPUT_DIR / "metadata" / "llm_token_count.json").exists():
        try:
            token_usage = json.loads((OUTPUT_DIR / "metadata" / "llm_token_count.json").read_text(encoding="utf-8"))
        except Exception as e:
            print(f"   Предупреждение: не удалось прочитать llm_token_count.json: {e}")

    try:
        from reports.report_assets import ensure_report_assets
        ensure_report_assets(OUTPUT_DIR)
    except Exception as e:
        print(f"   Предупреждение: не удалось подготовить assets отчёта: {e}")
    evidence_df = None
    try:
        from exporters.export_excel import piro_fragments_to_dataframe
        # В отчёте — только очищенные фрагменты (без оглавления/колонтитулов/позиционных повторов)
        evidence_df = piro_fragments_to_dataframe(piro_clean)
    except Exception as e:
        if piro_clean:
            evidence_df = pd.DataFrame(piro_clean)
        print(f"   Предупреждение: fallback evidence_df из piro_clean (ошибка export helper): {e}")
    run_passport = build_run_passport(corpus, raw_df, clean_df)
    passport_path = OUTPUT_DIR / "metadata" / "run_passport.json"
    if passport_path.exists():
        try:
            saved = json.loads(passport_path.read_text(encoding="utf-8"))
            run_passport["noise_filter"] = saved.get("noise_filter", run_passport.get("noise_filter"))
        except Exception as e:
            print(f"   Предупреждение: не удалось прочитать run_passport.json: {e}")

    from reports.build_report import build_report
    scientific_sections = None
    try:
        from llm.scientific_report import load_scientific_report
        scientific_sections = load_scientific_report(OUTPUT_DIR / "llm_memos" / "scientific_report.json")
    except Exception as e:
        print(f"   Предупреждение: не удалось загрузить scientific_report.json: {e}")
    html_path = build_report(
        corpus=corpus,
        raw_df=raw_df,
        clean_df=clean_df,
        piro_raw=piro_raw,
        piro_clean=piro_clean,
        keyness_tables=keyness_tables_combined,
        norm_ethnos=norm_ethnos,
        norm_R=norm_R,
        norm_O=norm_O,
        essentialization_table=essentialization_table,
        essentialization_examples=essentialization_examples,
        interaction_edges=interaction_edges,
        comention_raw=comention_raw,
        evidence_df=evidence_df,
        cluster_validation=None,
        llm_memos=report_memos,
        derived_indices=derived_indices,
        synthesis=synthesis_data,
        derived_profiles=derived_profiles_records,
        derived_stats_tests=derived_stats_tests,
        derived_correlations=derived_correlations_records,
        derived_clusters=derived_clusters_records,
        scientific_report_sections=scientific_sections,
        llm_enabled=False,
        api_base=os.environ.get("DASHBOARD_API_BASE", "http://127.0.0.1:5000"),
        token_usage=token_usage,
        run_passport=run_passport,
        output_path=OUTPUT_DIR / "report.html",
    )
    print(f"   Отчёт: {html_path}")
    print("Готово.")


def _run_scientific_report_only(run_llm: bool = False, staged: bool = False) -> None:
    """
    Генерирует только научный отчёт (scientific_report.json, scientific_report.html) из уже сохранённых данных.
    Данные берутся из pipeline.db, output/derived/, output/llm_memos/. Без прогона пайплайна.
    С --run-llm вызывается DeepSeek для плотного текста; без флага — rule-based заглушки.
    """
    from src.pipeline_db import load_pipeline, DEFAULT_DB_PATH
    from llm.scientific_report import build_scientific_report_payload, generate_scientific_report

    print("Режим: только научный отчёт (данные из pipeline.db и файлов).")
    pipeline_data = load_pipeline()
    if not pipeline_data:
        if not DEFAULT_DB_PATH.exists():
            print("   Файл output/pipeline.db не найден. Сначала выполните полный прогон: python main.py")
        else:
            print("   Кэш pipeline пуст или устарел. Выполните: python main.py (без --scientific-report-only)")
        return

    corpus = pipeline_data["corpus"]
    raw_df = pipeline_data["raw_df"]
    clean_df = pipeline_data["clean_df"]
    piro_clean = pipeline_data.get("piro_clean") or []
    keyness_tables_combined = pipeline_data.get("keyness_tables") or {}
    essentialization_table = pipeline_data.get("essentialization_table") or {}
    interaction_edges = pipeline_data.get("interaction_edges") or []
    derived_indices = pipeline_data.get("derived_indices")
    n_raw, n_clean = len(raw_df), len(clean_df)
    num_sents = sum(len(d.get("sentences", [])) for d in corpus) if corpus else 0
    print(f"   Загружено: {len(corpus)} док., {len(piro_clean)} упоминаний PIRO.")

    ethnos_excl = dict(Counter(r.get("P") or r.get("ethnos_norm", "") for r in piro_clean if (r.get("P") or r.get("ethnos_norm"))))
    R_excl = dict(Counter(r.get("R", "") for r in piro_clean))
    O_excl = dict(Counter(r.get("O_situation", "") for r in piro_clean))

    keyness_top = {}
    for k, df in (keyness_tables_combined or {}).items():
        if df is not None and not df.empty and "word" in df.columns:
            keyness_top[k] = df["word"].astype(str).tolist()[:30]

    pr_list = None
    tests_dict = None
    corr_list = None
    if (OUTPUT_DIR / "derived" / "derived_indices.json").exists():
        try:
            derived_indices = derived_indices or json.loads((OUTPUT_DIR / "derived" / "derived_indices.json").read_text(encoding="utf-8"))
        except Exception:
            pass
    if (OUTPUT_DIR / "tables" / "ethnic_profiles.csv").exists():
        try:
            pr = pd.read_csv(OUTPUT_DIR / "tables" / "ethnic_profiles.csv", encoding="utf-8")
            if not pr.empty:
                pr_list = pr.fillna("").head(25).to_dict(orient="records")
        except Exception:
            pass
    if (OUTPUT_DIR / "derived" / "stats_tests.json").exists():
        try:
            tests_dict = json.loads((OUTPUT_DIR / "derived" / "stats_tests.json").read_text(encoding="utf-8"))
        except Exception:
            pass
    if (OUTPUT_DIR / "derived" / "correlations.csv").exists():
        try:
            corr = pd.read_csv(OUTPUT_DIR / "derived" / "correlations.csv", encoding="utf-8")
            if not corr.empty:
                corr_list = corr.head(20).to_dict(orient="records")
        except Exception:
            pass

    report_memos_for_sci = {}
    try:
        from llm.memo_engine import load_all_memos, get_rule_only_memo, get_fallback_memo
        loaded_memos = load_all_memos(OUTPUT_DIR / "llm_memos")
        report_memos_for_sci["representations"] = loaded_memos.get("representation") or get_fallback_memo("representations")
        report_memos_for_sci["situations"] = loaded_memos.get("situation") or get_fallback_memo("situations")
        report_memos_for_sci["keyness"] = loaded_memos.get("keyness") or get_fallback_memo("keyness")
        report_memos_for_sci["essentialization"] = loaded_memos.get("essentialization") or get_fallback_memo("essentialization")
        report_memos_for_sci["networks"] = loaded_memos.get("network") or get_fallback_memo("networks")
        report_memos_for_sci["embedding"] = loaded_memos.get("embeddings") or get_fallback_memo("embedding")
        report_memos_for_sci["corpus"] = get_rule_only_memo("corpus", {"raw": n_raw, "clean": n_clean})
        report_memos_for_sci["evidence"] = get_rule_only_memo("evidence", {})
        report_memos_for_sci["limits"] = get_rule_only_memo("limits", {})
    except Exception as e:
        print(f"   Загрузка мемо: {e}")
    synthesis_for_sci = None
    path_syn = OUTPUT_DIR / "llm_memos" / "synthesis.json"
    if path_syn.exists():
        try:
            synthesis_for_sci = json.loads(path_syn.read_text(encoding="utf-8"))
        except Exception:
            pass

    doc_meta = _extract_document_metadata(corpus)
    sci_payload = build_scientific_report_payload(
        num_docs=len(corpus) if corpus else 0,
        num_sents=num_sents,
        n_raw=n_raw,
        n_clean=n_clean,
        ethnos_raw_excl=ethnos_excl,
        R_raw_excl=R_excl,
        O_raw_excl=O_excl,
        keyness_top=keyness_top,
        essentialization_table=essentialization_table,
        interaction_edges=interaction_edges,
        derived_indices=derived_indices,
        ethnic_profiles=pr_list,
        stats_tests=tests_dict,
        correlations=corr_list,
        report_memos=report_memos_for_sci,
        synthesis=synthesis_for_sci,
        document_metadata=doc_meta,
    )
    run_passport_sci = build_run_passport(corpus, raw_df, clean_df)
    passport_path = OUTPUT_DIR / "metadata" / "run_passport.json"
    if passport_path.exists():
        try:
            saved = json.loads(passport_path.read_text(encoding="utf-8"))
            run_passport_sci["noise_filter"] = saved.get("noise_filter", run_passport_sci.get("noise_filter"))
        except Exception:
            pass
    evidence_sample_sci = []
    candidates = [r for r in (piro_clean or []) if _is_likely_content_sentence(r)]
    if not candidates:
        candidates = (piro_clean or [])[:15]
    for r in candidates[:10]:
        fn = r.get("file_name") or (r.get("O_metadata") or {}).get("file", "")
        sid = r.get("sent_idx") if r.get("sent_idx") is not None else (r.get("O_metadata") or {}).get("sentence_index", "")
        evidence_sample_sci.append({
            "source_pointer": f"{fn}#{sid}" if fn or sid else "",
            "P": r.get("P"), "R": r.get("R"), "sentence_text": r.get("sentence_text"),
            "file_name": fn, "sent_idx": sid,
        })
    generate_scientific_report(
        sci_payload,
        run_llm=run_llm,
        output_json_path=OUTPUT_DIR / "llm_memos" / "scientific_report.json",
        output_html_path=OUTPUT_DIR / "scientific_report.html",
        run_passport=run_passport_sci,
        evidence_sample=evidence_sample_sci,
        staged=staged,
    )
    print(f"   Сохранено: output/llm_memos/scientific_report.json, output/scientific_report.html")
    if run_llm:
        print("   Для обновления главного отчёта с этим блоком запустите: python main.py --report-only")
    print("Готово.")


def main(
    use_ocr_symspell: bool = False,
    n_jobs: int = 1,
    from_cache: bool = False,
    from_db: bool = False,
    export_excel: bool = False,
    build_evidence_pack: bool = False,
    run_embeddings: bool = False,
    run_llm: bool = False,
    run_deepseek: bool = False,
    run_derived: bool = True,
    report_pdf: bool = False,
    full: bool = False,
    report_only: bool = False,
    scientific_report_only: bool = False,
    scientific_report_staged: bool = False,
    documents_filter: Optional[List[str]] = None,
    use_lemmas: bool = False,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus = None
    pipeline_data = None
    if run_deepseek:
        run_llm = True
    if full:
        run_embeddings = True
        run_llm = True
        export_excel = True
        build_evidence_pack = True

    # Режим «только отчёт»: все данные из pipeline.db и файлов, без прогона
    if report_only:
        _run_report_only()
        return
    # Режим «только научный отчёт»: данные из pipeline.db и файлов, генерация scientific_report.json/html
    if scientific_report_only:
        _run_scientific_report_only(run_llm=run_llm, staged=scientific_report_staged)
        return

    # При выборе документов (--documents) не используем pipeline.db — всегда считаем по подкорпусу
    if documents_filter:
        from_db = False
        print(f"   Анализ по выбранным документам: {documents_filter}")

    # Режим запуска: полный с LLM / из базы (опционально с LLM) / полный без LLM
    if from_db:
        if run_llm:
            print("Режим: загрузка из базы (output/pipeline.db), аналитика без пересчёта; мемо и синтез через LLM.")
        else:
            print("Режим: загрузка из базы (output/pipeline.db), аналитика без пересчёта и без LLM.")
    elif run_llm:
        print("Режим: полный прогон с LLM — максимально полный академический отчёт и учёт токенов.")
    else:
        print("Режим: полный прогон без LLM. Для отчёта с интерпретацией и токенами используйте --run-llm или --full.")

    if from_db:
        from src.pipeline_db import load_pipeline, DEFAULT_DB_PATH
        pipeline_data = load_pipeline()
        if pipeline_data:
            print("Загрузка корпуса и аналитики из кэша (output/pipeline.db)...")
            corpus = pipeline_data["corpus"]
            print(f"   Документов: {len(corpus)}, предложений: {sum(len(d.get('sentences', [])) for d in corpus)}")
        else:
            pipeline_data = None
            db_path = DEFAULT_DB_PATH
            if not db_path.exists():
                print("   Кэш pipeline не найден.")
            else:
                print("   Кэш pipeline пуст или устарел.")
            print("   Запуск из базы возможен только после полного прогона. Сделайте:")
            print("   1) python -m spacy download en_core_web_sm")
            print("   2) python main.py   (без --from-db; создаст output/pipeline.db)")
            print("   3) python main.py --from-db   (дальше — загрузка из кэша, spaCy не нужен)")
            return

    if pipeline_data is None and from_cache and corpus is None:
        print("Загрузка корпуса из кэша (SQLite)...")
        corpus = load_corpus_db()
        if corpus:
            corpus = filter_corpus_by_docs(corpus, documents_filter)
            if not corpus and documents_filter:
                print(f"   По селекторам {documents_filter} в кэше нет документов. Проверьте имена файлов в data/texts/.")
                return
            print(f"   Документов: {len(corpus)}, предложений: {sum(len(d.get('sentences', [])) for d in corpus)}")
        else:
            print("   Кэш пуст. Выполняется полный пайплайн.")

    if corpus is None and pipeline_data is None:
        print("1. Загрузка текстов...")
        documents = load_texts(include_docs=documents_filter)
        if not documents:
            if documents_filter:
                print(f"   Нет .txt в data/texts/, подходящих под: {documents_filter}. Укажите подстроку имени файла (например, michie) или имя файла.")
            else:
                print("   В data/texts/ нет .txt. Добавьте тексты травелогов.")
            return
        print(f"   Загружено документов: {len(documents)}")
        print("2. Очистка OCR...")
        documents = clean_corpus(documents, show_progress=True, use_symspell=use_ocr_symspell)
        print("3. Предобработка (spaCy)...")
        try:
            corpus = preprocess_corpus(documents, nlp=None, n_jobs=n_jobs)
        except Exception as e:
            print(f"   Ошибка: {e}. Выполните: python -m spacy download en_core_web_sm")
            return
        print(f"   Предложений: {sum(len(d['sentences']) for d in corpus)}")
        save_corpus_db(corpus)
        print("   Кэш сохранён.")

    if pipeline_data is None:
        print("4. Единая таблица упоминаний (mentions_df)...")
        if use_lemmas:
            print("   Режим: поиск этнонимов и совпадения лексиконов по леммам (spaCy).")
        mentions_df = build_mentions_df_from_corpus(corpus, use_lemmas=use_lemmas)
        if mentions_df.empty:
            print("   Нет упоминаний этнонимов. Проверьте resources/ethnonyms.yml и тексты.")
            return
        print(f"   Упоминаний: {len(mentions_df)}")

        print("5. Фильтр шума (header/footer + позиционная дедупликация)...")
        try:
            _nlp = _get_nlp()
        except Exception:
            _nlp = None
        raw_df, clean_df = run_noise_filter(
            mentions_df,
            repeat_threshold=10,
            output_dir=OUTPUT_DIR,
            nlp=_nlp,
            K_position=2,
            M_global=5,
            N_cross_doc=3,
        )
        print(f"   Raw: {len(raw_df)}, clean: {len(clean_df)}, удалено: {len(raw_df) - len(clean_df)}")

        print("6. PIRO (эпитеты, R, O, эссенциализация)...")
        piro_raw = build_piro_from_mentions_df(raw_df, use_lemmas=use_lemmas)
        piro_clean = build_piro_from_mentions_df(clean_df, use_lemmas=use_lemmas)
        patterns = get_ethnonym_patterns_cached()
        essentialization_table = get_essentialization_table(corpus, patterns)
        essentialization_examples = get_essentialization_examples(piro_clean, min_per_ethnos=5)
        print(f"   PIRO clean: {len(piro_clean)}")
        print("   Экспорт PIRO fragments (Excel)...")
        from exporters.export_excel import export_piro_fragments, export_evidence_base
        export_piro_fragments(piro_raw, output_path=OUTPUT_DIR / "piro_fragments.xlsx")
        export_evidence_base(piro_raw, piro_clean, output_path=OUTPUT_DIR / "evidence_base.xlsx")
        print("   Сохранено: output/piro_fragments.xlsx, output/evidence_base.xlsx")

        print("7. Нормировка и Keyness (таблицы в output/tables/)...")
        norm_ethnos = normalized_stats_ethnos(clean_df)
        norm_df = clean_df[["file_name", "doc_sent_count", "ethnos_norm"]].copy()
        norm_df["R"] = [r.get("R", "") for r in piro_clean]
        norm_df["O_situation"] = [r.get("O_situation", "") for r in piro_clean]
        norm_R = normalized_stats_R(norm_df)
        norm_O = normalized_stats_O(norm_df)
        tables_dir = OUTPUT_DIR / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        keyness_paths = run_keyness_visualizations(piro_clean, OUTPUT_DIR, tables_dir=tables_dir, nlp=_nlp)
        keyness_by_R = keyness_by_representation(piro_clean, OUTPUT_DIR, tables_dir=tables_dir, nlp=_nlp, pos_mode="content_words")
        keyness_by_eth = keyness_by_ethnos(piro_clean, OUTPUT_DIR, tables_dir=tables_dir, nlp=_nlp, pos_mode="content_words")
        keyness_top = {}
        for name, df in list(keyness_by_R.items()) + list(keyness_by_eth.items()):
            if not df.empty and "word" in df.columns:
                keyness_top[name] = df["word"].head(20).tolist()

        print("8. Сети (co-mention + interaction)...")
        rel_result = run_relations_pipeline(corpus, OUTPUT_DIR)
        interaction_matrix = rel_result.get("interaction_matrix", {})
        interaction_edges = rel_result.get("interaction_edges") or []
        comention_raw = rel_result.get("comention_raw") or {}
        comention_jaccard = rel_result.get("comention_jaccard") or {}
        image_paths = run_all_visualizations(
            piro_clean,
            essentialization_table,
            interaction_matrix,
            output_dir=OUTPUT_DIR,
        )
        for name, path in keyness_paths.items():
            image_paths[name] = path
        network_path = OUTPUT_DIR / "network_interaction_directed.png"
        if network_path.exists():
            image_paths["Сеть interaction"] = str(network_path)
        for label, path in [("Co-mention raw", "network_comention_raw.png"), ("Co-mention Jaccard", "network_comention_jaccard.png")]:
            p = OUTPUT_DIR / path
            if p.exists():
                image_paths[label] = str(p)

        print("8b. Производные индексы (OI, AS, ED, EPS)...")
        derived_indices = None
        try:
            from analysis.derived_indices import run_derived_indices
            derived_indices = run_derived_indices(
                piro_clean,
                norm_ethnos=norm_ethnos,
                essentialization_table=essentialization_table,
                interaction_edges=interaction_edges,
                output_path=OUTPUT_DIR / "derived_indices.json",
            )
            print("   Сохранено: output/derived_indices.json")
        except Exception as e:
            print(f"   Индексы: {e}")

        keyness_tables_combined = {**keyness_by_R, **keyness_by_eth}
        from src.pipeline_db import save_pipeline
        save_pipeline(
            corpus=corpus,
            raw_df=raw_df,
            clean_df=clean_df,
            piro_raw=piro_raw,
            piro_clean=piro_clean,
            norm_ethnos=norm_ethnos,
            norm_R=norm_R,
            norm_O=norm_O,
            keyness_tables=keyness_tables_combined,
            essentialization_table=essentialization_table,
            essentialization_examples=essentialization_examples,
            interaction_edges=interaction_edges,
            comention_raw=comention_raw,
            derived_indices=derived_indices,
        )
        print("   Кэш пайплайна сохранён: output/pipeline.db")
    else:
        raw_df = pipeline_data["raw_df"]
        clean_df = pipeline_data["clean_df"]
        piro_raw = pipeline_data["piro_raw"]
        piro_clean = pipeline_data["piro_clean"]
        norm_ethnos = pipeline_data["norm_ethnos"]
        norm_R = pipeline_data["norm_R"]
        norm_O = pipeline_data["norm_O"]
        essentialization_table = pipeline_data["essentialization_table"]
        essentialization_examples = pipeline_data["essentialization_examples"]
        interaction_edges = pipeline_data["interaction_edges"]
        comention_raw = pipeline_data["comention_raw"]
        derived_indices = pipeline_data.get("derived_indices")
        keyness_tables_combined = pipeline_data["keyness_tables"]
        keyness_top = {}
        for name, df in (keyness_tables_combined or {}).items():
            if df is not None and not df.empty and "word" in df.columns:
                keyness_top[name] = df["word"].head(20).tolist()
        image_paths = {}
        keyness_paths = {}

    derived_result = {}
    if run_derived:
        print("8c. Derived analytics (профили этносов, тесты, корреляции, кластеры)...")
        try:
            from analysis.derived_analytics import run_derived_analytics
            pipeline_data = {
                "corpus": corpus,
                "raw_df": raw_df,
                "clean_df": clean_df,
                "piro_clean": piro_clean,
                "norm_ethnos": norm_ethnos,
                "essentialization_table": essentialization_table,
                "interaction_edges": interaction_edges,
            }
            derived_result = run_derived_analytics(
                pipeline_data=pipeline_data,
                output_derived=OUTPUT_DIR / "derived",
                output_tables=OUTPUT_DIR / "tables",
            )
            if derived_result:
                print("   Сохранено: output/derived/*, output/tables/ethnic_profiles.*")
        except Exception as e:
            print(f"   Derived analytics: {e}")

    if build_evidence_pack or full:
        print("9. Evidence pack...")
        from src.evidence_pack import save_evidence_pack
        save_evidence_pack(piro_clean, OUTPUT_DIR / "evidence_pack.xlsx", top_n=10)
        print("   Сохранено: output/evidence_pack.xlsx")

    if export_excel or full:
        print("10. Экспорт Excel (piro_full_database.xlsx)...")
        from src.export_excel import export_piro_full_database
        export_piro_full_database(piro_raw, piro_clean, keyness_tables=keyness_tables_combined, output_path=OUTPUT_DIR / "piro_full_database.xlsx")
        print("   Сохранено: output/piro_full_database.xlsx")

    cluster_validation = None
    if run_embeddings or full:
        print("11. Embeddings и кластеризация (валидация)...")
        try:
            from src.embeddings_clusters import run_embeddings_pipeline
            res = run_embeddings_pipeline(clean_df, OUTPUT_DIR, n_sample=3000, k=8)
            cluster_validation = res
            print("   cluster_validation.json, umap_*.png")
        except Exception as e:
            print(f"   Пропущено: {e}")

    llm_analysis = {"observations": [], "probable_patterns": [], "limitations": []}

    print("12. Аналитический модуль LLM (мемо по блокам)...")
    n_raw = len(raw_df)
    n_clean = len(clean_df)
    n_noise = n_raw - n_clean
    noise_share = round(n_noise / n_raw, 4) if n_raw else 0
    try:
        from llm.memo_engine import (
            generate_memo,
            load_all_memos,
            get_rule_only_memo,
            write_llm_metadata,
        )
        MEMOS_DIR = OUTPUT_DIR / "llm_memos"
        MEMOS_DIR.mkdir(parents=True, exist_ok=True)
        if run_llm:
            from llm.deepseek_client import reset_token_count
            reset_token_count()
            print("   Счётчик токенов сброшен (output/metadata/llm_token_count.json)")

        def _summary_representation():
            r_counts = dict(Counter(r.get("R", "") for r in piro_clean))
            return {
                "counts": r_counts,
                "normalized": norm_R,
                "top_values": list(r_counts.keys())[:10],
                "confidence": "R_confidence из PIRO",
                "noise_share": noise_share,
                "sample_examples": [],
            }

        def _summary_situation():
            o_counts = dict(Counter(r.get("O_situation", "") for r in piro_clean))
            return {
                "counts": o_counts,
                "normalized": norm_O,
                "top_values": list(o_counts.keys())[:10],
                "confidence": "O_confidence из PIRO",
                "noise_share": noise_share,
                "sample_examples": [],
            }

        def _summary_keyness():
            top_keyness = {k: v[:30] for k, v in keyness_top.items()}
            return {
                "counts": {},
                "normalized": {},
                "top_values": top_keyness,
                "confidence": "G2",
                "noise_share": noise_share,
                "sample_examples": [],
            }

        def _summary_essentialization():
            examples = list(essentialization_examples.items())[:5]
            sample = []
            for eth, ex_list in examples:
                for ex in (ex_list or [])[:2]:
                    sample.append({"ethnos": eth, "sentence_preview": (ex.get("sentence_text") or "")[:150]})
            return {
                "counts": essentialization_table,
                "normalized": {},
                "top_values": list(essentialization_table.keys())[:15],
                "confidence": "pattern match",
                "noise_share": noise_share,
                "sample_examples": sample[:5],
            }

        def _summary_network():
            sample = []
            for e in (interaction_edges or [])[:5]:
                ex = (e.get("examples") or [])
                if ex:
                    sample.append({"src": e.get("src"), "dst": e.get("dst"), "example": ex[0][:120]})
            return {
                "counts": {"interaction_edges": len(interaction_edges or []), "comention_pairs": sum(len(v) for v in (comention_raw or {}).values()) // 2},
                "normalized": {},
                "top_values": [],
                "confidence": "verb lexicon",
                "noise_share": noise_share,
                "sample_examples": sample[:5],
            }

        def _summary_embeddings():
            cv = (cluster_validation or {}).get("validation", cluster_validation) if isinstance(cluster_validation, dict) else (cluster_validation or {})
            return {
                "counts": cv,
                "normalized": {},
                "top_values": [],
                "confidence": "silhouette, purity",
                "noise_share": cv.get("noise_ratio"),
                "sample_examples": [],
            }

        blocks = [
            ("representation", _summary_representation),
            ("situation", _summary_situation),
            ("keyness", _summary_keyness),
            ("essentialization", _summary_essentialization),
            ("network", _summary_network),
        ]
        for table_type, summary_fn in blocks:
            try:
                generate_memo(table_type, summary_fn(), run_llm=run_llm, output_dir=MEMOS_DIR)
            except Exception as e:
                print(f"   Мемо {table_type}: {e}")
        if cluster_validation:
            try:
                generate_memo("embeddings", _summary_embeddings(), run_llm=run_llm, output_dir=MEMOS_DIR)
            except Exception as e:
                print(f"   Мемо embeddings: {e}")
        write_llm_metadata(run_llm=run_llm, output_dir=OUTPUT_DIR, seed=42)
        print("   Сохранено: output/llm_memos/*.json, output/metadata/llm_mode.json")
        if derived_result:
            try:
                from llm.memo_engine import generate_synthesis_memo
                pr = derived_result.get("profiles_df")
                tests = derived_result.get("tests") or {}
                corr = derived_result.get("correlations_df")
                clusters_df = derived_result.get("clusters_df")
                run_config = derived_result.get("run_config") or {}
                top_oi = []
                top_ed = []
                if pr is not None and not pr.empty:
                    for _, row in pr.nlargest(10, "mentions_raw").iterrows():
                        eth = row.get("ethnos")
                        ci = (tests.get("bootstrap_CI") or {}).get(eth, {})
                        oi_val = row.get("OI")
                        ed_val = row.get("ED")
                        top_oi.append({"ethnos": eth, "OI": float(oi_val) if oi_val is not None and str(oi_val) != "nan" else None, "CI": ci.get("OI", {})})
                        top_ed.append({"ethnos": eth, "ED": float(ed_val) if ed_val is not None and str(ed_val) != "nan" else None, "CI": ci.get("ED", {})})
                payload = {
                    "top_10_oi": top_oi,
                    "top_10_ed": top_ed,
                    "AS": derived_result.get("derived_indices", {}).get("AS"),
                    "correlations": corr.to_dict(orient="records") if corr is not None and not corr.empty else [],
                    "clusters": clusters_df.to_dict(orient="records") if clusters_df is not None and not clusters_df.empty else [],
                    "chi2_R": tests.get("chi2_R"),
                    "chi2_O": tests.get("chi2_O"),
                    "noise_share": pr["noise_share"].mean() if pr is not None and not pr.empty and "noise_share" in pr.columns else None,
                    "uncertain_R_share": pr["uncertain_R_share"].mean() if pr is not None and not pr.empty and "uncertain_R_share" in pr.columns else None,
                    "n_mentions": run_config.get("n_mentions"),
                    "n_ethnos": run_config.get("n_ethnos"),
                    "fields_used": ["OI", "ED", "EPS", "AS", "mentions_norm"],
                }
                generate_synthesis_memo(payload, run_llm=run_llm, output_path=OUTPUT_DIR / "llm_memos" / "synthesis.json")
                print("   Сохранено: output/llm_memos/synthesis.json (из derived)")
            except Exception as ex:
                print(f"   Синтез (derived): {ex}")
        elif derived_indices:
            try:
                from llm.synthesis import generate_synthesis
                oi = derived_indices.get("OI", {})
                as_scores = derived_indices.get("AS", {})
                ed = derived_indices.get("ED", {})
                eps = derived_indices.get("EPS", {})
                n_edges = sum(e.get("count", 1) for e in interaction_edges)
                n_nodes = len(set(e.get("src") for e in interaction_edges) | set(e.get("dst") for e in interaction_edges))
                density = (2 * n_edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
                generate_synthesis(
                    oi=oi,
                    as_scores=as_scores,
                    ed=ed,
                    eps=eps,
                    keyness_top_terms=keyness_top,
                    network_density=round(density, 4),
                    run_llm=run_llm,
                    output_path=OUTPUT_DIR / "llm_memos" / "synthesis.json",
                )
                print("   Сохранено: output/llm_memos/synthesis.json")
            except Exception as ex:
                print(f"   Синтез: {ex}")
        # Единый обзор для PDF (observations, probable_patterns, limitations)
        if run_llm:
            try:
                from src.deepseek_analysis import run_analysis
                keyness_top20 = {}
                for k, df in (keyness_tables_combined or {}).items():
                    if df is not None and not df.empty and "word" in df.columns:
                        keyness_top20[k] = df.head(20).to_dict(orient="records")
                n_unc = sum(1 for r in piro_clean if r.get("R") == "uncertain")
                n_unk = sum(1 for r in piro_clean if r.get("O_situation") in ("unknown", "mixed"))
                share_uncertain = n_unc / len(piro_clean) if piro_clean else 0
                share_unknown_o = n_unk / len(piro_clean) if piro_clean else 0
                cv = cluster_validation
                if isinstance(cv, dict) and "validation" in cv:
                    cv = cv.get("validation")
                llm_analysis.update(run_analysis(
                    normalized_rates_ethnos=norm_ethnos,
                    normalized_rates_R=norm_R,
                    normalized_rates_O=norm_O,
                    keyness_top20=keyness_top20 or None,
                    essentialization=essentialization_table,
                    cluster_validation=cv,
                    share_unknown_uncertain={"uncertain": share_uncertain, "unknown_O": share_unknown_o},
                    noise_stats={"removed": n_raw - n_clean} if n_raw else None,
                    interaction_summary={
                        "n_edges": len(interaction_edges or []),
                        "top_pairs": [{"src": e.get("src"), "dst": e.get("dst"), "count": e.get("count")} for e in (interaction_edges or [])[:20]],
                    },
                    output_path=OUTPUT_DIR / "llm_analysis.json",
                ))
                print("   Сохранено: output/llm_analysis.json (обзор для PDF)")
            except Exception as ex:
                print(f"   LLM обзор для PDF: {ex}")
        # Плотный научный отчёт ИИ (абзацы по каждой таблице/блоку); учитываем готовые интерпретации из отчёта (memos, synthesis)
        print("12b. Научный отчёт ИИ (плотный аналитический текст по данным)...")
        try:
            from llm.scientific_report import build_scientific_report_payload, generate_scientific_report, load_scientific_report
            num_sents = sum(len(d.get("sentences", [])) for d in corpus) if corpus else 0
            ethnos_excl = dict(Counter(r.get("P") or r.get("ethnos_norm", "") for r in piro_clean if (r.get("P") or r.get("ethnos_norm"))))
            R_excl = dict(Counter(r.get("R", "") for r in piro_clean))
            O_excl = dict(Counter(r.get("O_situation", "") for r in piro_clean))
            pr_list = None
            tests_dict = None
            corr_list = None
            if derived_result:
                pr = derived_result.get("profiles_df")
                if pr is not None and not pr.empty:
                    pr_list = pr.fillna("").head(25).to_dict(orient="records")
                tests_dict = derived_result.get("tests")
                corr = derived_result.get("correlations_df")
                if corr is not None and not corr.empty:
                    corr_list = corr.head(20).to_dict(orient="records")
            report_memos_for_sci = {}
            try:
                from llm.memo_engine import load_all_memos, get_rule_only_memo, get_fallback_memo
                loaded_memos = load_all_memos(OUTPUT_DIR / "llm_memos")
                report_memos_for_sci["representations"] = loaded_memos.get("representation") or get_fallback_memo("representations")
                report_memos_for_sci["situations"] = loaded_memos.get("situation") or get_fallback_memo("situations")
                report_memos_for_sci["keyness"] = loaded_memos.get("keyness") or get_fallback_memo("keyness")
                report_memos_for_sci["essentialization"] = loaded_memos.get("essentialization") or get_fallback_memo("essentialization")
                report_memos_for_sci["networks"] = loaded_memos.get("network") or get_fallback_memo("networks")
                report_memos_for_sci["embedding"] = loaded_memos.get("embeddings") or get_fallback_memo("embedding")
                report_memos_for_sci["corpus"] = get_rule_only_memo("corpus", {"raw": len(raw_df), "clean": len(clean_df)})
                report_memos_for_sci["evidence"] = get_rule_only_memo("evidence", {})
                report_memos_for_sci["limits"] = get_rule_only_memo("limits", {})
            except Exception:
                pass
            synthesis_for_sci = None
            try:
                path_syn = OUTPUT_DIR / "llm_memos" / "synthesis.json"
                if path_syn.exists():
                    synthesis_for_sci = json.loads(path_syn.read_text(encoding="utf-8"))
            except Exception:
                pass
            doc_meta = _extract_document_metadata(corpus)
            sci_payload = build_scientific_report_payload(
                num_docs=len(corpus) if corpus else 0,
                num_sents=num_sents,
                n_raw=len(raw_df),
                n_clean=len(clean_df),
                ethnos_raw_excl=ethnos_excl,
                R_raw_excl=R_excl,
                O_raw_excl=O_excl,
                keyness_top=keyness_top,
                essentialization_table=essentialization_table,
                interaction_edges=interaction_edges,
                derived_indices=derived_indices,
                ethnic_profiles=pr_list,
                stats_tests=tests_dict,
                correlations=corr_list,
                report_memos=report_memos_for_sci,
                synthesis=synthesis_for_sci,
                document_metadata=doc_meta,
            )
            run_passport_sci = build_run_passport(
                corpus, raw_df, clean_df,
                noise_params={"repeat_threshold": 10, "K_position": 2, "M_global": 5, "N_cross_doc": 3},
            )
            evidence_sample_sci = []
            candidates = [r for r in (piro_clean or []) if _is_likely_content_sentence(r)]
            if not candidates:
                candidates = (piro_clean or [])[:15]
            for r in candidates[:10]:
                fn = r.get("file_name") or (r.get("O_metadata") or {}).get("file", "")
                sid = r.get("sent_idx") if r.get("sent_idx") is not None else (r.get("O_metadata") or {}).get("sentence_index", "")
                evidence_sample_sci.append({
                    "source_pointer": f"{fn}#{sid}" if fn or sid else "",
                    "P": r.get("P"), "R": r.get("R"), "sentence_text": r.get("sentence_text"),
                    "file_name": fn, "sent_idx": sid,
                })
            generate_scientific_report(
                sci_payload,
                run_llm=run_llm,
                output_json_path=OUTPUT_DIR / "llm_memos" / "scientific_report.json",
                output_html_path=OUTPUT_DIR / "scientific_report.html",
                run_passport=run_passport_sci,
                evidence_sample=evidence_sample_sci,
                staged=scientific_report_staged,
            )
            print("   Сохранено: output/llm_memos/scientific_report.json, output/scientific_report.html")
        except Exception as ex:
            print(f"   Научный отчёт ИИ: {ex}")
    except Exception as e:
        print(f"   LLM модуль: {e}")

    print("13. Загрузка мемо для отчёта (из файлов или rule-based fallback)...")
    report_memos = {}
    try:
        from llm.memo_engine import load_all_memos, get_rule_only_memo, get_fallback_memo
        loaded = load_all_memos(OUTPUT_DIR / "llm_memos")
        # Если файла нет (например, запуск только --from-db без --run-llm), подставляем rule-based мемо
        report_memos["representations"] = loaded.get("representation") or get_fallback_memo("representations")
        report_memos["situations"] = loaded.get("situation") or get_fallback_memo("situations")
        report_memos["keyness"] = loaded.get("keyness") or get_fallback_memo("keyness")
        report_memos["essentialization"] = loaded.get("essentialization") or get_fallback_memo("essentialization")
        report_memos["networks"] = loaded.get("network") or get_fallback_memo("networks")
        report_memos["embedding"] = loaded.get("embeddings") or get_fallback_memo("embedding")
        report_memos["corpus"] = get_rule_only_memo("corpus", {"raw": n_raw, "clean": n_clean})
        report_memos["evidence"] = get_rule_only_memo("evidence", {})
        report_memos["limits"] = get_rule_only_memo("limits", {})
    except Exception as e:
        print(f"   Загрузка мемо: {e}")

    print("14. Локальные ресурсы отчёта (assets)...")
    try:
        from reports.report_assets import ensure_report_assets
        ensure_report_assets(OUTPUT_DIR)
    except Exception as e:
        print(f"   Предупреждение: {e}")

    print("15. HTML-отчёт (главный артефакт)...")
    evidence_df = None
    try:
        from exporters.export_excel import piro_fragments_to_dataframe
        # Evidence Pack в отчёте — только piro_clean (без шума: оглавление, колонтитулы, повторы)
        evidence_df = piro_fragments_to_dataframe(piro_clean)
    except Exception as e:
        if piro_clean:
            evidence_df = pd.DataFrame(piro_clean)
        print(f"   Предупреждение: fallback evidence_df из piro_clean (ошибка export helper): {e}")
    run_passport = build_run_passport(
        corpus, raw_df, clean_df,
        input_documents=documents_filter if documents_filter else [d.get("filename", "") for d in (corpus or [])],
        noise_params={"repeat_threshold": 10, "K_position": 2, "M_global": 5, "N_cross_doc": 3},
    )
    (OUTPUT_DIR / "metadata").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "metadata" / "run_passport.json").write_text(
        json.dumps(run_passport, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    from reports.build_report import build_report
    synthesis_data = None
    try:
        path_syn = OUTPUT_DIR / "llm_memos" / "synthesis.json"
        if path_syn.exists():
            synthesis_data = json.loads(path_syn.read_text(encoding="utf-8"))
        else:
            from llm.synthesis import load_synthesis
            synthesis_data = load_synthesis(path_syn)
    except Exception as e:
        print(f"   Предупреждение: не удалось загрузить synthesis.json: {e}")
    token_usage = None
    token_count_path = OUTPUT_DIR / "metadata" / "llm_token_count.json"
    if token_count_path.exists():
        try:
            token_usage = json.loads(token_count_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"   Предупреждение: не удалось загрузить llm_token_count.json: {e}")
    derived_profiles_records = None
    derived_stats_tests = None
    derived_correlations_records = None
    derived_clusters_records = None
    if derived_result:
        if derived_result.get("profiles_df") is not None and not derived_result["profiles_df"].empty:
            derived_profiles_records = derived_result["profiles_df"].fillna("").to_dict(orient="records")
        derived_stats_tests = derived_result.get("tests")
        if derived_result.get("correlations_df") is not None and not derived_result["correlations_df"].empty:
            derived_correlations_records = derived_result["correlations_df"].to_dict(orient="records")
        if derived_result.get("clusters_df") is not None and not derived_result["clusters_df"].empty:
            derived_clusters_records = derived_result["clusters_df"].to_dict(orient="records")
    scientific_sections = None
    try:
        from llm.scientific_report import load_scientific_report
        scientific_sections = load_scientific_report(OUTPUT_DIR / "llm_memos" / "scientific_report.json")
    except Exception as e:
        print(f"   Предупреждение: не удалось загрузить scientific_report.json: {e}")
    html_path = build_report(
        corpus=corpus,
        raw_df=raw_df,
        clean_df=clean_df,
        piro_raw=piro_raw,
        piro_clean=piro_clean,
        keyness_tables=keyness_tables_combined,
        norm_ethnos=norm_ethnos,
        norm_R=norm_R,
        norm_O=norm_O,
        essentialization_table=essentialization_table,
        essentialization_examples=essentialization_examples,
        interaction_edges=interaction_edges,
        comention_raw=comention_raw,
        evidence_df=evidence_df,
        cluster_validation=cluster_validation,
        llm_memos=report_memos,
        derived_indices=derived_indices,
        synthesis=synthesis_data,
        derived_profiles=derived_profiles_records,
        derived_stats_tests=derived_stats_tests,
        derived_correlations=derived_correlations_records,
        derived_clusters=derived_clusters_records,
        scientific_report_sections=scientific_sections,
        llm_enabled=run_llm,
        api_base=os.environ.get("DASHBOARD_API_BASE", "http://127.0.0.1:5000"),
        token_usage=token_usage,
        run_passport=run_passport,
        output_path=OUTPUT_DIR / "report.html",
    )
    print(f"   Отчёт: {html_path}")

    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    for name, path in image_paths.items():
        if path and Path(path).exists():
            try:
                shutil.copy2(path, figures_dir / Path(path).name)
            except Exception:
                pass

    print("16. Самопроверки (sanity_checks)...")
    try:
        from sanity_checks import run_sanity_checks
        sc = run_sanity_checks(
            raw_count=len(raw_df),
            clean_count=len(clean_df),
            keyness_top_words=keyness_top,
            interaction_edges=interaction_edges,
            output_dir=OUTPUT_DIR,
        )
        if sc.get("warnings"):
            for w in sc["warnings"]:
                print(f"   Предупреждение: {w}")
        print(f"   Лог: {sc.get('log_path', '')}")
    except Exception as e:
        print(f"   Sanity checks: {e}")

    if report_pdf or full:
        print("17. PDF-отчёт (краткая версия, по флагу --report-pdf)...")
        n_unc = sum(1 for r in piro_clean if r.get("R") == "uncertain")
        n_unk = sum(1 for r in piro_clean if r.get("O_situation") in ("unknown", "mixed"))
        share_uncertain = n_unc / len(piro_clean) if piro_clean else 0
        share_unknown_o = n_unk / len(piro_clean) if piro_clean else 0
        report_path = build_report_from_pipeline(
            doc_list=corpus,
            piro_records=piro_clean,
            essentialization_table=essentialization_table,
            interaction_matrix=interaction_matrix,
            llm_analysis=llm_analysis,
            image_paths=image_paths,
            output_path=OUTPUT_DIR / "final_report.pdf",
            raw_df_len=len(raw_df),
            clean_df_len=len(clean_df),
            share_uncertain=share_uncertain,
            share_unknown_o=share_unknown_o,
            keyness_top=keyness_top,
            cluster_validation=cluster_validation,
        )
        print(f"   PDF: {report_path}")
    print("Готово.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Пайплайн анализа травелогов о Сибири")
    parser.add_argument("--ocr-symspell", action="store_true", help="Очистка OCR с SymSpell")
    parser.add_argument("--jobs", type=int, default=1, metavar="N", help="Процессов для spaCy")
    parser.add_argument("--from-cache", action="store_true", help="Загрузить корпус из output/corpus.db")
    parser.add_argument("--from-db", action="store_true", help="Загрузить корпус и аналитику из output/pipeline.db (не пересчитывать шаги 4–8b)")
    parser.add_argument("--export-excel", action="store_true", help="Экспорт piro_full_database.xlsx")
    parser.add_argument("--build-evidence-pack", action="store_true", help="Собрать evidence_pack.xlsx")
    parser.add_argument("--run-embeddings", action="store_true", help="Кластеризация и UMAP")
    parser.add_argument("--run-llm", action="store_true", help="Аналитический модуль LLM: мемо по блокам (representation, situation, keyness, …)")
    parser.add_argument("--run-deepseek", action="store_true", help="То же что --run-llm (совместимость)")
    parser.add_argument("--no-derived", action="store_true", help="Отключить второй слой аналитики (профили, тесты, корреляции, кластеры)")
    parser.add_argument("--report-only", action="store_true", help="Только собрать report.html из pipeline.db и файлов (без прогона пайплайна)")
    parser.add_argument("--scientific-report-only", action="store_true", help="Только научный отчёт: scientific_report.json и scientific_report.html из имеющихся данных (с --run-llm — вызов DeepSeek)")
    parser.add_argument("--scientific-report-staged", action="store_true", help="Поэтапная генерация scientific report (несколько LLM-вызовов по секциям)")
    parser.add_argument("--report-pdf", action="store_true", help="Дополнительно сгенерировать краткий PDF (по умолчанию только HTML)")
    parser.add_argument("--full", action="store_true", help="Всё по очереди (включая embeddings, deepseek, report-pdf)")
    parser.add_argument(
        "--documents",
        nargs="*",
        metavar="NAME",
        help="Анализ только по выбранным документам: подстрока имени файла (напр. michie) или имя файла. Несколько: --documents michie atkinson или --documents michie,atkinson",
    )
    parser.add_argument(
        "--use-lemmas",
        action="store_true",
        help="Поиск этнонимов и совпадения лексиконов (R, O) по леммам токенов (spaCy); без флага — по словоформам и regex.",
    )
    args = parser.parse_args()
    run_derived = not getattr(args, "no_derived", False)
    documents_filter = None
    if getattr(args, "documents", None):
        documents_filter = []
        for x in args.documents:
            documents_filter.extend([s.strip() for s in str(x).split(",") if s.strip()])
    main(
        use_ocr_symspell=args.ocr_symspell,
        n_jobs=max(1, args.jobs),
        from_cache=args.from_cache,
        from_db=args.from_db,
        export_excel=args.export_excel,
        build_evidence_pack=args.build_evidence_pack,
        run_embeddings=args.run_embeddings,
        run_llm=args.run_llm,
        run_deepseek=args.run_deepseek,
        run_derived=run_derived,
        report_pdf=args.report_pdf,
        full=args.full,
        report_only=getattr(args, "report_only", False),
        scientific_report_only=getattr(args, "scientific_report_only", False),
        scientific_report_staged=getattr(args, "scientific_report_staged", False),
        documents_filter=documents_filter,
        use_lemmas=getattr(args, "use_lemmas", False),
    )
