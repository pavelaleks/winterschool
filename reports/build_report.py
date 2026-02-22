"""
Генератор главного отчёта output/report.html: офлайн, интерактивные контролы, аналитические блоки из llm_memos.
Нормировка: на 10k предложений (единый стандарт для всех графиков).
"""

import math
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

# Локальные ресурсы — относительные пути от output/report.html
ASSETS_REL = "assets"


def _json_safe(obj: Any) -> Any:
    """Делает данные сериализуемыми в JSON: NaN/Inf → null, numpy → native."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    try:
        import numpy as np
        if isinstance(obj, (np.floating, np.integer)):
            if isinstance(obj, np.floating) and (np.isnan(obj) or np.isinf(obj)):
                return None
            return float(obj) if isinstance(obj, np.floating) else int(obj)
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if hasattr(obj, "isoformat"):  # datetime
        return obj.isoformat()
    return obj


def _cluster_umap_dict(cluster_validation: Optional[Dict]) -> Optional[Dict]:
    """Из результата run_embeddings_pipeline извлекает umap_xy и labels для scatter."""
    if not cluster_validation or not isinstance(cluster_validation, dict):
        return None
    xy = cluster_validation.get("umap_xy")
    labels = cluster_validation.get("labels")
    if not xy or not labels:
        return None
    return {"umap_xy": xy, "labels": labels}


def _h(s: str) -> str:
    """Экранирование для HTML (UTF-8)."""
    if s is None or not isinstance(s, str):
        return ""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _build_keyness_kwic(
    keyness_tables: Dict[str, pd.DataFrame],
    evidence_rows: List[Dict],
    top_words_per_table: int = 10,
    kwic_per_word: int = 3,
    context_chars: int = 50,
) -> Dict[str, List[Dict]]:
    """Для каждой таблицы keyness — топ слов и до kwic_per_word примеров (left, node, right, source_pointer) из evidence."""
    result = {}
    if not evidence_rows:
        return result
    # Индекс предложений по словам (lower) для быстрого поиска
    sent_by_word = defaultdict(list)
    for r in evidence_rows:
        sent = (r.get("sentence_text") or "").strip()
        if len(sent) < 10:
            continue
        ptr = r.get("source_pointer") or (str(r.get("file_name", "")) + "#" + str(r.get("sent_idx", "")))
        low = sent.lower()
        for w in set(low.split()):
            if len(w) >= 3:
                sent_by_word[w].append({"sent": sent, "source_pointer": ptr})
    for name, df in (keyness_tables or {}).items():
        if df is None or df.empty or "word" not in df.columns:
            continue
        rows = []
        for _, row in df.head(top_words_per_table).iterrows():
            word = (row.get("word") or "").strip().lower()
            if not word:
                continue
            g2 = row.get("G2") if "G2" in df.columns else row.get("g2")
            examples = sent_by_word.get(word, [])[:kwic_per_word]
            kwic_list = []
            for ex in examples:
                s = ex["sent"]
                idx = s.lower().find(word)
                if idx < 0:
                    kwic_list.append({"left": s[:context_chars], "node": word, "right": s[-context_chars:] if len(s) > context_chars else "", "source_pointer": ex["source_pointer"]})
                else:
                    left = s[max(0, idx - context_chars):idx].strip()
                    right = s[idx + len(word):idx + len(word) + context_chars].strip()
                    kwic_list.append({"left": left, "node": word, "right": right, "source_pointer": ex["source_pointer"]})
            rows.append({"word": word, "G2": g2, "kwic": kwic_list})
        result[name] = rows
    return result


def _build_ethnonym_variants(evidence_rows: List[Dict]) -> List[Dict]:
    """Агрегат: canonical (ethnos_norm) → варианты написания (ethnos_raw) с count и долей."""
    from collections import Counter
    by_canonical = defaultdict(Counter)
    for r in evidence_rows:
        canon = (r.get("ethnos_norm") or r.get("P") or "").strip()
        raw = (r.get("ethnos_raw") or r.get("mention") or canon or "").strip()
        if not canon:
            continue
        by_canonical[canon][raw] += 1
    out = []
    for canon in sorted(by_canonical.keys()):
        c = by_canonical[canon]
        total = sum(c.values())
        for variant, cnt in c.most_common(15):
            pct = round(100 * cnt / total, 1) if total else 0
            out.append({"canonical": canon, "variant": variant, "count": cnt, "pct": pct, "total": total})
    return out


def _prepare_report_data(
    corpus: List[Dict],
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    piro_raw: List[Dict],
    piro_clean: List[Dict],
    keyness_tables: Dict[str, pd.DataFrame],
    norm_ethnos: Dict,
    norm_R: Dict,
    norm_O: Dict,
    essentialization_table: Dict[str, int],
    essentialization_examples: Dict[str, List[Dict]],
    interaction_edges: List[Dict],
    comention_raw: Dict,
    evidence_rows: List[Dict],
    cluster_validation: Optional[Dict],
    memos: Dict[str, Dict],
    top_n_options: List[int],
    confidence_options: List[float],
) -> Dict[str, Any]:
    """Подготовка JSON для отчёта: все варианты raw/normalized, include/exclude noise, topN."""
    num_docs = len(corpus)
    num_sents = sum(len(d.get("sentences", [])) for d in corpus)
    n_raw = len(raw_df)
    n_clean = len(clean_df)
    n_noise = n_raw - n_clean

    # Этносы: raw counts (incl/excl noise), normalized (rate per 10k)
    ethnos_raw_incl = dict(Counter(r.get("P") or r.get("ethnos_norm", "") for r in piro_raw if (r.get("P") or r.get("ethnos_norm"))))
    ethnos_raw_excl = dict(Counter(r.get("P") or r.get("ethnos_norm", "") for r in piro_clean if (r.get("P") or r.get("ethnos_norm"))))
    ethnos_norm_incl = {k: v.get("mean_rate_per_10k") for k, v in norm_ethnos.items()} if norm_ethnos else {}
    # normalized excl noise — из clean_df уже считается в norm_ethnos если norm_ethnos от clean
    ethnos_norm_excl = ethnos_norm_incl  # мы считаем norm_* по clean_df в main

    R_raw_incl = dict(Counter(r.get("R", "") for r in piro_raw))
    R_raw_excl = dict(Counter(r.get("R", "") for r in piro_clean))
    R_norm = {k: v.get("mean_rate_per_10k") for k, v in norm_R.items()} if norm_R else {}
    O_raw_incl = dict(Counter(r.get("O_situation", "") for r in piro_raw))
    O_raw_excl = dict(Counter(r.get("O_situation", "") for r in piro_clean))
    O_norm = {k: v.get("mean_rate_per_10k") for k, v in norm_O.items()} if norm_O else {}

    # Keyness: по имени таблицы — топ-10/15/25 слов (word, G2)
    keyness_series = {}
    for name, df in keyness_tables.items():
        if df is None or df.empty:
            continue
        keyness_series[name] = {
            n: df.head(n).to_dict(orient="records") for n in top_n_options
        }

    # Co-mention: список рёбер {a, b, weight}
    comention_edges = []
    for a, targets in (comention_raw or {}).items():
        for b, w in targets.items():
            if a != b and w > 0:
                comention_edges.append({"a": a, "b": b, "weight": w})

    # Interaction: уже список с examples
    interaction_list = []
    for e in (interaction_edges or []):
        ex = e.get("examples", [])
        interaction_list.append({
            "src": e.get("src"), "dst": e.get("dst"), "type": e.get("type"),
            "count": e.get("count", 0),
            "example_1": ex[0][:200] if len(ex) > 0 else "",
            "example_2": ex[1][:200] if len(ex) > 1 else "",
            "example_3": ex[2][:200] if len(ex) > 2 else "",
        })

    keyness_kwic = _build_keyness_kwic(keyness_tables, evidence_rows, top_words_per_table=10, kwic_per_word=3)
    ethnonym_variants = _build_ethnonym_variants(evidence_rows)

    return {
        "corpus": {"num_docs": num_docs, "num_sents": num_sents, "n_raw": n_raw, "n_clean": n_clean, "n_noise": n_noise},
        "ethnos": {"raw_incl": ethnos_raw_incl, "raw_excl": ethnos_raw_excl, "norm_incl": ethnos_norm_incl, "norm_excl": ethnos_norm_excl},
        "R": {"raw_incl": R_raw_incl, "raw_excl": R_raw_excl, "norm": R_norm},
        "O": {"raw_incl": O_raw_incl, "raw_excl": O_raw_excl, "norm": O_norm},
        "keyness": keyness_series,
        "keyness_kwic": keyness_kwic,
        "essentialization": {"counts": essentialization_table, "examples": essentialization_examples},
        "interaction_edges": interaction_list,
        "comention_edges": comention_edges,
        "evidence": evidence_rows,
        "ethnonym_variants": ethnonym_variants,
        "cluster_validation": (cluster_validation or {}).get("validation", cluster_validation or {}) if isinstance(cluster_validation, dict) else (cluster_validation or {}),
        "cluster_umap": _cluster_umap_dict(cluster_validation),
        "memos": memos,
        "top_n_options": top_n_options,
        "confidence_options": confidence_options,
    }


def _prepare_report_data_with_indices(
    base_data: Dict[str, Any],
    derived_indices: Optional[Dict] = None,
    synthesis: Optional[Dict] = None,
    derived_profiles: Optional[List[Dict]] = None,
    derived_stats_tests: Optional[Dict] = None,
    derived_correlations: Optional[List[Dict]] = None,
    derived_clusters: Optional[List[Dict]] = None,
    scientific_report_sections: Optional[Dict[str, str]] = None,
    llm_enabled: bool = True,
    api_base: str = "",
) -> Dict[str, Any]:
    """Дополняет report_data индексами, синтезом, профилями этносов, научным отчётом ИИ и настройками дашборда."""
    out = dict(base_data)
    out["derived_indices"] = derived_indices or {}
    out["synthesis"] = synthesis or {}
    out["derived_profiles"] = derived_profiles or []
    out["derived_stats_tests"] = derived_stats_tests or {}
    out["derived_correlations"] = derived_correlations or []
    out["derived_clusters"] = derived_clusters or []
    out["scientific_report_sections"] = scientific_report_sections or {}
    out["llm_enabled"] = llm_enabled
    out["api_base"] = api_base or ""
    return out


def _section_analytics(data: Dict) -> Dict[str, str]:
    """Краткая аналитика под каждой таблицей/графиком: что показывает блок и на что обратить внимание."""
    c = data.get("corpus", {})
    eth = data.get("ethnos", {})
    r = data.get("R", {})
    o = data.get("O", {})
    keyness = data.get("keyness", {})
    ess = data.get("essentialization", {})
    edges = data.get("interaction_edges", [])
    comention = data.get("comention_edges", [])
    di = data.get("derived_indices", {})
    cv = data.get("cluster_validation", {})

    n_docs = c.get("num_docs", 0)
    n_sents = c.get("num_sents", 0)
    n_clean = c.get("n_clean", 0)
    n_noise = c.get("n_noise", 0)
    n_raw = c.get("n_raw", 1)

    # Топ этносов по сырым частотам (excl noise)
    raw_excl = eth.get("raw_excl") or {}
    top_ethnos = sorted(raw_excl.items(), key=lambda x: -(x[1] or 0))[:5]
    top_ethnos_str = ", ".join(e[0] for e in top_ethnos if e[0]) or "—"

    # Топ R
    r_excl = r.get("raw_excl") or {}
    top_r = sorted(r_excl.items(), key=lambda x: -(x[1] or 0))[:3]
    top_r_str = ", ".join(x[0] for x in top_r if x[0]) or "—"

    # Топ O
    o_excl = o.get("raw_excl") or {}
    top_o = sorted(o_excl.items(), key=lambda x: -(x[1] or 0))[:3]
    top_o_str = ", ".join(x[0] for x in top_o if x[0]) or "—"

    analytics = {}

    analytics["corpus"] = (
        f"Таблица отражает объём корпуса ({n_docs} документов, {n_sents} предложений) и результат фильтра шума: "
        f"из {n_raw} упоминаний этнонимов после удаления колонтитулов и позиционной дедупликации осталось {n_clean}. "
        f"Исключено {n_noise} записей ({round(100 * n_noise / n_raw, 1) if n_raw else 0}%). "
        "Все последующие распределения и индексы считаются по очищенной выборке."
    )

    analytics["distributions"] = (
        f"Графики показывают распределение упоминаний по этносам и по типам репрезентации (R) и ситуации (O). "
        f"По частотам лидируют этносы: {top_ethnos_str}. "
        f"По типу репрезентации преобладают: {top_r_str}; по ситуации: {top_o_str}. "
        "В режиме «на 10k предл.» значения нормированы на объём документа для сопоставимости текстов разной длины."
    )

    oi = (di.get("OI") or {}).get("raw_OI") or {}
    as_ = di.get("AS") or {}
    ed = di.get("ED") or {}
    eps = di.get("EPS") or {}
    n_eth = len(set(list(oi.keys()) + list(ed.keys()) + list(eps.keys())))
    analytics["indices"] = (
        "Таблица и графики — производные индексы ориентализации: OI (доля негатив+экзотика), AS (асимметрия агентности), ED (эссенциализация), EPS (перекос негатив/позитив). "
        f"Рассчитаны по {n_eth} этносам. "
        "Ранжирование по OI показывает, какие группы сильнее представлены в негативно/экзотизирующем дискурсе. Диаграммы OI–AS и ED–EPS помогают увидеть кластеры (например, высокий OI при низком AS)."
    )

    n_keyness = len(keyness)
    analytics["keyness"] = (
        f"Таблицы keyness по {n_keyness} подкорпусам: выделены слова с наибольшей статистической значимостью (G2) относительно референсного корпуса. "
        "Топ-слова — кандидаты на маркеры ориенталистского дискурса (негативизация, экзотизация, нейтральные контексты). "
        "Рекомендуется проверить конкорданс по нескольким топ-словам вручную для интерпретации."
    )

    ess_counts = ess.get("counts") or {}
    n_ess_ethnos = len(ess_counts)
    top_ess = sorted(ess_counts.items(), key=lambda x: -(x[1] or 0))[:3]
    top_ess_str = ", ".join(f"{e[0]} ({e[1]})" for e in top_ess if e[0]) or "—"
    analytics["essentialization"] = (
        f"Частоты эссенциализирующих конструкций (паттерны типа «the X are», «X by nature») по {n_ess_ethnos} этносам. "
        f"Наибольшие частоты: {top_ess_str}. "
        "Ниже приведены примеры предложений по этносам для качественной верификации; источник (source_pointer) позволяет найти фрагмент в тексте."
    )

    n_inter = len(edges)
    n_com = len(comention)
    analytics["networks"] = (
        f"Сеть interaction: {n_inter} направленных рёбер (субъект–глагол–объект по словарю глаголов взаимодействия). "
        f"Co-mention: {n_com} пар совместного упоминания в контексте. "
        "Таблицы содержат примеры контекстов по рёбрам. Interaction отражает, кто кого «действует» в дискурсе; co-mention — какие этносы часто оказываются в одном контексте. "
        "<strong>Ограничение:</strong> разрежённость графа interaction может отражать узкий словарь (resources/interaction_verbs.yml) и жёсткие правила извлечения SVO, а не фактическое отсутствие взаимодействий в текстах."
    )

    evidence = data.get("evidence") or []
    analytics["evidence"] = (
        f"Таблица фрагментов для ручной разметки и верификации ({len(evidence)} строк). "
        "Используйте сортировку и поиск DataTables. Поля source_pointer и file_name позволяют найти предложение в исходном тексте. "
        "Рекомендуется выборочная проверка разметки R/O и границ эссенциализации."
    )

    if cv and (cv.get("n_points") or cv.get("silhouette_score") is not None):
        n_pts = cv.get("n_points", "—")
        sil = cv.get("silhouette_score")
        sil_str = f"{sil:.3f}" if sil is not None else "—"
        analytics["embedding"] = (
            f"Метрики кластеризации контекстов упоминаний: выборка {n_pts} точек, silhouette score {sil_str}. "
            "UMAP визуализирует семантическую близость контекстов; цвет — кластер. "
            "Используется для валидации расхождения негатив/экзотика по разным зонам представления."
        )
    else:
        analytics["embedding"] = (
            "Блок заполняется при запуске пайплайна с флагом --run-embeddings. "
            "Кластеризация эмбеддингов контекстов позволяет проверить, образуют ли негативные и экзотизирующие упоминания отдельные семантические кластеры."
        )

    return analytics


def _executive_summary_bullets(data: Dict) -> List[str]:
    """5–7 буллетов для Executive summary с указанием источника цифр."""
    c = data.get("corpus", {})
    bullets = [
        f"Корпус: {c.get('num_docs', 0)} документов, {c.get('num_sents', 0)} предложений (источник: предобработка).",
        f"Упоминаний этнонимов (raw): {c.get('n_raw', 0)}; после исключения шума: {c.get('n_clean', 0)} (источник: mentions + noise filter).",
        f"Исключено как шум: {c.get('n_noise', 0)} ({round(c.get('n_noise', 0) / max(c.get('n_raw', 1), 1) * 100, 1)}%) — колонтитулы, индексные строки, позиционная дедупликация.",
        "Распределения R (тип репрезентации) и O (ситуация) нормированы на 10k предложений по документам для сопоставимости.",
        "Keyness (G2) считан по контентным словам (NOUN/ADJ/VERB) с отсечением OCR-подобных токенов; топ-30 — кандидаты на маркеры ориентализма.",
        "Сети: co-mention (совместное упоминание в контексте) и interaction (субъект–глагол–объект по словарю глаголов); для interaction исключены индексоподобные предложения.",
    ]
    if data.get("cluster_validation") and data["cluster_validation"].get("n_points"):
        bullets.append("Кластеризация эмбеддингов контекстов (UMAP + HDBSCAN/KMeans) использована для валидации расхождения негатив/экзотика по семантическим зонам.")
    return bullets


def _format_token_usage_html(token_usage: Optional[Dict]) -> str:
    """Формирует HTML-блок раздела «Использование LLM (токены)» для отчёта."""
    if not token_usage or not isinstance(token_usage, dict):
        return (
            "<p class=\"explain\">В этом запуске LLM не использовался. Учёт токенов появляется после полного прогона с флагом <code>--run-llm</code> (или <code>--full</code>).</p>"
        )
    total = token_usage.get("total_tokens", 0) or 0
    n_used = token_usage.get("n_calls_llm_used", 0) or 0
    n_without = token_usage.get("n_calls_without_llm", 0) or 0
    if total == 0 and n_used == 0:
        return (
            "<p class=\"explain\">LLM в этом запуске не вызывался (нет ключа или ошибка). Счётчик: <code>output/metadata/llm_token_count.json</code>.</p>"
        )
    pt = token_usage.get("total_prompt_tokens", 0) or 0
    ct = token_usage.get("total_completion_tokens", 0) or 0
    last = token_usage.get("last_updated", "")
    cost_usd = token_usage.get("estimated_cost_usd")
    if cost_usd is None and (pt or ct):
        # DeepSeek deepseek-chat: $0.27/1M input, $1.10/1M output (см. api-docs.deepseek.com)
        cost_usd = (pt / 1_000_000) * 0.27 + (ct / 1_000_000) * 1.10
    cost_str = f"${cost_usd:.4f}" if cost_usd is not None and cost_usd >= 0 else "—"
    rows = [
        ["Всего токенов", str(total)],
        ["Токенов запроса (промпт)", str(pt)],
        ["Токенов ответа (completion)", str(ct)],
        ["Примерная стоимость (DeepSeek, USD)", cost_str],
        ["Вызовов с использованием LLM", str(n_used)],
        ["Вызовов без LLM (ошибка/нет ключа)", str(n_without)],
        ["Последнее обновление счётчика", _h(last)],
    ]
    html = '<table class="display"><thead><tr><th>Показатель</th><th>Значение</th></tr></thead><tbody>'
    for r in rows:
        html += f"<tr><td>{_h(r[0])}</td><td>{_h(r[1])}</td></tr>"
    html += "</tbody></table>"
    html += '<p class="explain">Источник: <code>output/metadata/llm_token_count.json</code>. Счётчик сбрасывается в начале каждого запуска с <code>--run-llm</code>. Стоимость рассчитана по тарифам DeepSeek (deepseek-chat: $0.27/1M ввод, $1.10/1M вывод; актуальные цены: <a href="https://api-docs.deepseek.com/quick_start/pricing-details-usd" target="_blank" rel="noopener">api-docs.deepseek.com</a>).</p>'
    return html


def build_report(
    corpus: List[Dict],
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    piro_raw: List[Dict],
    piro_clean: List[Dict],
    keyness_tables: Dict[str, pd.DataFrame],
    norm_ethnos: Dict,
    norm_R: Dict,
    norm_O: Dict,
    essentialization_table: Dict[str, int],
    essentialization_examples: Dict[str, List[Dict]],
    interaction_edges: List[Dict],
    comention_raw: Dict,
    evidence_df: Optional[pd.DataFrame],
    cluster_validation: Optional[Dict],
    llm_memos: Optional[Dict[str, Dict]] = None,
    derived_indices: Optional[Dict] = None,
    synthesis: Optional[Dict] = None,
    derived_profiles: Optional[List[Dict]] = None,
    derived_stats_tests: Optional[Dict] = None,
    derived_correlations: Optional[List[Dict]] = None,
    derived_clusters: Optional[List[Dict]] = None,
    scientific_report_sections: Optional[Dict[str, str]] = None,
    llm_enabled: bool = True,
    api_base: str = "",
    token_usage: Optional[Dict] = None,
    run_passport: Optional[Dict] = None,
    output_path: Optional[Path] = None,
    assets_dir: Optional[Path] = None,
) -> str:
    """
    Собирает output/report.html: офлайн, UTF-8, интерактивные контролы (режим raw/normalized, topN, include noise),
    аналитические блоки под каждым разделом из llm_memos.
    """
    output_path = output_path or Path(__file__).resolve().parent.parent / "output" / "report.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assets_dir = assets_dir or output_path.parent / "assets"
    # Гарантируем наличие assets (jQuery, Plotly, DataTables) — иначе в браузере отчёт пустой
    try:
        from reports.report_assets import ensure_report_assets
        ensure_report_assets(output_path.parent)
    except Exception:
        pass

    memos = llm_memos or {}
    evidence_rows = []
    if evidence_df is not None and not evidence_df.empty:
        evidence_rows = evidence_df.fillna("").to_dict(orient="records")
        for r in evidence_rows:
            for k, v in list(r.items()):
                if isinstance(v, str) and len(v) > 500:
                    r[k] = v[:500] + "…"
            if not r.get("source_pointer"):
                r["source_pointer"] = str(r.get("file_name") or "") + "#" + str(r.get("sent_idx") or "")

    report_data = _prepare_report_data(
        corpus=corpus,
        raw_df=raw_df,
        clean_df=clean_df,
        piro_raw=piro_raw,
        piro_clean=piro_clean,
        keyness_tables=keyness_tables,
        norm_ethnos=norm_ethnos,
        norm_R=norm_R,
        norm_O=norm_O,
        essentialization_table=essentialization_table,
        essentialization_examples=essentialization_examples,
        interaction_edges=interaction_edges,
        comention_raw=comention_raw,
        evidence_rows=evidence_rows,
        cluster_validation=cluster_validation,
        memos=memos,
        top_n_options=[10, 15, 25, 30],
        confidence_options=[0.4, 0.6, 0.8],
    )
    report_data = _prepare_report_data_with_indices(
        report_data,
        derived_indices=derived_indices,
        synthesis=synthesis,
        derived_profiles=derived_profiles,
        derived_stats_tests=derived_stats_tests,
        derived_correlations=derived_correlations,
        derived_clusters=derived_clusters,
        scientific_report_sections=scientific_report_sections,
        llm_enabled=llm_enabled,
        api_base=api_base,
    )
    report_data["timestamp"] = datetime.now().isoformat()
    report_data["run_passport"] = run_passport or {}

    # Scatter OI vs AS, ED vs EPS: matplotlib с adjustText, jitter, квадранты, SVG/PNG
    if derived_indices:
        try:
            from reports.scatter_plots import build_indices_scatter_plots
            scatter_paths = build_indices_scatter_plots(derived_indices, output_path.parent)
            report_data["scatter_oi_as_svg"] = scatter_paths.get("oi_vs_as_svg")
            report_data["scatter_oi_as_png"] = scatter_paths.get("oi_vs_as_png")
            report_data["scatter_ed_eps_svg"] = scatter_paths.get("ed_vs_eps_svg")
            report_data["scatter_ed_eps_png"] = scatter_paths.get("ed_vs_eps_png")
        except Exception:
            report_data["scatter_oi_as_svg"] = None
            report_data["scatter_oi_as_png"] = None
            report_data["scatter_ed_eps_svg"] = None
            report_data["scatter_ed_eps_png"] = None
    else:
        report_data["scatter_oi_as_svg"] = None
        report_data["scatter_oi_as_png"] = None
        report_data["scatter_ed_eps_svg"] = None
        report_data["scatter_ed_eps_png"] = None

    exec_bullets = _executive_summary_bullets(report_data)
    section_analytics = _section_analytics(report_data)
    analytics_corpus = _h(section_analytics.get("corpus", ""))
    analytics_distributions = _h(section_analytics.get("distributions", ""))
    analytics_indices = _h(section_analytics.get("indices", ""))
    analytics_keyness = _h(section_analytics.get("keyness", ""))
    analytics_essentialization = _h(section_analytics.get("essentialization", ""))
    analytics_networks = _h(section_analytics.get("networks", ""))
    analytics_evidence = _h(section_analytics.get("evidence", ""))
    analytics_embedding = _h(section_analytics.get("embedding", ""))

    token_usage_html = _format_token_usage_html(token_usage)

    # Научный отчёт ИИ: плотный текст по секциям (статический HTML)
    sci_sections = report_data.get("scientific_report_sections") or {}
    sci_titles = {
        "corpus": "1. Корпус и данные",
        "distributions": "2. Распределения: этносы, репрезентация (R), ситуация (O)",
        "keyness": "3. Keyness и маркеры дискурса",
        "essentialization": "4. Эссенциализация",
        "networks": "5. Сети взаимодействий и co-mention",
        "indices": "6. Количественные индексы (OI, ED, EPS, AS)",
        "ethnic_profiles": "7. Профили этносов и статистические проверки",
        "conclusion": "8. Заключение",
    }
    scientific_report_html_parts = []
    for key in ["corpus", "distributions", "keyness", "essentialization", "networks", "indices", "ethnic_profiles", "conclusion"]:
        if key.startswith("_"):
            continue
        text = (sci_sections.get(key) or "").strip()
        if not text:
            continue
        title = sci_titles.get(key, key)
        paras = [_h(p.strip()) for p in text.split("\n\n") if p.strip()]
        scientific_report_html_parts.append(
            f"<h3>{_h(title)}</h3>" + "".join(f"<p>{p}</p>" for p in paras)
        )
    if scientific_report_html_parts:
        scientific_report_html = (
            '<p class="explain">Плотный аналитический текст, сгенерированный ИИ по данным таблиц и графиков отчёта. '
            'Отдельный файл: <a href="scientific_report.html">scientific_report.html</a>.</p>'
            + "".join(f'<div class="analytics report-prose">{p}</div>' for p in scientific_report_html_parts)
        )
    else:
        scientific_report_html = (
            '<p class="explain">Научный отчёт не сформирован. Запустите пайплайн с <code>--run-llm</code> для генерации '
            'плотного аналитического текста по данным (или откройте <a href="scientific_report.html">scientific_report.html</a> после генерации).</p>'
        )

    # Статическая таблица корпуса (видна даже если JS не загрузится)
    c = report_data.get("corpus", {})
    corpus_table_static = (
        '<table class="display"><thead><tr><th>Показатель</th><th>Значение</th></tr></thead><tbody>'
        f'<tr><td>Документов</td><td>{c.get("num_docs", 0)}</td></tr>'
        f'<tr><td>Предложений</td><td>{c.get("num_sents", 0)}</td></tr>'
        f'<tr><td>Упоминаний (raw)</td><td>{c.get("n_raw", 0)}</td></tr>'
        f'<tr><td>После очистки</td><td>{c.get("n_clean", 0)}</td></tr>'
        f'<tr><td>Исключено шум</td><td>{c.get("n_noise", 0)}</td></tr></tbody></table>'
    )

    # Пути к локальным скриптам (относительно report.html)
    jq = f"{ASSETS_REL}/jquery-3.7.1.min.js"
    plotly = f"{ASSETS_REL}/plotly-2.27.0.min.js"
    dt_js = f"{ASSETS_REL}/jquery.dataTables.min.js"
    dt_css = f"{ASSETS_REL}/jquery.dataTables.min.css"

    # Таблица вариантов написания этнонимов (alias → canonical)
    ethnonym_variants = report_data.get("ethnonym_variants") or []
    ethnonym_variants_rows = []
    for v in ethnonym_variants[:250]:
        ethnonym_variants_rows.append(
            f"<tr><td>{_h(v.get('canonical', ''))}</td><td>{_h(v.get('variant', ''))}</td><td>{v.get('count', 0)}</td><td>{v.get('pct', 0)}%</td></tr>"
        )
    ethnonym_variants_table_html = (
        '<table class="display"><thead><tr><th>Canonical</th><th>Вариант написания</th><th>N</th><th>%</th></tr></thead><tbody>'
        + "".join(ethnonym_variants_rows) + "</tbody></table>"
        if ethnonym_variants_rows
        else "<p>Нет данных (в Evidence нужны колонки ethnos_norm и ethnos_raw).</p>"
    )

    report_data_safe = _json_safe(report_data)
    try:
        data_json = json.dumps(report_data_safe, ensure_ascii=False)
    except (TypeError, ValueError):
        data_json = json.dumps(report_data_safe, ensure_ascii=False, default=lambda x: None)
    # Иначе вставка в <script> ломает разбор: браузер видит </script> внутри строки
    if "</script>" in data_json:
        data_json = data_json.replace("</script>", "<\\/script>")
    report_generated = report_data.get("timestamp") or datetime.now().isoformat()
    if isinstance(report_generated, str) and "T" in report_generated:
        try:
            t = datetime.fromisoformat(report_generated.replace("Z", "+00:00"))
            report_generated = t.strftime("%d.%m.%Y %H:%M")
        except Exception:
            pass

    c = report_data.get("corpus", {})
    eth = report_data.get("ethnos") or {}
    raw_excl = eth.get("raw_excl")
    if isinstance(raw_excl, dict):
        n_ethnos = len(set(raw_excl.keys()) - {""})
    elif isinstance(raw_excl, list):
        n_ethnos = len(set((x[0] if isinstance(x, (list, tuple)) else x) for x in raw_excl) - {""})
    else:
        n_ethnos = 0
    key_numbers_html = (
        f'<div class="key-numbers">'
        f'<span>{c.get("num_docs", 0)} док.</span>'
        f'<span>{c.get("num_sents", 0)} предл.</span>'
        f'<span>{c.get("n_clean", 0)} упоминаний</span>'
        f'<span>{n_ethnos} этносов</span>'
        f'</div>'
    )

    passport = report_data.get("run_passport") or {}
    if passport:
        ts = passport.get("run_ts", "")
        if ts and "T" in ts:
            try:
                from datetime import datetime as dt
                t = dt.fromisoformat(ts.replace("Z", "+00:00"))
                ts = t.strftime("%d.%m.%Y %H:%M")
            except Exception:
                pass
        docs_str = ", ".join(passport.get("input_documents") or []) or "—"
        nf = passport.get("noise_filter") or {}
        filter_str = ", ".join(f"{k}={v}" for k, v in nf.items()) or "—"
        passport_html = (
            '<div class="method-note" id="sec-passport" style="margin-top:0.5rem;">'
            "<strong>Паспорт прогона</strong> (единый источник цифр): "
            f"дата/время: {ts}; входные документы: {_h(docs_str)}; "
            f"документов: {passport.get('num_docs', 0)}, предложений: {passport.get('num_sents', 0)}, "
            f"упоминаний raw: {passport.get('n_raw', 0)}, после очистки: {passport.get('n_clean', 0)}, "
            f"исключено шум: {passport.get('n_noise', 0)} ({passport.get('noise_pct', 0)}%). "
            f"Параметры фильтра шума: {_h(filter_str)}. "
            "Все таблицы и графики ниже рассчитаны по очищенной выборке."
            "</div>"
        )
    else:
        passport_html = ""

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Отчёт: анализ травелогов о Сибири</title>
<link rel="stylesheet" href="{dt_css}">
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: Georgia, serif; margin: 0; padding: 0; background: #fafafa; color: #222; }}
.layout {{ display: flex; min-height: 100vh; }}
.toc {{ width: 220px; background: #2c3e50; color: #ecf0f1; padding: 1rem; position: sticky; top: 0; height: 100vh; overflow-y: auto; }}
.toc ul {{ list-style: none; padding: 0; }}
.toc a {{ color: #bdc3c7; text-decoration: none; display: block; padding: 0.35rem 0; }}
.toc a:hover {{ color: #fff; }}
.main {{ flex: 1; padding: 2rem 3rem; max-width: 1200px; }}
.report-section {{ margin-bottom: 2.5rem; }}
.report-section h2 {{ color: #1a5276; border-bottom: 1px solid #1a5276; padding-bottom: 0.3rem; }}
.controls {{ margin: 1rem 0; display: flex; flex-wrap: wrap; gap: 1rem; align-items: center; }}
.controls label {{ margin-right: 0.5rem; }}
.explain {{ color: #555; margin-bottom: 0.8rem; font-size: 0.95rem; }}
.analytics {{ background: #e8f6f3; padding: 1rem; margin-top: 1rem; border-radius: 4px; font-size: 0.95rem; }}
.analytics ul {{ margin: 0.3rem 0; }}
.analysis-block {{ background: #e8f6f3; padding: 1rem; margin-top: 0.5rem; border-radius: 4px; font-size: 0.95rem; border: 1px solid #b8d4ce; }}
.analysis-block h3 {{ margin-top: 0; color: #1a5276; font-size: 1.05rem; }}
.analysis-block h4 {{ margin: 1rem 0 0.3rem; color: #2c3e50; font-size: 0.98rem; }}
.analysis-block .memo-text {{ margin: 0.5rem 0; }}
.toggle-analysis {{ margin-top: 1rem; padding: 0.4rem 0.8rem; cursor: pointer; background: #34495e; color: #fff; border: none; border-radius: 4px; font-size: 0.9rem; }}
.toggle-analysis:hover {{ background: #2c3e50; }}
table {{ border-collapse: collapse; margin: 1rem 0; font-size: 0.9rem; }}
th, td {{ border: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: left; }}
th {{ background: #34495e; color: #fff; }}
.plot-container {{ margin: 1rem 0; min-height: 300px; }}
.exec-bullets {{ margin: 1rem 0; }}
.exec-bullets li {{ margin: 0.4rem 0; }}
.key-numbers {{ display: flex; flex-wrap: wrap; gap: 1rem; margin: 1rem 0; }}
.key-numbers span {{ background: #e8f6f3; padding: 0.5rem 1rem; border-radius: 6px; font-weight: 600; color: #1a5276; }}
.report-footer {{ margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #ddd; font-size: 0.85rem; color: #666; }}
.aux-links {{ margin: 0.5rem 0; }}
.aux-links a {{ margin-right: 1rem; }}
.method-note {{ font-size: 0.88rem; color: #555; margin-top: 0.5rem; margin-bottom: 1rem; padding: 0.5rem 0.75rem; background: #f5f5f5; border-left: 3px solid #1a5276; border-radius: 0 4px 4px 0; }}
.evidence-table-scroll {{ max-height: 70vh; overflow: auto; margin: 1rem 0; border: 1px solid #ddd; border-radius: 4px; }}
.btn-evidence {{ margin: 0.5rem 0; padding: 0.5rem 1rem; background: #1a5276; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-size: 1rem; }}
.btn-evidence:hover {{ background: #2c3e50; }}
.evidence-modal-overlay {{ display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); z-index: 9998; }}
.evidence-modal-overlay.show {{ display: block; }}
.evidence-modal {{ position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: #fff; border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); z-index: 9999; max-width: 95vw; max-height: 90vh; overflow: hidden; display: flex; flex-direction: column; }}
.evidence-modal .modal-header {{ padding: 1rem 1.5rem; border-bottom: 1px solid #ddd; display: flex; justify-content: space-between; align-items: center; }}
.evidence-modal .modal-body {{ padding: 1rem 1.5rem; overflow: auto; flex: 1; }}
.evidence-modal .modal-filters {{ display: flex; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 1rem; align-items: center; }}
.evidence-modal .modal-filters label {{ margin-right: 0.25rem; }}
.evidence-modal .modal-close {{ background: #666; color: #fff; border: none; padding: 0.35rem 0.75rem; border-radius: 4px; cursor: pointer; }}
@media print {{
  .toc {{ position: static; height: auto; }}
  .toggle-analysis, .recalc-analysis {{ display: none !important; }}
  .main {{ max-width: none; }}
  .evidence-table-scroll {{ max-height: none; overflow: visible; }}
}}
</style>
</head>
<body>
<div class="layout">
<nav class="toc">
<h3>Оглавление</h3>
<ul>
<li><a href="#sec-summary">Краткое резюме</a></li>
<li><a href="#sec-corpus">Качество корпуса</a></li>
<li><a href="#sec-ethnic-profile">Профиль этноса (Ethnic Profile Matrix)</a></li>
<li><a href="#sec-distributions">Распределения</a></li>
<li><a href="#sec-ethnonym-variants">Варианты написания этнонимов</a></li>
<li><a href="#sec-keyness">Keyness</a></li>
<li><a href="#sec-essentialization">Эссенциализация</a></li>
<li><a href="#sec-networks">Сети</a></li>
<li><a href="#sec-indices">Количественные индексы</a></li>
<li><a href="#sec-evidence">Evidence Pack</a></li>
<li><a href="#sec-embedding">Embedding validation</a></li>
<li><a href="#sec-scientific-report">Научный отчёт (текст ИИ)</a></li>
<li><a href="#sec-synthesis">Структурная модель</a></li>
<li><a href="#sec-limits">Ограничения</a></li>
<li><a href="#sec-tokens">Использование LLM (токены)</a></li>
<li><a href="#sec-aux">Вспомогательные визуализации</a></li>
</ul>
</nav>
<main class="main">
<h1>Исследовательский отчёт: травелоги о Сибири</h1>
{passport_html}
{key_numbers_html}
<p>Интерактивные графики и таблицы. Контролы: режим (raw / нормированный), топ-N, включение шума — перерисовывают графики.</p>
<p class="explain" style="background:#fff3cd;padding:0.5rem;border-radius:4px;">Если таблицы и графики ниже пустые, откройте <code>report.html</code> через локальный сервер из папки <code>output</code>: <code>python -m http.server 8000</code>, затем в браузере: <code>http://localhost:8000/report.html</code>. Убедитесь, что рядом с отчётом есть папка <code>assets</code> (jQuery, Plotly, DataTables).</p>
<button type="button" class="btn-evidence" id="btn-open-evidence">Открыть доказательства</button>
<p class="explain">Доказательная база: выборка предложений с этнонимами из <code>derived/evidence_pack_sample.json</code>. Сгенерировать: <code>python tools/build_evidence_layer.py</code>.</p>

<section id="sec-summary" class="report-section">
<h2>Краткое резюме</h2>
<ul class="exec-bullets">
{"".join(f"<li>{_h(b)}</li>" for b in exec_bullets)}
</ul>
</section>

<section id="sec-corpus" class="report-section">
<h2>Качество корпуса</h2>
<p class="explain">Объём корпуса и результат фильтра колонтитулов и позиционной дедупликации. <strong>Актуальные цифры этого прогона</strong> — в блоке «Паспорт прогона» в начале отчёта.</p>
<div id="table-corpus-wrap">{corpus_table_static}</div>
<p class="method-note"><strong>Что показано:</strong> Документы и предложения — из предобработки (сегментация по предложениям). Упоминания — предложения с этнонимом из словаря (resources/ethnonyms.yml). <strong>Шум:</strong> ядро — позиционная дедупликация по (sentence_norm + position_bucket): повторяющиеся строки на одинаковых позициях в документе/корпусе маркируются как header/footer; дополнительно — regex/частотные маркеры (колонтитулы, оглавления, индексные строки). В Evidence Pack видны position_bucket, is_position_header_footer, noise_reason.</p>
<div class="analytics"><strong>Аналитика.</strong> {analytics_corpus}</div>
<button type="button" class="toggle-analysis" data-target="memo-corpus">Скрыть расширенную интерпретацию</button>
<button type="button" class="recalc-analysis" data-block-id="corpus" data-target="memo-corpus">Пересчитать аналитическую интерпретацию</button>
<div class="analysis-block" id="memo-corpus" style="display:block;"><h3>Расширенная интерпретация</h3><div class="memo-text"></div><h4>Проверяемые гипотезы</h4><ul class="hypotheses"></ul><h4>Методологические риски</h4><ul class="pitfalls"></ul><div class="memo-meta" style="font-size:0.85rem;color:#666;margin-top:0.5rem;"></div></div>
</section>

<section id="sec-ethnic-profile" class="report-section">
<h2>Профиль этноса (Ethnic Profile Matrix)</h2>
<p class="explain">Второй аналитический слой: профили по этносам (OI, ED, EPS, AS), нормализация на 10k предложений, effect sizes (delta от взвешенного среднего), статистические проверки и кластеры. Данные из <code>output/derived/</code> и <code>output/tables/ethnic_profiles.*</code>.</p>
<div id="ethnic-profiles-table-wrap"></div>
<p class="method-note"><strong>Таблица:</strong> mentions_raw, mentions_norm, OI, EPS, ED, AS, uncertain_R_share, unknown_O_share, noise_share, OI_delta, ED_delta, EPS_delta, AS_delta.</p>
<div class="plot-container" id="plot-profile-oi-ranking"></div>
<div class="plot-container" id="plot-profile-ed-ranking"></div>
<p class="method-note"><strong>Ранжирование:</strong> Топ этносов по OI и по ED.</p>
<div class="plot-container" id="plot-profile-oi-ed"></div>
<div class="plot-container" id="plot-profile-oi-as"></div>
<div class="plot-container" id="plot-profile-oi-mentions"></div>
<p class="method-note"><strong>Диаграммы рассеяния:</strong> OI vs ED, OI vs AS, OI vs mentions_norm.</p>
<div id="derived-stats-tests-wrap"></div>
<p class="method-note"><strong>Статистические проверки:</strong> Chi-square Ethnos×R, Ethnos×O, Cramér's V; bootstrap 95% CI по топ-этносам.</p>
<div id="derived-clusters-wrap"></div>
<p class="method-note"><strong>Кластеры этносов:</strong> KMeans по (OI, ED, EPS, AS), выбор k по silhouette.</p>
</section>

<section id="sec-indices" class="report-section">
<h2>Количественные индексы ориентализации</h2>
<p class="explain">Единица наблюдения: одно упоминание этнонима в контексте предложения (очищенная выборка). O_locator = file_name + sentence_id (source_pointer) — поиск в источнике; O_scene = тип ситуации (everyday_life, military, trade и т.д.) по лексикону situation_domains.yml.</p>
<p class="explain">Формулы: OI = (negative+exotic)/total; AS = (out−in)/(out+in) по узлу interaction; ED = эссенциализация/total; EPS = (negative−positive)/total.</p>
<details class="explain" style="margin:0.5rem 0;"><summary>Расшифровка индексов</summary><ul style="margin:0.5rem 0;"><li><strong>OI</strong> (Orientalization Index) — доля упоминаний с негативной или экзотизирующей репрезентацией.</li><li><strong>AS</strong> (Agency Score) — асимметрия агентности: кто чаще субъект, кто объект в контекстах взаимодействия.</li><li><strong>ED</strong> (Essentialization Degree) — доля эссенциализирующих конструкций («the X are», «by nature» и т.п.).</li><li><strong>EPS</strong> — перекос негатив/позитив в репрезентации.</li></ul></details>
<div id="indices-table-wrap"></div>
<p class="method-note"><strong>Таблица:</strong> N (число упоминаний), OI, AS, ED, EPS по очищенной выборке; CI — 95% доверительный интервал. <strong>Надёжность:</strong> при N&lt;20 индексы статистически неустойчивы; для выводов уровня статьи рекомендуется порог N≥20 или явная пометка «низкая надёжность».</p>
<div class="plot-container" id="plot-oi-ranking"></div>
<p class="method-note"><strong>График:</strong> Ранжирование этносов по OI. Чем выше столбец, тем чаще группа представлена в негативно или экзотизирующем дискурсе.</p>
<div class="plot-container" id="plot-scatter-oi-as"></div>
<p class="method-note"><strong>Диаграмма рассеяния:</strong> OI (ось X) vs AS (ось Y). Позволяет увидеть кластеры: например, высокий OI при низкой агентности.</p>
<div class="plot-container" id="plot-scatter-ed-eps"></div>
<p class="method-note"><strong>Диаграмма рассеяния:</strong> ED (эссенциализация) vs EPS (перекос негатив/позитив) по этносам.</p>
<div class="analytics"><strong>Аналитика.</strong> {analytics_indices}</div>
</section>

<section id="sec-distributions" class="report-section">
<h2>Распределения: этносы, R, O</h2>
<div class="controls">
<label>Режим: <select id="ctrl-mode"><option value="raw">Сырые частоты</option><option value="normalized">На 10k предл.</option></select></label>
<label>Включить шум: <select id="ctrl-noise"><option value="excl">Нет</option><option value="incl">Да</option></select></label>
<label>Топ этносов: <select id="ctrl-topn"><option value="10">10</option><option value="15">15</option><option value="25">25</option></select></label>
</div>
<div class="plot-container" id="plot-ethnos"></div>
<p class="method-note"><strong>График:</strong> Число упоминаний по каноническим именам этносов. Режим «сырые частоты» или «на 10k предл.» задаётся переключателем; нормировка на 10k делает документы разной длины сопоставимыми.</p>
<div class="plot-container" id="plot-R"></div>
<p class="method-note"><strong>График:</strong> Тип репрезентации (R): neutral, negative, exotic, uncertain. Разметка из пайплайна PIRO (правила + опционально LLM).</p>
<div class="plot-container" id="plot-O"></div>
<p class="method-note"><strong>График:</strong> Ситуация (O) — контекст упоминания: everyday_life, descriptive_scene, mixed, military и др. по разметке PIRO.</p>
<div class="analytics"><strong>Аналитика.</strong> {analytics_distributions}</div>
<button type="button" class="toggle-analysis" data-target="memo-representations">Скрыть расширенную интерпретацию (R)</button>
<button type="button" class="recalc-analysis" data-block-id="representations" data-target="memo-representations">Пересчитать аналитическую интерпретацию</button>
<div class="analysis-block" id="memo-representations" style="display:block;"><h3>Расширенная интерпретация (R)</h3><div class="memo-text"></div><h4>Проверяемые гипотезы</h4><ul class="hypotheses"></ul><h4>Методологические риски</h4><ul class="pitfalls"></ul><div class="memo-meta" style="font-size:0.85rem;color:#666;margin-top:0.5rem;"></div></div>
<button type="button" class="toggle-analysis" data-target="memo-situations">Скрыть расширенную интерпретацию (O)</button>
<button type="button" class="recalc-analysis" data-block-id="situations" data-target="memo-situations">Пересчитать аналитическую интерпретацию</button>
<div class="analysis-block" id="memo-situations" style="display:block;"><h3>Расширенная интерпретация (O)</h3><div class="memo-text"></div><h4>Проверяемые гипотезы</h4><ul class="hypotheses"></ul><h4>Методологические риски</h4><ul class="pitfalls"></ul><div class="memo-meta" style="font-size:0.85rem;color:#666;margin-top:0.5rem;"></div></div>
</section>

<section id="sec-ethnonym-variants" class="report-section">
<h2>Варианты написания этнонимов (alias → canonical)</h2>
<p class="explain">Агрегация в отчёте ведётся по каноническому имени (resources/ethnonyms.yml). Ниже — вклад вариантов написания в корпусе: каноническое имя, форма в тексте, число вхождений и доля (%).</p>
<div id="ethnonym-variants-wrap">{ethnonym_variants_table_html}</div>
<p class="method-note"><strong>Использование:</strong> для верификации и цитирования полезно видеть, какие варианты (kirghiz, kirgis, kyrgyz и т.д.) встречаются и с какой частотой.</p>
</section>

<section id="sec-keyness" class="report-section">
<h2>Keyness (контентные слова, топ-30)</h2>
<div class="controls">
<label>Топ-N: <select id="ctrl-keyness-topn"><option value="10">10</option><option value="15">15</option><option value="25">25</option><option value="30">30</option></select></label>
</div>
<div id="keyness-tables-wrap"></div>
<p class="method-note"><strong>Таблицы:</strong> Для каждого подкорпуса (по типу репрезентации R или по этносу) — слова с наибольшей статистической значимостью (G2) относительно референсного корпуса. Контентные слова (NOUN/ADJ/VERB); OCR-подобные токены отсекаются. Ниже для топ-слов приведены KWIC-примеры (контексты из Evidence) с source_pointer для верификации.</p>
<div class="analytics"><strong>Аналитика.</strong> {analytics_keyness}</div>
<button type="button" class="toggle-analysis" data-target="memo-keyness">Скрыть расширенную интерпретацию</button>
<button type="button" class="recalc-analysis" data-block-id="keyness" data-target="memo-keyness">Пересчитать аналитическую интерпретацию</button>
<div class="analysis-block" id="memo-keyness" style="display:block;"><h3>Расширенная интерпретация</h3><div class="memo-text"></div><h4>Проверяемые гипотезы</h4><ul class="hypotheses"></ul><h4>Методологические риски</h4><ul class="pitfalls"></ul><div class="memo-meta" style="font-size:0.85rem;color:#666;margin-top:0.5rem;"></div></div>
</section>

<section id="sec-essentialization" class="report-section">
<h2>Эссенциализация</h2>
<div id="essentialization-wrap"></div>
<p class="method-note"><strong>Таблица и примеры:</strong> Частоты эссенциализирующих конструкций («the X are», «X by nature», «as a people/tribe» и т.п.) по этносам. Ниже — примеры предложений по каждому этносу для качественной верификации; source_pointer позволяет найти фрагмент в тексте.</p>
<div class="analytics"><strong>Аналитика.</strong> {analytics_essentialization}</div>
<button type="button" class="toggle-analysis" data-target="memo-essentialization">Скрыть расширенную интерпретацию</button>
<button type="button" class="recalc-analysis" data-block-id="essentialization" data-target="memo-essentialization">Пересчитать аналитическую интерпретацию</button>
<div class="analysis-block" id="memo-essentialization" style="display:block;"><h3>Расширенная интерпретация</h3><div class="memo-text"></div><h4>Проверяемые гипотезы</h4><ul class="hypotheses"></ul><h4>Методологические риски</h4><ul class="pitfalls"></ul><div class="memo-meta" style="font-size:0.85rem;color:#666;margin-top:0.5rem;"></div></div>
</section>

<section id="sec-networks" class="report-section">
<h2>Сети: co-mention и interaction</h2>
<div id="networks-wrap"></div>
<p class="method-note"><strong>Две таблицы:</strong> (1) <strong>Interaction</strong> — направленные пары «источник → цель» по словарю глаголов взаимодействия (кто кого «действует» в дискурсе); примеры контекстов в колонках. (2) <strong>Co-mention</strong> — пары этносов, совместно упомянутых в одном контексте, с весом (частота совместной встречаемости).</p>
<div class="analytics"><strong>Аналитика.</strong> {analytics_networks}</div>
<button type="button" class="toggle-analysis" data-target="memo-networks">Скрыть расширенную интерпретацию</button>
<button type="button" class="recalc-analysis" data-block-id="networks" data-target="memo-networks">Пересчитать аналитическую интерпретацию</button>
<div class="analysis-block" id="memo-networks" style="display:block;"><h3>Расширенная интерпретация</h3><div class="memo-text"></div><h4>Проверяемые гипотезы</h4><ul class="hypotheses"></ul><h4>Методологические риски</h4><ul class="pitfalls"></ul><div class="memo-meta" style="font-size:0.85rem;color:#666;margin-top:0.5rem;"></div></div>
</section>

<section id="sec-evidence" class="report-section">
<h2>Evidence Pack</h2>
<p class="explain">Таблица фрагментов для ручной разметки (сортировка, поиск). Прокрутка — внутри блока ниже.</p>
<div class="evidence-table-scroll"><div id="evidence-wrap"></div></div>
<p class="method-note"><strong>Таблица:</strong> Только очищенные фрагменты (после фильтра шума): без оглавления, колонтитулов и позиционных повторов. Предложение и контекст с разметкой R/O; file_name и source_pointer — для поиска в тексте. Полная выгрузка для верификации и статьи: <code>output/evidence_base.xlsx</code> (листы evidence_clean и excluded_noise с source_pointer и noise_reason). Сырой экспорт: <code>output/piro_fragments.xlsx</code>.</p>
<div class="analytics"><strong>Аналитика.</strong> {analytics_evidence}</div>
<button type="button" class="toggle-analysis" data-target="memo-evidence">Скрыть расширенную интерпретацию</button>
<button type="button" class="recalc-analysis" data-block-id="evidence" data-target="memo-evidence">Пересчитать аналитическую интерпретацию</button>
<div class="analysis-block" id="memo-evidence" style="display:block;"><h3>Расширенная интерпретация</h3><div class="memo-text"></div><h4>Проверяемые гипотезы</h4><ul class="hypotheses"></ul><h4>Методологические риски</h4><ul class="pitfalls"></ul><div class="memo-meta" style="font-size:0.85rem;color:#666;margin-top:0.5rem;"></div></div>
</section>

<section id="sec-embedding" class="report-section">
<h2>Embedding validation</h2>
<div id="embedding-wrap"></div>
<p class="method-note"><strong>Блок:</strong> Метрики кластеризации контекстов упоминаний (число точек, silhouette score и др.) и при наличии — scatter UMAP (цвет = кластер). Рассчитывается при запуске пайплайна с флагом <code>--run-embeddings</code>.</p>
<div class="analytics"><strong>Аналитика.</strong> {analytics_embedding}</div>
<button type="button" class="toggle-analysis" data-target="memo-embedding">Скрыть расширенную интерпретацию</button>
<button type="button" class="recalc-analysis" data-block-id="embedding" data-target="memo-embedding">Пересчитать аналитическую интерпретацию</button>
<div class="analysis-block" id="memo-embedding" style="display:block;"><h3>Расширенная интерпретация</h3><div class="memo-text"></div><h4>Проверяемые гипотезы</h4><ul class="hypotheses"></ul><h4>Методологические риски</h4><ul class="pitfalls"></ul><div class="memo-meta" style="font-size:0.85rem;color:#666;margin-top:0.5rem;"></div></div>
</section>

<section id="sec-scientific-report" class="report-section">
<h2>Научный отчёт (аналитический текст ИИ)</h2>
{scientific_report_html}
</section>

<section id="sec-synthesis" class="report-section">
<h2>Структурная модель ориентализации</h2>
<div id="synthesis-wrap"></div>
</section>

<section id="sec-limits" class="report-section">
<h2>Ограничения и проверки</h2>
<p class="method-note"><strong>Методологические ограничения (для статьи):</strong></p>
<ul class="explain" style="margin:0.5rem 0 1rem 1.5rem;">
<li>Единица наблюдения — одно упоминание этнонима в контексте предложения; агрегация по каноническому этнониму (alias→canonical).</li>
<li>При N&lt;20 индексы OI, ED, EPS, AS статистически неустойчивы; в таблице индексов помечены как «низкая надёжность».</li>
<li>Keyness: пороги частоты и G2, лемматизация и фильтр OCR заданы в пайплайне; интерпретация опирается на таблицы и KWIC-примеры ниже.</li>
<li>Сети interaction: разрежённость может быть следствием узкого словаря глаголов и правил SVO, а не отсутствия взаимодействий в текстах.</li>
<li>R и O классифицируются по лексиконам (и опционально LLM); возможны пропуски и шум.</li>
<li>Корпус и фильтр шума задают границы репрезентативности; верификация по source_pointer и evidence_base.xlsx рекомендуется.</li>
</ul>
<p class="explain">Графики: для экспорта в статью используйте кнопку в углу графика (Plotly) → SVG/PNG; точные числа — в таблицах отчёта.</p>
<button type="button" class="toggle-analysis" data-target="memo-limits">Скрыть расширенную интерпретацию</button>
<button type="button" class="recalc-analysis" data-block-id="limits" data-target="memo-limits">Пересчитать аналитическую интерпретацию</button>
<div class="analysis-block" id="memo-limits" style="display:block;"><h3>Расширенная интерпретация</h3><div class="memo-text"></div><h4>Проверяемые гипотезы</h4><ul class="hypotheses"></ul><h4>Методологические риски</h4><ul class="pitfalls"></ul><div class="memo-meta" style="font-size:0.85rem;color:#666;margin-top:0.5rem;"></div></div>
</section>

<section id="sec-tokens" class="report-section">
<h2>Использование LLM (учёт токенов)</h2>
{token_usage_html}
</section>

<section id="sec-aux" class="report-section">
<h2>Вспомогательные визуализации</h2>
<p class="explain">Графики и таблицы (при наличии) лежат в подпапках <code>output</code>. Открывайте отчёт из папки <code>output</code> (<code>python -m http.server 8000</code>), чтобы ссылки работали.</p>
<div class="aux-links"><a href="figures/">Графики (figures/)</a><a href="tables/">Таблицы keyness (tables/)</a></div>
</section>

<footer class="report-footer">Отчёт сгенерирован: {report_generated}</footer>
</main>
</div>

<div class="evidence-modal-overlay" id="evidence-modal-overlay">
<div class="evidence-modal">
<div class="modal-header">
<h3 style="margin:0;">Доказательная база</h3>
<button type="button" class="modal-close" id="evidence-modal-close">Закрыть</button>
</div>
<div class="modal-body">
<div class="modal-filters">
<label>Этнос: <select id="evidence-filter-ethnos"><option value="">— все —</option></select></label>
<label>R: <select id="evidence-filter-r"><option value="">— все —</option></select></label>
<label>O: <select id="evidence-filter-o"><option value="">— все —</option></select></label>
<label>Шум: <select id="evidence-filter-noise"><option value="">— все —</option><option value="0">Нет</option><option value="1">Да</option></select></label>
</div>
<div id="evidence-modal-table-wrap"></div>
<p class="explain" id="evidence-modal-message">Загрузка…</p>
</div>
</div>
</div>

<script src="{jq}"></script>
<script src="{plotly}"></script>
<script src="{dt_js}"></script>
<script>
var reportData = {data_json};
var config = {{ mode: 'raw', noise: 'excl', topn: 10, keynessTopn: 30 }};
var API_BASE = (reportData.api_base || '').replace(/\\/$/, '');
var LLM_ENABLED = reportData.llm_enabled !== false;

function renderCorpus() {{
  var c = reportData.corpus;
  var html = '<table class="display"><thead><tr><th>Показатель</th><th>Значение</th></tr></thead><tbody>';
  html += '<tr><td>Документов</td><td>' + c.num_docs + '</td></tr>';
  html += '<tr><td>Предложений</td><td>' + c.num_sents + '</td></tr>';
  html += '<tr><td>Упоминаний (raw)</td><td>' + c.n_raw + '</td></tr>';
  html += '<tr><td>После очистки</td><td>' + c.n_clean + '</td></tr>';
  html += '<tr><td>Исключено шум</td><td>' + c.n_noise + '</td></tr></tbody></table>';
  document.getElementById('table-corpus-wrap').innerHTML = html;
  if ($.fn.DataTable) {{
    var tbl = document.getElementById('table-corpus-wrap').querySelector('table');
    if (tbl && !$(tbl).hasClass('dataTable')) $(tbl).DataTable({{ pageLength: 25, order: [[0, 'asc']] }});
  }}
}}

function renderMemos() {{
  var memos = reportData.memos || {{}};
  ['corpus','representations','situations','keyness','essentialization','networks','evidence','embedding','limits'].forEach(function(id) {{
    var block = document.getElementById('memo-' + id);
    if (!block || !block.classList.contains('analysis-block')) return;
    var m = memos[id];
    var textEl = block.querySelector('.memo-text');
    var hypEl = block.querySelector('ul.hypotheses');
    var pitEl = block.querySelector('ul.pitfalls');
    if (textEl) textEl.textContent = m && m.memo_text ? m.memo_text : '';
    if (hypEl) {{
      hypEl.innerHTML = '';
      if (m && m.hypotheses_to_check && m.hypotheses_to_check.length)
        m.hypotheses_to_check.forEach(function(h) {{ var li = document.createElement('li'); li.textContent = h; hypEl.appendChild(li); }});
    }}
    if (pitEl) {{
      pitEl.innerHTML = '';
      if (m && m.pitfalls && m.pitfalls.length)
        m.pitfalls.forEach(function(p) {{ var li = document.createElement('li'); li.textContent = p; pitEl.appendChild(li); }});
    }}
  }});
  document.querySelectorAll('.toggle-analysis').forEach(function(btn) {{
    btn.onclick = function() {{
      var targetId = this.getAttribute('data-target');
      var block = document.getElementById(targetId);
      if (!block) return;
      var visible = block.style.display !== 'none';
      block.style.display = visible ? 'none' : 'block';
      var suffix = (this.textContent.indexOf(' (R)') !== -1) ? ' (R)' : ((this.textContent.indexOf(' (O)') !== -1) ? ' (O)' : '');
      this.textContent = visible ? ('Показать расширенную интерпретацию' + suffix) : ('Скрыть расширенную интерпретацию' + suffix);
    }};
  }});
}}

function getEthnosSeries() {{
  var mode = config.mode, noise = config.noise, n = parseInt(config.topn, 10) || 10;
  var eth = reportData.ethnos;
  var labels = [], values = [];
  var obj = (mode === 'normalized') ? (noise === 'incl' ? eth.norm_incl : eth.norm_excl) : (noise === 'incl' ? eth.raw_incl : eth.raw_excl);
  if (!obj) return {{ labels: [], values: [] }};
  var entries = Object.entries(obj).filter(function(e) {{ return e[0] !== '' && e[0] != null; }}).sort(function(a,b) {{ return (b[1]||0) - (a[1]||0); }}).slice(0, n);
  entries.forEach(function(e) {{ labels.push(e[0]); values.push(Number(e[1]) || 0); }});
  return {{ labels: labels, values: values }};
}}

function getRSeries() {{
  var mode = config.mode, noise = config.noise;
  var r = reportData.R;
  var obj = (mode === 'normalized') ? r.norm : (noise === 'incl' ? r.raw_incl : r.raw_excl);
  if (!obj) return {{ labels: [], values: [] }};
  var entries = Object.entries(obj).filter(function(e) {{ return e[0] !== ''; }}).sort(function(a,b) {{ return (b[1]||0) - (a[1]||0); }});
  return {{ labels: entries.map(function(e) {{ return e[0]; }}), values: entries.map(function(e) {{ return Number(e[1]) || 0; }}) }};
}}

function getOSeries() {{
  var mode = config.mode, noise = config.noise;
  var o = reportData.O;
  var obj = (mode === 'normalized') ? o.norm : (noise === 'incl' ? o.raw_incl : o.raw_excl);
  if (!obj) return {{ labels: [], values: [] }};
  var entries = Object.entries(obj).filter(function(e) {{ return e[0] !== ''; }}).sort(function(a,b) {{ return (b[1]||0) - (a[1]||0); }});
  return {{ labels: entries.map(function(e) {{ return e[0]; }}), values: entries.map(function(e) {{ return Number(e[1]) || 0; }}) }};
}}

function redrawCharts() {{
  var eth = getEthnosSeries();
  if (eth.labels.length) {{
    var trace = {{ y: eth.labels, x: eth.values, type: 'bar', orientation: 'h', marker: {{ color: '#4a6fa5' }} }};
    var layout = {{ title: config.mode === 'normalized' ? 'Этносы (на 10k предл.)' : 'Этносы (сырые частоты)', margin: {{ l: 120 }}, xaxis: {{ title: config.mode === 'normalized' ? 'На 10k предл.' : 'Количество' }} }};
    Plotly.newPlot(document.getElementById('plot-ethnos'), [trace], layout, {{ responsive: true }});
  }}
  var r = getRSeries();
  if (r.labels.length) {{
    var traceR = {{ x: r.labels, y: r.values, type: 'bar', marker: {{ color: '#7d8b9e' }} }};
    var layoutR = {{ title: 'Тип репрезентации (R)', margin: {{ b: 80 }} }};
    Plotly.newPlot(document.getElementById('plot-R'), [traceR], layoutR, {{ responsive: true }});
  }}
  var o = getOSeries();
  if (o.labels.length) {{
    var traceO = {{ x: o.labels, y: o.values, type: 'bar', marker: {{ color: '#8b7355' }} }};
    var layoutO = {{ title: 'Ситуация (O)', margin: {{ b: 80 }} }};
    Plotly.newPlot(document.getElementById('plot-O'), [traceO], layoutO, {{ responsive: true }});
  }}
}}

function renderKeyness() {{
  var n = parseInt(config.keynessTopn, 10) || 30;
  var keyness = reportData.keyness || {{}};
  var names = Object.keys(keyness);
  var html = '';
  names.slice(0, 8).forEach(function(name) {{
    var data = keyness[name];
    var arr = data[n] || data[25] || data[10] || [];
    if (arr.length === 0) return;
    html += '<h4>' + name.replace(/keyness_/g,'').replace(/_/g,' ') + '</h4><table class="display keyness-table"><thead><tr><th>Слово</th><th>G2</th><th>freq_focus</th></tr></thead><tbody>';
    arr.forEach(function(row) {{
      html += '<tr><td>' + (row.word || '') + '</td><td>' + (row.G2 != null ? row.G2 : '') + '</td><td>' + (row.freq_focus != null ? row.freq_focus : '') + '</td></tr>';
    }});
    html += '</tbody></table>';
    var kwicData = reportData.keyness_kwic && reportData.keyness_kwic[name];
    if (kwicData && kwicData.length) {{
      html += '<details class="method-note" style="margin-top:0.5rem;"><summary>KWIC (примеры контекстов, source_pointer для верификации)</summary>';
      kwicData.forEach(function(item) {{
        html += '<p><strong>' + (item.word || '') + '</strong>' + (item.G2 != null ? ' (G2: ' + item.G2 + ')' : '') + '</p><ul style="margin:0.3rem 0 1rem 0;">';
        (item.kwic || []).forEach(function(k) {{
          var left = (k.left || '').replace(/</g,'&lt;').replace(/>/g,'&gt;');
          var right = (k.right || '').replace(/</g,'&lt;').replace(/>/g,'&gt;');
          html += '<li><code>' + left + ' <b>' + (k.node || '') + '</b> ' + right + '</code> <small>' + (k.source_pointer || '') + '</small></li>';
        }});
        html += '</ul>';
      }});
      html += '</details>';
    }}
  }});
  document.getElementById('keyness-tables-wrap').innerHTML = html || '<p>Нет данных keyness.</p>';
  if ($.fn.DataTable) {{
    $('.keyness-table').each(function() {{
      if ($(this).find('tbody tr').length) $(this).DataTable({{ pageLength: 15, order: [[1, 'desc']] }});
    }});
  }}
}}

function renderEssentialization() {{
  var ess = reportData.essentialization || {{}};
  var counts = ess.counts || {{}};
  var examples = ess.examples || {{}};
  var html = '<p class="explain">Частоты эссенциализирующих конструкций по этносам.</p><table class="display"><thead><tr><th>Этнос</th><th>Количество</th></tr></thead><tbody>';
  Object.entries(counts).sort(function(a,b) {{ return (b[1]||0) - (a[1]||0); }}).forEach(function(e) {{
    html += '<tr><td>' + e[0] + '</td><td>' + e[1] + '</td></tr>';
  }});
  html += '</tbody></table>';
  html += '<h4>Примеры (до 5 на этнос, без дубликатов)</h4>';
  Object.entries(examples).forEach(function(e) {{
    var eth = e[0], list = e[1] || [];
    html += '<p><strong>' + eth + '</strong></p><ul>';
    list.slice(0, 5).forEach(function(x) {{ html += '<li>' + (x.sentence_text || '').substring(0, 250) + ' … <code>' + (x.source_pointer || '') + '</code></li>'; }});
    html += '</ul>';
  }});
  document.getElementById('essentialization-wrap').innerHTML = html;
  if ($.fn.DataTable) $('#essentialization-wrap table.display').DataTable({{ pageLength: 20 }});
}}

function renderNetworks() {{
  var edges = reportData.interaction_edges || [];
  var comention = reportData.comention_edges || [];
  var html = '<h4>Interaction (направленная)</h4><table class="display"><thead><tr><th>src</th><th>dst</th><th>type</th><th>count</th><th>Пример 1</th></tr></thead><tbody>';
  edges.forEach(function(e) {{
    html += '<tr><td>' + (e.src||'') + '</td><td>' + (e.dst||'') + '</td><td>' + (e.type||'') + '</td><td>' + (e.count||0) + '</td><td>' + (e.example_1||'').substring(0, 120) + '…</td></tr>';
  }});
  html += '</tbody></table><h4>Co-mention (первые 100 рёбер)</h4><table class="display"><thead><tr><th>Этнос 1</th><th>Этнос 2</th><th>Вес</th></tr></thead><tbody>';
  comention.slice(0, 100).forEach(function(e) {{
    html += '<tr><td>' + (e.a||'') + '</td><td>' + (e.b||'') + '</td><td>' + (e.weight||0) + '</td></tr>';
  }});
  html += '</tbody></table>';
  document.getElementById('networks-wrap').innerHTML = html;
  if ($.fn.DataTable) {{
    $('#networks-wrap table.display').each(function() {{
      $(this).DataTable({{ pageLength: 25 }});
    }});
  }}
}}

function renderEvidence() {{
  var wrap = document.getElementById('evidence-wrap');
  if (!wrap) return;
  var rows = reportData.evidence || [];
  if (rows.length === 0) {{ wrap.innerHTML = '<p>Нет данных. Соберите piro_fragments.xlsx и перезапустите пайплайн.</p>'; return; }}
  var cols = Object.keys(rows[0] || {{}});
  var html = '<table id="evidence-table" class="display"><thead><tr>' + cols.map(function(c) {{ return '<th>' + c + '</th>'; }}).join('') + '</tr></thead><tbody>';
  rows.forEach(function(r) {{
    html += '<tr>' + cols.map(function(c) {{ var v = r[c]; return '<td>' + (v != null ? String(v).substring(0, 300) : '') + '</td>'; }}).join('') + '</tr>';
  }});
  html += '</tbody></table>';
  wrap.innerHTML = html;
  if ($.fn.DataTable) $('#evidence-table').DataTable({{ pageLength: 25, order: [[0, 'asc']] }});
}}

function renderIndices() {{
  var di = reportData.derived_indices || {{}};
  var oi = di.OI || {{}};
  var rawOI = oi.raw_OI || {{}};
  var normOI = oi.normalized_OI || rawOI;
  var ci = oi.confidence_interval || {{}};
  var as_ = di.AS || {{}};
  var ed = di.ED || {{}};
  var eps = di.EPS || {{}};
  var nMentions = di.mentions_per_ethnos || {{}};
  var ethnosSet = new Set([].concat(Object.keys(rawOI), Object.keys(ed), Object.keys(eps), Object.keys(nMentions)));
  var ethnosList = Array.from(ethnosSet).sort();
  if (ethnosList.length === 0) {{
    document.getElementById('indices-table-wrap').innerHTML = '<p>Индексы не рассчитаны. Запустите пайплайн с расчётом derived_indices.</p>';
    return;
  }}
  var html = '<table class="display" id="indices-table"><thead><tr><th>Этнос</th><th>N</th><th>Надёжность</th><th>raw_OI</th><th>norm_OI</th><th>CI low</th><th>CI high</th><th>AS</th><th>ED</th><th>EPS</th></tr></thead><tbody>';
  ethnosList.forEach(function(eth) {{
    var r = rawOI[eth], n = normOI[eth], c = ci[eth] || {{}}, a = as_[eth], e = ed[eth], p = eps[eth], nCnt = nMentions[eth];
    var rel = (nCnt != null && nCnt < 20) ? 'низкая (N<20)' : '—';
    html += '<tr><td>' + eth + '</td><td>' + (nCnt != null ? nCnt : '') + '</td><td>' + rel + '</td><td>' + (r != null ? r : '') + '</td><td>' + (n != null ? n : '') + '</td><td>' + (c.low != null ? c.low : '') + '</td><td>' + (c.high != null ? c.high : '') + '</td><td>' + (a != null ? a : '') + '</td><td>' + (e != null ? e : '') + '</td><td>' + (p != null ? p : '') + '</td></tr>';
  }});
  html += '</tbody></table>';
  document.getElementById('indices-table-wrap').innerHTML = html;
  if ($.fn.DataTable) $('#indices-table').DataTable({{ pageLength: 20, order: [[4, 'desc']] }});
  var oiSorted = ethnosList.map(function(eth) {{ return {{ ethnos: eth, val: rawOI[eth] != null ? rawOI[eth] : 0 }}; }}).sort(function(a,b) {{ return b.val - a.val; }});
  var traceBar = {{ x: oiSorted.map(function(x) {{ return x.ethnos; }}), y: oiSorted.map(function(x) {{ return x.val; }}), type: 'bar', name: 'OI' }};
  Plotly.newPlot(document.getElementById('plot-oi-ranking'), [traceBar], {{ title: 'Ранжирование по Orientalization Index (raw_OI)', xaxis: {{ title: 'Этнос' }}, yaxis: {{ title: 'OI' }}, margin: {{ b: 120 }} }}, {{ responsive: true }});
  var wrapOiAs = document.getElementById('plot-scatter-oi-as');
  var wrapEdEps = document.getElementById('plot-scatter-ed-eps');
  if (reportData.scatter_oi_as_svg && wrapOiAs) {{
    wrapOiAs.innerHTML = '<img src="' + reportData.scatter_oi_as_svg + '" alt="OI vs Agency Score" style="max-width:100%; height:auto;">' +
      '<p class="method-note">Экспорт: <a href="' + (reportData.scatter_oi_as_svg || '') + '">SVG</a>' +
      (reportData.scatter_oi_as_png ? ' · <a href="' + reportData.scatter_oi_as_png + '">PNG</a>' : '') + '</p>';
  }} else {{
    var oiAsX = [], oiAsY = [], oiAsLabels = [];
    ethnosList.forEach(function(eth) {{
      if (rawOI[eth] != null && as_[eth] != null) {{ oiAsX.push(rawOI[eth]); oiAsY.push(as_[eth]); oiAsLabels.push(eth); }}
    }});
    if (oiAsX.length && wrapOiAs) {{
      var traceScatter1 = {{ x: oiAsX, y: oiAsY, mode: 'markers+text', type: 'scatter', text: oiAsLabels, textposition: 'top center', marker: {{ size: 10 }} }};
      Plotly.newPlot(wrapOiAs, [traceScatter1], {{ title: 'OI vs Agency Score', margin: {{ l: 70, r: 70, t: 50, b: 50 }}, xaxis: {{ title: 'OI', automargin: true }}, yaxis: {{ title: 'AS', automargin: true }} }}, {{ responsive: true }});
    }}
  }}
  if (reportData.scatter_ed_eps_svg && wrapEdEps) {{
    wrapEdEps.innerHTML = '<img src="' + reportData.scatter_ed_eps_svg + '" alt="ED vs EPS" style="max-width:100%; height:auto;">' +
      '<p class="method-note">Экспорт: <a href="' + (reportData.scatter_ed_eps_svg || '') + '">SVG</a>' +
      (reportData.scatter_ed_eps_png ? ' · <a href="' + reportData.scatter_ed_eps_png + '">PNG</a>' : '') + '</p>';
  }} else {{
    var edEpsX = [], edEpsY = [], edEpsLabels = [];
    ethnosList.forEach(function(eth) {{
      if (ed[eth] != null && eps[eth] != null) {{ edEpsX.push(ed[eth]); edEpsY.push(eps[eth]); edEpsLabels.push(eth); }}
    }});
    if (edEpsX.length && wrapEdEps) {{
      var traceScatter2 = {{ x: edEpsX, y: edEpsY, mode: 'markers+text', type: 'scatter', text: edEpsLabels, textposition: 'top center', marker: {{ size: 10 }} }};
      Plotly.newPlot(wrapEdEps, [traceScatter2], {{ title: 'ED vs EPS', margin: {{ l: 70, r: 70, t: 50, b: 50 }}, xaxis: {{ title: 'ED', automargin: true }}, yaxis: {{ title: 'EPS', automargin: true }} }}, {{ responsive: true }});
    }}
  }}
}}

function renderSynthesis() {{
  var syn = reportData.synthesis || {{}};
  var wrap = document.getElementById('synthesis-wrap');
  if (!wrap) return;
  if (!syn || !Object.keys(syn).length) {{
    wrap.innerHTML = '<p>Синтез не сформирован. Запустите пайплайн с <code>--run-llm</code> или с derived analytics.</p>';
    return;
  }}
  var html = '';
  if (syn.observation || syn.theoretical_framing || syn.diagnostics) {{
    html += '<h3>Синтетическая интерпретация (LLM)</h3>';
    if (syn.observation) html += '<h4>Наблюдение данных</h4><p>' + syn.observation + '</p>';
    if (syn.theoretical_framing) html += '<h4>Теоретическое соотнесение</h4><p>' + syn.theoretical_framing + '</p>';
    if (syn.diagnostics) html += '<h4>Диагностика</h4><p>' + syn.diagnostics + '</p>';
    if (syn.hypotheses && syn.hypotheses.length) {{
      html += '<h4>Гипотезы (3)</h4><ul>';
      syn.hypotheses.forEach(function(h) {{ html += '<li>' + h + '</li>'; }});
      html += '</ul>';
    }}
    if (syn.risks && syn.risks.length) {{
      html += '<h4>Риски</h4><ul>';
      syn.risks.forEach(function(r) {{ html += '<li>' + r + '</li>'; }});
      html += '</ul>';
    }}
    if (syn.llm_disabled_note) html += '<p class="explain" style="color:#856404;">' + syn.llm_disabled_note + '</p>';
  }} else {{
    if (syn.structural_model) html += '<h4>Структурная модель</h4><p>' + syn.structural_model + '</p>';
    if (syn.consistency_note) html += '<h4>Согласованность индексов</h4><p>' + syn.consistency_note + '</p>';
    if (syn.synthesis_text) html += '<h4>Синтетическая интерпретация</h4><p>' + syn.synthesis_text + '</p>';
    if (syn.research_directions && syn.research_directions.length) {{
      html += '<h4>Исследовательские направления (5)</h4><ul>';
      syn.research_directions.forEach(function(d) {{ html += '<li>' + d + '</li>'; }});
      html += '</ul>';
    }}
    if (syn.hypotheses && syn.hypotheses.length) {{
      html += '<h4>Комплексные гипотезы (3)</h4><ul>';
      syn.hypotheses.forEach(function(h) {{ html += '<li>' + h + '</li>'; }});
      html += '</ul>';
    }}
    if (syn.limitations && syn.limitations.length) {{
      html += '<h4>Методологические ограничения</h4><ul>';
      syn.limitations.forEach(function(l) {{ html += '<li>' + l + '</li>'; }});
      html += '</ul>';
    }}
  }}
  if (syn.llm_payload_summary) {{
    var pl = syn.llm_payload_summary;
    html += '<details class="explain" style="margin-top:1rem;"><summary>Что передано в LLM</summary><ul style="margin:0.5rem 0;">';
    html += '<li>n_mentions: ' + (pl.n_mentions != null ? pl.n_mentions : '—') + ', n_ethnos: ' + (pl.n_ethnos != null ? pl.n_ethnos : '—') + '</li>';
    html += '<li>Поля: ' + (pl.fields_used && pl.fields_used.length ? pl.fields_used.join(', ') : '—') + '</li>';
    html += '<li>noise_share: ' + (pl.noise_share != null ? pl.noise_share : '—') + '</li>';
    html += '</ul></details>';
  }}
  wrap.innerHTML = html || '<p>Нет содержимого синтеза.</p>';
}}

function renderEthnicProfile() {{
  var profiles = reportData.derived_profiles || [];
  var tests = reportData.derived_stats_tests || {{}};
  var correlations = reportData.derived_correlations || [];
  var clusters = reportData.derived_clusters || [];
  var tableWrap = document.getElementById('ethnic-profiles-table-wrap');
  var statsWrap = document.getElementById('derived-stats-tests-wrap');
  var clustersWrap = document.getElementById('derived-clusters-wrap');
  if (tableWrap) {{
    if (!profiles.length) {{ tableWrap.innerHTML = '<p>Нет данных профилей. Запустите пайплайн без <code>--no-derived</code>.</p>'; }}
    else {{
      var cols = Object.keys(profiles[0]);
      var thead = cols.map(function(c) {{ return '<th>' + c + '</th>'; }}).join('');
      var rows = profiles.map(function(r) {{ return '<tr>' + cols.map(function(c) {{ return '<td>' + (r[c] != null && r[c] !== '' ? r[c] : '') + '</td>'; }}).join('') + '</tr>'; }}).join('');
      tableWrap.innerHTML = '<table class="display"><thead><tr>' + thead + '</tr></thead><tbody>' + rows + '</tbody></table>';
      if (typeof $ !== 'undefined' && $.fn.DataTable) {{
        var tbl = tableWrap.querySelector('table');
        if (tbl && !$(tbl).hasClass('dataTable')) $(tbl).DataTable({{ pageLength: 15, order: [[1, 'desc']] }});
      }}
    }}
  }}
  if (profiles.length) {{
    var oiSort = profiles.slice().sort(function(a,b) {{ return (b.OI || 0) - (a.OI || 0); }}).slice(0, 15);
    var edSort = profiles.slice().sort(function(a,b) {{ return (b.ED || 0) - (a.ED || 0); }}).slice(0, 15);
    var oiY = oiSort.map(function(p) {{ return p.ethnos || ''; }});
    var oiX = oiSort.map(function(p) {{ return Number(p.OI) || 0; }});
    var edY = edSort.map(function(p) {{ return p.ethnos || ''; }});
    var edX = edSort.map(function(p) {{ return Number(p.ED) || 0; }});
    if (document.getElementById('plot-profile-oi-ranking') && oiX.length) {{
      var tO = {{ y: oiY, x: oiX, type: 'bar', orientation: 'h', marker: {{ color: '#4a6fa5' }} }};
      Plotly.newPlot('plot-profile-oi-ranking', [tO], {{ title: 'OI (топ-15)', margin: {{ l: 120 }}, xaxis: {{ title: 'OI' }} }}, {{ responsive: true }});
    }}
    if (document.getElementById('plot-profile-ed-ranking') && edX.length) {{
      var tE = {{ y: edY, x: edX, type: 'bar', orientation: 'h', marker: {{ color: '#6b8e23' }} }};
      Plotly.newPlot('plot-profile-ed-ranking', [tE], {{ title: 'ED (топ-15)', margin: {{ l: 120 }}, xaxis: {{ title: 'ED' }} }}, {{ responsive: true }});
    }}
    var oiV = profiles.map(function(p) {{ return Number(p.OI); }}).filter(function(x) {{ return !isNaN(x); }});
    var edV = profiles.map(function(p) {{ return Number(p.ED); }}).filter(function(x) {{ return !isNaN(x); }});
    var asV = profiles.map(function(p) {{ return Number(p.AS); }}).filter(function(x) {{ return !isNaN(x); }});
    var mnV = profiles.map(function(p) {{ return Number(p.mentions_norm); }}).filter(function(x) {{ return !isNaN(x); }});
    var lbl = profiles.map(function(p) {{ return p.ethnos || ''; }});
    if (document.getElementById('plot-profile-oi-ed') && oiV.length && edV.length) {{
      var s1 = {{ x: oiV, y: edV, mode: 'markers+text', type: 'scatter', text: lbl, textposition: 'top center', marker: {{ size: 10 }} }};
      Plotly.newPlot('plot-profile-oi-ed', [s1], {{ title: 'OI vs ED', margin: {{ l: 70, r: 70, t: 50, b: 50 }}, xaxis: {{ title: 'OI', automargin: true }}, yaxis: {{ title: 'ED', automargin: true }} }}, {{ responsive: true }});
    }}
    if (document.getElementById('plot-profile-oi-as') && oiV.length && asV.length) {{
      var s2 = {{ x: oiV, y: asV, mode: 'markers+text', type: 'scatter', text: lbl, textposition: 'top center', marker: {{ size: 10 }} }};
      Plotly.newPlot('plot-profile-oi-as', [s2], {{ title: 'OI vs AS', margin: {{ l: 70, r: 70, t: 50, b: 50 }}, xaxis: {{ title: 'OI', automargin: true }}, yaxis: {{ title: 'AS', automargin: true }} }}, {{ responsive: true }});
    }}
    if (document.getElementById('plot-profile-oi-mentions') && oiV.length && mnV.length) {{
      var s3 = {{ x: oiV, y: mnV, mode: 'markers+text', type: 'scatter', text: lbl, textposition: 'top center', marker: {{ size: 10 }} }};
      Plotly.newPlot('plot-profile-oi-mentions', [s3], {{ title: 'OI vs mentions_norm', margin: {{ l: 70, r: 70, t: 50, b: 50 }}, xaxis: {{ title: 'OI', automargin: true }}, yaxis: {{ title: 'mentions_norm', automargin: true }} }}, {{ responsive: true }});
    }}
  }}
  if (statsWrap) {{
    var stHtml = '';
    if (tests.chi2_R) stHtml += '<p><strong>Chi-square Ethnos×R:</strong> χ²=' + tests.chi2_R.chi2 + ', p=' + tests.chi2_R.p + ', Cramér\\'s V=' + tests.chi2_R.cramers_v + '</p>';
    if (tests.chi2_O) stHtml += '<p><strong>Chi-square Ethnos×O:</strong> χ²=' + tests.chi2_O.chi2 + ', p=' + tests.chi2_O.p + ', Cramér\\'s V=' + tests.chi2_O.cramers_v + '</p>';
    if (tests.bootstrap_CI && Object.keys(tests.bootstrap_CI).length) stHtml += '<p>Bootstrap 95% CI по топ-этносам (OI, EPS, ED) — см. derived_indices.json / stats_tests.json.</p>';
    statsWrap.innerHTML = stHtml || '<p>Нет данных тестов.</p>';
  }}
  if (clustersWrap) {{
    if (!clusters.length) clustersWrap.innerHTML = '<p>Нет данных кластеров.</p>';
    else {{
      var cTable = '<table class="display"><thead><tr><th>ethnos</th><th>cluster_id</th></tr></thead><tbody>';
      clusters.forEach(function(r) {{ cTable += '<tr><td>' + (r.ethnos || '') + '</td><td>' + (r.cluster_id != null ? r.cluster_id : '') + '</td></tr>'; }});
      cTable += '</tbody></table>';
      clustersWrap.innerHTML = cTable;
      if (typeof $ !== 'undefined' && $.fn.DataTable) {{
        var ct = clustersWrap.querySelector('table');
        if (ct && !$(ct).hasClass('dataTable')) $(ct).DataTable({{ pageLength: 20 }});
      }}
    }}
  }}
}}

function renderEmbedding() {{
  var wrap = document.getElementById('embedding-wrap');
  if (!wrap) return;
  var cv = reportData.cluster_validation || {{}};
  var umap = reportData.cluster_umap;
  if (!cv || !Object.keys(cv).length) {{ wrap.innerHTML = '<p>Запустите с флагом <code>--run-embeddings</code> для расчёта кластеризации.</p>'; return; }}
  var html = '<table class="display"><thead><tr><th>Метрика</th><th>Значение</th></tr></thead><tbody>';
  Object.entries(cv).forEach(function(e) {{ html += '<tr><td>' + e[0] + '</td><td>' + e[1] + '</td></tr>'; }});
  html += '</tbody></table>';
  if (umap && umap.umap_xy && umap.labels && umap.umap_xy.length) {{
    html += '<div class="plot-container" id="plot-umap"></div>';
  }}
  wrap.innerHTML = html;
  if (umap && umap.umap_xy && umap.labels && umap.umap_xy.length && document.getElementById('plot-umap')) {{
    var xs = umap.umap_xy.map(function(p) {{ return p[0]; }});
    var ys = umap.umap_xy.map(function(p) {{ return p[1]; }});
    var trace = {{ x: xs, y: ys, mode: 'markers', type: 'scatter', marker: {{ size: 4, color: umap.labels, colorscale: 'Viridis' }} }};
    Plotly.newPlot('plot-umap', [trace], {{ title: 'UMAP (цвет = кластер)', xaxis: {{ title: 'UMAP1' }}, yaxis: {{ title: 'UMAP2' }} }}, {{ responsive: true }});
  }}
}}

function buildSummaryForBlock(blockId) {{
  var d = reportData;
  var summary = {{ block_id: blockId, corpus: d.corpus, config: config }};
  if (blockId === 'representations' || blockId === 'corpus') summary.ethnos = d.ethnos;
  if (blockId === 'representations' || blockId === 'situations') {{ summary.R = d.R; summary.O = d.O; }}
  if (blockId === 'keyness') {{ summary.keyness = d.keyness; summary.keynessTopn = config.keynessTopn; }}
  if (blockId === 'networks') {{ summary.interaction_edges = d.interaction_edges; summary.comention_edges = d.comention_edges; }}
  if (blockId === 'essentialization') summary.essentialization = d.essentialization;
  if (blockId === 'embedding') {{ summary.cluster_validation = d.cluster_validation; summary.cluster_umap = d.cluster_umap; }}
  if (blockId === 'evidence') summary.evidence = d.evidence;
  if (blockId === 'limits') summary.corpus = d.corpus;
  return JSON.stringify(summary);
}}

function recalcAnalysis(blockId, memoBlock) {{
  if (!API_BASE) {{ alert('Backend не настроен. Запустите: python -m api.app'); return; }}
  var summaryJson = buildSummaryForBlock(blockId);
  var btn = event && event.target;
  if (btn) btn.disabled = true;
  var payload = {{ block_id: blockId, summary_json: JSON.parse(summaryJson) }};
  fetch(API_BASE + '/api/analyze', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify(payload)
  }}).then(function(r) {{ return r.json(); }}).then(function(data) {{
    if (memoBlock) {{
      var t = memoBlock.querySelector('.memo-text');
      if (t) t.innerHTML = data.memo_text != null ? data.memo_text : '';
      var h = memoBlock.querySelector('.hypotheses');
      if (h) {{ var arr = data.hypotheses_to_check || []; h.innerHTML = arr.map(function(x) {{ return '<li>' + x + '</li>'; }}).join(''); }}
      var p = memoBlock.querySelector('.pitfalls');
      if (p) {{ var arr2 = data.pitfalls || []; p.innerHTML = arr2.map(function(x) {{ return '<li>' + x + '</li>'; }}).join(''); }}
      var m = memoBlock.querySelector('.memo-meta');
      if (m && data.meta) {{ m.innerHTML = 'Дата: ' + (data.meta.date || '') + ' | prompt: ' + (data.meta.system_prompt_version || '') + ' | модель: ' + (data.meta.model || '') + ' | записей: ' + (data.meta.n_records || '') + ' | noise_share: ' + (data.meta.noise_share || '') + (data.meta.cached ? ' | (кэш)' : ''); }}
    }}
  }}).catch(function(err) {{ alert('Ошибка запроса: ' + err); }}).finally(function() {{ if (btn) btn.disabled = false; }});
}}

function init() {{
  var mainEl = document.querySelector('main');
  if (!LLM_ENABLED && mainEl) {{
    var banner = document.createElement('p');
    banner.className = 'llm-disabled-banner';
    banner.style.cssText = 'background:#f8d7da;color:#721c24;padding:0.5rem 1rem;border-radius:4px;margin:0.5rem 0;';
    banner.textContent = 'LLM-анализ отключен. Используется rule-based summary. Для реактивной аналитики запустите с --run-llm и backend (python -m api.app).';
    mainEl.insertBefore(banner, mainEl.firstChild);
  }}
  try {{ renderCorpus(); }} catch (e) {{ console.warn('renderCorpus', e); }}
  try {{ renderEthnicProfile(); }} catch (e) {{ console.warn('renderEthnicProfile', e); }}
  try {{ renderIndices(); }} catch (e) {{ console.warn('renderIndices', e); }}
  try {{ renderSynthesis(); }} catch (e) {{ console.warn('renderSynthesis', e); }}
  try {{ renderMemos(); }} catch (e) {{ console.warn('renderMemos', e); }}
  try {{ redrawCharts(); }} catch (e) {{ console.warn('redrawCharts', e); }}
  try {{ renderKeyness(); }} catch (e) {{ console.warn('renderKeyness', e); }}
  try {{ renderEssentialization(); }} catch (e) {{ console.warn('renderEssentialization', e); }}
  try {{ renderNetworks(); }} catch (e) {{ console.warn('renderNetworks', e); }}
  try {{ renderEvidence(); }} catch (e) {{ console.warn('renderEvidence', e); }}
  try {{ renderEmbedding(); }} catch (e) {{ console.warn('renderEmbedding', e); }}

  document.querySelectorAll('.recalc-analysis').forEach(function(btn) {{
    var blockId = btn.getAttribute('data-block-id');
    var targetId = btn.getAttribute('data-target');
    if (!blockId || !targetId) return;
    btn.addEventListener('click', function() {{
      var memoBlock = document.getElementById(targetId);
      recalcAnalysis(blockId, memoBlock);
    }});
  }});

  var ctrlMode = document.getElementById('ctrl-mode');
  if (ctrlMode) ctrlMode.addEventListener('change', function() {{ config.mode = this.value; redrawCharts(); }});
  var ctrlNoise = document.getElementById('ctrl-noise');
  if (ctrlNoise) ctrlNoise.addEventListener('change', function() {{ config.noise = this.value; redrawCharts(); }});
  var ctrlTopn = document.getElementById('ctrl-topn');
  if (ctrlTopn) ctrlTopn.addEventListener('change', function() {{ config.topn = this.value; redrawCharts(); }});
  var ctrlKeyness = document.getElementById('ctrl-keyness-topn');
  if (ctrlKeyness) ctrlKeyness.addEventListener('change', function() {{ config.keynessTopn = this.value; renderKeyness(); }});

  var evidenceData = null;
  var evidenceTableApi = null;
  function openEvidenceModal() {{
    var overlay = document.getElementById('evidence-modal-overlay');
    var msg = document.getElementById('evidence-modal-message');
    var wrap = document.getElementById('evidence-modal-table-wrap');
    if (!overlay) return;
    if (evidenceData) {{
      overlay.classList.add('show');
      return;
    }}
    msg.textContent = 'Загрузка derived/evidence_pack_sample.json…';
    wrap.innerHTML = '';
    fetch('derived/evidence_pack_sample.json').then(function(r) {{ return r.ok ? r.json() : Promise.reject(new Error(r.status)); }})
      .then(function(data) {{
        evidenceData = Array.isArray(data) ? data : [];
        msg.textContent = evidenceData.length ? '' : 'Нет данных. Запустите: python tools/build_evidence_layer.py';
        if (evidenceData.length === 0) {{ overlay.classList.add('show'); return; }}
        wrap.innerHTML = '<table id="evidence-modal-table" class="display"><thead><tr><th>ethnos_norm</th><th>R_label</th><th>O_label</th><th>sentence_text</th><th>source_pointer</th><th>is_noise</th></tr></thead><tbody></tbody></table>';
        var ethnosSet = {{}}, rSet = {{}}, oSet = {{}};
        evidenceData.forEach(function(r) {{
          if (r.ethnos_norm) ethnosSet[r.ethnos_norm] = true;
          if (r.R_label) rSet[r.R_label] = true;
          if (r.O_label) oSet[r.O_label] = true;
        }});
        var selEthnos = document.getElementById('evidence-filter-ethnos');
        var selR = document.getElementById('evidence-filter-r');
        var selO = document.getElementById('evidence-filter-o');
        if (selEthnos) {{ Object.keys(ethnosSet).sort().forEach(function(e) {{ var o = document.createElement('option'); o.value = e; o.textContent = e; selEthnos.appendChild(o); }}); }}
        if (selR) {{ Object.keys(rSet).sort().forEach(function(e) {{ var o = document.createElement('option'); o.value = e; o.textContent = e; selR.appendChild(o); }}); }}
        if (selO) {{ Object.keys(oSet).sort().forEach(function(e) {{ var o = document.createElement('option'); o.value = e; o.textContent = e; selO.appendChild(o); }}); }}
        overlay.classList.add('show');
        if (typeof $ !== 'undefined' && $.fn.DataTable) {{
          if (evidenceTableApi) evidenceTableApi.destroy();
          evidenceTableApi = $('#evidence-modal-table').DataTable({{ data: evidenceData, columns: [{{ data: 'ethnos_norm' }}, {{ data: 'R_label' }}, {{ data: 'O_label' }}, {{ data: 'sentence_text' }}, {{ data: 'source_pointer' }}, {{ data: 'is_noise', render: function(v) {{ return v === true ? 'Да' : (v === false ? 'Нет' : (v != null ? String(v) : '')); }} }}], pageLength: 25, order: [[0, 'asc']] }});
        }}
      }})
      .catch(function() {{ msg.textContent = 'Файл derived/evidence_pack_sample.json не найден. Запустите из папки output: python -m http.server 8000, затем откройте report.html. Сгенерируйте выборку: python tools/build_evidence_layer.py'; }});
  }}
  function closeEvidenceModal() {{
    var overlay = document.getElementById('evidence-modal-overlay');
    if (overlay) overlay.classList.remove('show');
  }}
  document.getElementById('btn-open-evidence') && document.getElementById('btn-open-evidence').addEventListener('click', openEvidenceModal);
  document.getElementById('evidence-modal-close') && document.getElementById('evidence-modal-close').addEventListener('click', closeEvidenceModal);
  document.getElementById('evidence-modal-overlay') && document.getElementById('evidence-modal-overlay').addEventListener('click', function(e) {{ if (e.target === this) closeEvidenceModal(); }});
  var filterEthnos = document.getElementById('evidence-filter-ethnos');
  var filterR = document.getElementById('evidence-filter-r');
  var filterO = document.getElementById('evidence-filter-o');
  var filterNoise = document.getElementById('evidence-filter-noise');
  function applyEvidenceFilters() {{
    if (!evidenceTableApi) return;
    var e = filterEthnos && filterEthnos.value;
    var r = filterR && filterR.value;
    var o = filterO && filterO.value;
    var n = filterNoise && filterNoise.value;
    $.fn.dataTable.ext.search = [];
    $.fn.dataTable.ext.search.push(function(settings, row, data) {{
      if (e && data.ethnos_norm !== e) return false;
      if (r && data.R_label !== r) return false;
      if (o && data.O_label !== o) return false;
      if (n !== '' && n !== undefined) {{ var noise = data.is_noise ? '1' : '0'; if (noise !== n) return false; }}
      return true;
    }});
    evidenceTableApi.draw();
  }}
  if (filterEthnos) filterEthnos.addEventListener('change', function() {{ if (evidenceTableApi) applyEvidenceFilters(); }});
  if (filterR) filterR.addEventListener('change', function() {{ if (evidenceTableApi) applyEvidenceFilters(); }});
  if (filterO) filterO.addEventListener('change', function() {{ if (evidenceTableApi) applyEvidenceFilters(); }});
  if (filterNoise) filterNoise.addEventListener('change', function() {{ if (evidenceTableApi) applyEvidenceFilters(); }});
}}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
else init();
</script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return str(output_path)
