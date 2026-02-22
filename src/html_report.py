"""
Главный артефакт отчётности: интерактивный самодостаточный HTML (output/report.html).
Оглавление слева, таблицы с DataTables (сортировка, поиск), раскрывающиеся цитаты.
PDF генерируется только по флагу --report-pdf как краткая выжимка.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict

import pandas as pd


def _h(s: str) -> str:
    """Экранирование для HTML."""
    if s is None or not isinstance(s, str):
        return ""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _table_to_html(
    df: pd.DataFrame,
    table_id: str,
    explain: str,
    expandable_col: Optional[str] = None,
) -> str:
    """Таблица с thead/tbody и опциональной колонкой с раскрытием."""
    if df is None or df.empty:
        return f'<p class="explain">{_h(explain)}</p><p>Нет данных.</p>'
    html = [f'<p class="explain">{_h(explain)}</p>']
    html.append(f'<table id="{table_id}" class="display compact" style="width:100%"><thead><tr>')
    for c in df.columns:
        html.append(f"<th>{_h(str(c))}</th>")
    html.append("</tr></thead><tbody>")
    for _, row in df.iterrows():
        html.append("<tr>")
        for c in df.columns:
            val = row[c]
            if expandable_col and c == expandable_col and val:
                val = _h(str(val)[:200])
                html.append(f'<td class="expandable" data-full="{_h(str(row[c])[:2000])}">{val}… <a href="#" class="toggle">развернуть</a></td>')
            else:
                html.append(f"<td>{_h(str(val))}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")
    return "\n".join(html)


def _analytical_checks(
    piro_records: List[Dict],
    keyness_negative: Optional[pd.DataFrame],
    interaction_edges: List[Dict],
) -> Dict[str, Any]:
    """Корреляция R и O, эссенциализация по O, асимметрия interaction, negative vs military."""
    out = {}
    if not piro_records:
        return out
    # R × O contingency
    r_o = defaultdict(lambda: defaultdict(int))
    for r in piro_records:
        ro = r.get("R") or "neutral"
        oo = r.get("O_situation") or "unknown"
        r_o[ro][oo] += 1
    out["R_O_contingency"] = {k: dict(v) for k, v in r_o.items()}
    # Essentialization by O
    ess_o = defaultdict(int)
    for r in piro_records:
        if r.get("is_essentializing"):
            oo = r.get("O_situation") or "unknown"
            ess_o[oo] += 1
    out["essentialization_by_O"] = dict(ess_o)
    # Asymmetry: in-degree vs out-degree per node
    in_d = defaultdict(int)
    out_d = defaultdict(int)
    for e in interaction_edges or []:
        s, d = e.get("src"), e.get("dst")
        if s:
            out_d[s] += e.get("count", 0)
        if d:
            in_d[d] += e.get("count", 0)
    out["interaction_in_out"] = {
        "in_degree": dict(in_d),
        "out_degree": dict(out_d),
    }
    # Negative keywords vs military: доля контекстов negative с маркерами military
    military_words = {"fight", "war", "attack", "army", "battle", "conflict", "raid", "troops"}
    neg_ctx = [r.get("context_text") or "" for r in piro_records if r.get("R") == "negative"]
    neg_with_mil = sum(1 for c in neg_ctx if military_words & set((c or "").lower().split()))
    out["negative_vs_military"] = {
        "negative_contexts": len(neg_ctx),
        "with_military_markers": neg_with_mil,
        "share": round(neg_with_mil / len(neg_ctx), 4) if neg_ctx else 0,
    }
    return out


def build_html_report(
    corpus: List[Dict],
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    piro_clean: List[Dict],
    keyness_tables: Dict[str, pd.DataFrame],
    norm_ethnos: Dict[str, Dict],
    norm_R: Dict[str, Dict],
    norm_O: Dict[str, Dict],
    essentialization_table: Dict[str, int],
    interaction_edges: List[Dict],
    comention_raw: Dict,
    comention_jaccard: Dict,
    evidence_df: Optional[pd.DataFrame],
    cluster_validation: Optional[Dict],
    llm_memos: Optional[Dict[str, str]],
    output_path: Optional[Path] = None,
) -> str:
    """
    Собирает output/report.html: самодостаточный HTML с оглавлением и интерактивными таблицами.
    """
    output_path = output_path or Path(__file__).resolve().parent.parent / "output" / "report.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    analytical = _analytical_checks(piro_clean, keyness_tables.get("keyness_negative_vs_neutral"), interaction_edges)
    llm_memos = llm_memos or {}

    num_docs = len(corpus)
    num_sents = sum(len(d.get("sentences", [])) for d in corpus)
    n_raw = len(raw_df)
    n_clean = len(clean_df)
    n_noise = n_raw - n_clean

    sections = []

    # 1) Корпус и очистка
    df_corpus = pd.DataFrame([
        {"Показатель": "Документов", "Значение": num_docs},
        {"Показатель": "Предложений", "Значение": num_sents},
        {"Показатель": "Упоминаний (raw)", "Значение": n_raw},
        {"Показатель": "После очистки шума", "Значение": n_clean},
        {"Показатель": "Исключено как шум", "Значение": n_noise},
    ])
    sections.append((
        "corpus",
        "1. Корпус и очистка",
        _table_to_html(df_corpus, "table-corpus", "Объём корпуса и результат фильтра колонтитулов и позиционной дедупликации."),
        llm_memos.get("corpus", ""),
    ))

    # 2) Репрезентации
    r_counts = Counter(r.get("R") for r in piro_clean)
    df_r = pd.DataFrame([{"Тип R": k, "Количество": v} for k, v in r_counts.most_common()])
    if norm_R:
        rows = [{"Тип R": k, "count": v.get("count", 0), "mean_rate_per_10k": v.get("mean_rate_per_10k")} for k, v in norm_R.items()]
        df_r_norm = pd.DataFrame(rows)
    else:
        df_r_norm = df_r
    sections.append((
        "representations",
        "2. Репрезентации (counts и нормированные)",
        _table_to_html(df_r_norm, "table-r", "Распределение типов репрезентации (R). Нормировка: упоминаний на 10k предложений по документам.") + "\n" + _table_to_html(df_r, "table-r-counts", "Сырые количества."),
        llm_memos.get("representations", ""),
    ))

    # 3) Ситуации
    o_counts = Counter(r.get("O_situation") for r in piro_clean)
    df_o = pd.DataFrame([{"O_situation": k, "Количество": v} for k, v in o_counts.most_common()])
    sections.append((
        "situations",
        "3. Ситуации (включая unknown/mixed)",
        _table_to_html(df_o, "table-o", "Распределение типов ситуации (O). unknown и mixed — случаи низкой уверенности классификатора."),
        llm_memos.get("situations", ""),
    ))

    # 4) Эссенциализация
    df_ess = pd.DataFrame([{"Этнос": k, "Конструкций": v} for k, v in sorted(essentialization_table.items(), key=lambda x: -x[1])])
    examples_ess = [r for r in piro_clean if r.get("is_essentializing")][:10]
    rows_ess_ex = [{"Пример": (r.get("sentence_text") or r.get("essentialization_span") or "")[:250]} for r in examples_ess]
    df_ess_ex = pd.DataFrame(rows_ess_ex) if rows_ess_ex else pd.DataFrame()
    sections.append((
        "essentialization",
        "4. Эссенциализация (частоты и примеры)",
        _table_to_html(df_ess, "table-ess", "Количество эссенциализирующих конструкций по этносам.") + "\n" + _table_to_html(df_ess_ex, "table-ess-ex", "Примеры предложений с эссенциализацией."),
        llm_memos.get("essentialization", ""),
    ))

    # 5) Keyness
    keyness_html = []
    for name, df in list(keyness_tables.items())[:8]:
        if df is None or df.empty:
            continue
        title = name.replace("keyness_", "").replace("_", " ")
        keyness_html.append(f"<h4>{_h(title)}</h4>")
        keyness_html.append(_table_to_html(df.head(50), f"table-keyness-{name}", f"Ключевые слова (G2) для сравнения: {title}. Стоп-слова и служебные токены исключены; при наличии POS — фильтр content words (NOUN/ADJ/VERB)."))
    sections.append((
        "keyness",
        "5. Keyness (главный блок)",
        "\n".join(keyness_html) if keyness_html else "<p>Нет данных keyness.</p>",
        llm_memos.get("keyness", ""),
    ))

    # 6) Сети
    edges_df = pd.DataFrame()
    if interaction_edges:
        rows = []
        for r in interaction_edges:
            ex = r.get("examples", [])
            rows.append({
                "src": r.get("src"),
                "dst": r.get("dst"),
                "type": r.get("type"),
                "count": r.get("count"),
                "example_1": ex[0][:150] + "…" if len(ex) > 0 else "",
                "example_2": ex[1][:150] + "…" if len(ex) > 1 else "",
                "example_3": ex[2][:150] + "…" if len(ex) > 2 else "",
            })
        edges_df = pd.DataFrame(rows)
    comention_raw_df = pd.DataFrame()
    if comention_raw:
        rows = [{"ethnos_1": a, "ethnos_2": b, "weight": w} for a, targets in comention_raw.items() for b, w in targets.items() if a != b and w > 0]
        if rows:
            comention_raw_df = pd.DataFrame(rows)
    sections.append((
        "networks",
        "6. Сети (co-mention и interaction)",
        _table_to_html(edges_df, "table-interaction", "Рёбра направленной сети взаимодействий (subject → verb → object). Три примера предложений на ребро.", expandable_col="example_1")
        + "\n"
        + _table_to_html(comention_raw_df.head(100), "table-comention", "Co-mention: совместные упоминания в контексте ±2 предложения (без самопетель). Вес — число общих контекстов."),
        llm_memos.get("networks", ""),
    ))

    # 7) Evidence Pack
    if evidence_df is not None and not evidence_df.empty:
        ev_df = evidence_df.head(200).copy()
        if "context_text" in ev_df.columns:
            ev_df["context_short"] = ev_df["context_text"].astype(str).str[:150] + "…"
        sections.append((
            "evidence",
            "7. Evidence Pack (фильтрируемый)",
            _table_to_html(ev_df, "table-evidence", "Примеры для ручной валидации: ethnos, R, O, confidence, контекст. Сортировка и поиск в таблице.", expandable_col="context_text" if "context_text" in ev_df.columns else None),
            llm_memos.get("evidence", ""),
        ))
    else:
        sections.append(("evidence", "7. Evidence Pack", "<p>Запустите с флагом --build-evidence-pack для формирования evidence pack.</p>", ""))

    # 8) Embedding validation
    if cluster_validation:
        df_cv = pd.DataFrame([{"Метрика": k, "Значение": v} for k, v in cluster_validation.items()])
        sections.append((
            "embedding",
            "8. Embedding validation",
            _table_to_html(df_cv, "table-cv", "Метрики кластеризации (HDBSCAN/KMeans + UMAP): silhouette, purity по R/O, доля шума."),
            llm_memos.get("embedding", ""),
        ))
    else:
        sections.append(("embedding", "8. Embedding validation", "<p>Запустите с --run-embeddings для расчёта кластеризации и метрик.</p>", ""))

    # 9) Ограничения и проверки + Analytical checks
    df_analytical = pd.DataFrame()
    if analytical:
        rows = []
        for k, v in analytical.get("negative_vs_military", {}).items():
            rows.append({"Проверка": "negative_vs_military_" + k, "Значение": v})
        for k, v in analytical.get("essentialization_by_O", {}).items():
            rows.append({"Проверка": f"essentialization_O_{k}", "Значение": v})
        if rows:
            df_analytical = pd.DataFrame(rows)
    sections.append((
        "limits",
        "9. Ограничения и аналитические проверки",
        _table_to_html(df_analytical, "table-analytical", "Автоматические проверки: корреляция negative с military-маркерами, эссенциализация по типам ситуации. Рекомендуется ручная выборочная верификация.")
        + "<p>Ограничения: классификация R/O основана на лексиконах; keyness чувствителен к объёму подкорпусов; сети отражают только явные маркеры в тексте.</p>",
        llm_memos.get("limits", ""),
    ))

    # Build full HTML
    toc_items = "\n".join(
        f'<li><a href="#sec-{sid}">{_h(title)}</a></li>' for sid, title, _, _ in sections
    )
    body_sections = []
    for i, (sid, title, content, memo) in enumerate(sections):
        memo_html = ""
        if memo:
            memo_html = f'<div class="ai-memo" id="memo-{sid}"><strong>Комментарий ИИ:</strong> {_h(memo)}</div>'
        body_sections.append(
            f'<section id="sec-{sid}" class="report-section">'
            f'<h2>{_h(title)}</h2>'
            f'{content}'
            f'{memo_html}'
            f'</section>'
        )

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Отчёт: цифровой анализ травелогов о Сибири</title>
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
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
.explain {{ color: #555; margin-bottom: 0.8rem; font-size: 0.95rem; }}
table {{ border-collapse: collapse; margin: 1rem 0; font-size: 0.9rem; }}
th, td {{ border: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: left; }}
th {{ background: #34495e; color: #fff; }}
.ai-memo {{ background: #e8f6f3; padding: 0.8rem; margin-top: 1rem; border-radius: 4px; }}
.toggle {{ font-size: 0.85rem; color: #2980b9; }}
</style>
</head>
<body>
<div class="layout">
<nav class="toc">
<h3>Оглавление</h3>
<ul>
{toc_items}
</ul>
</nav>
<main class="main">
<h1>Исследовательский отчёт: травелоги о Сибири</h1>
<p>Интерактивные таблицы: сортировка по клику на заголовок, поиск в поле фильтра. Раскрывающиеся цитаты — по ссылке «развернуть».</p>
{"".join(body_sections)}
<section id="appendix" class="report-section">
<h2>Приложение: Wordcloud</h2>
<p>Визуализация wordcloud оставлена только для справки; основной аналитический аргумент — keyness с POS-фильтрами (раздел 5).</p>
</section>
</main>
</div>
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
<script>
$(function() {{
  $('table.display').each(function() {{
    $(this).DataTable({{ pageLength: 25, order: [[0, 'asc']] }});
  }});
  $('.toggle').on('click', function(e) {{
    e.preventDefault();
    var td = $(this).closest('td');
    if (td.hasClass('expanded')) {{
      td.html(td.data('short') + '… <a href="#" class="toggle">развернуть</a>');
      td.removeClass('expanded');
    }} else {{
      td.data('short', td.text().replace(/… \\s*развернуть.*/, '').trim());
      td.html(td.data('full') + ' <a href="#" class="toggle">свернуть</a>');
      td.addClass('expanded');
    }}
  }});
}});
</script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return str(output_path)


def fetch_llm_memos_for_sections(
    sections_payloads: Dict[str, Dict],
    output_dir: Path,
) -> Dict[str, str]:
    """По каждому разделу отправляет summary в DeepSeek и сохраняет memo. Возвращает {section_id: memo_text}."""
    from .deepseek_analysis import call_deepseek
    memos = {}
    out_memos = output_dir / "llm_memos"
    out_memos.mkdir(parents=True, exist_ok=True)
    for sid, payload in sections_payloads.items():
        try:
            result = call_deepseek(payload)
            text = (result.get("observations") or [])[:3]
            text = " ".join(str(t) for t in text) if text else ""
            memos[sid] = text
            (out_memos / f"{sid}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            memos[sid] = ""
    return memos
