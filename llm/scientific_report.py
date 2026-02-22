"""
Генерация плотного научного отчёта ИИ по данным пайплайна.
Каждой таблице/блоку соответствует абзац или более связного ориенталистского анализа.
Результат: output/llm_memos/scientific_report.json и output/scientific_report.html.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .deepseek_client import call_deepseek
from .knowledge_loader import get_system_prompt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEMOS_DIR = PROJECT_ROOT / "output" / "llm_memos"
SCIENTIFIC_REPORT_JSON = MEMOS_DIR / "scientific_report.json"
SCIENTIFIC_REPORT_HTML = PROJECT_ROOT / "output" / "scientific_report.html"

SCIENTIFIC_REPORT_USER_PROMPT = """По приложенным агрегированным данным пайплайна напиши плотный научный отчёт на русском языке в стиле специалиста по травелогам, русско-европейским связям и постколониальным/имперским исследованиям (ориентализм, репрезентация «других», Сибирь в дискурсе).

Требования:
1. Строго по данным: каждое утверждение опирается на приведённые числа (частоты, индексы, топы). Не выдумывай факты о корпусе; используй переданную выше теоретическую рамку для интерпретации.
2. Одна или несколько связных абзацев на каждый блок: корпус, распределения (этносы, R, O), keyness, эссенциализация, сети, количественные индексы (OI, ED, EPS, AS), профили этносов и статистические проверки, заключение. Связывай цифры с ориенталистскими тропами, экзотизацией, эссенциализацией и имперским взглядом в травелогах.
3. Стиль: плотный, аналитический, с отсылкой к конкретным цифрам. В заключении — обобщение паттернов, соответствие/расхождение с теоретической рамкой, ограничения метода.
4. В данных может быть блок prior_interpretations (memos и synthesis): опирайся на них, развивай и обобщай, не дублируй дословно.

Формат ответа — только JSON без markdown-обёртки, с ключами (каждый значение — строка, один или несколько абзацев через \\n\\n):
- corpus
- distributions
- keyness
- essentialization
- networks
- indices
- ethnic_profiles
- conclusion"""


def _to_native(obj: Any) -> Any:
    """Сериализуемые в JSON значения (numpy → native)."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, bool, float)):
        return obj
    try:
        import numpy as np
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(x) for x in obj]
    return obj


def _parse_llm_response(text: str) -> Optional[Dict[str, str]]:
    """Извлекает JSON из ответа LLM."""
    if not text or "LLM disabled" in text:
        return None
    raw = text.strip()
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                try:
                    return json.loads(part)
                except json.JSONDecodeError:
                    continue
    if raw.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return None


def build_scientific_report_payload(
    num_docs: int,
    num_sents: int,
    n_raw: int,
    n_clean: int,
    ethnos_raw_excl: Dict[str, int],
    R_raw_excl: Dict[str, int],
    O_raw_excl: Dict[str, int],
    keyness_top: Dict[str, List[str]],
    essentialization_table: Dict[str, int],
    interaction_edges: List[Dict],
    derived_indices: Optional[Dict] = None,
    ethnic_profiles: Optional[List[Dict]] = None,
    stats_tests: Optional[Dict] = None,
    correlations: Optional[List[Dict]] = None,
    report_memos: Optional[Dict[str, Dict]] = None,
    synthesis: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Собирает payload для генерации научного отчёта: числа, агрегаты и готовые интерпретации из отчёта (memos, synthesis)."""
    payload = {
        "corpus": {"num_docs": num_docs, "num_sents": num_sents, "n_raw": n_raw, "n_clean": n_clean, "n_noise": n_raw - n_clean},
        "ethnos": dict(sorted(ethnos_raw_excl.items(), key=lambda x: -x[1])[:20]) if ethnos_raw_excl else {},
        "R": R_raw_excl or {},
        "O": O_raw_excl or {},
        "keyness_top": {k: (v[:25] if isinstance(v, list) else v) for k, v in list((keyness_top or {}).items())[:12]},
        "essentialization": dict(sorted((essentialization_table or {}).items(), key=lambda x: -x[1])[:15]),
        "networks": {
            "n_interaction_edges": len(interaction_edges or []),
            "top_edges": [{"src": e.get("src"), "dst": e.get("dst"), "count": e.get("count")} for e in (interaction_edges or [])[:15]],
        },
    }
    if derived_indices:
        oi = derived_indices.get("OI") or {}
        payload["indices"] = {
            "OI": oi.get("raw_OI") or oi,
            "AS": derived_indices.get("AS"),
            "ED": derived_indices.get("ED"),
            "EPS": derived_indices.get("EPS"),
            "mentions_per_ethnos": derived_indices.get("mentions_per_ethnos"),
        }
    else:
        payload["indices"] = {}
    if ethnic_profiles:
        payload["ethnic_profiles"] = ethnic_profiles[:25]
    else:
        payload["ethnic_profiles"] = []
    if stats_tests:
        payload["stats_tests"] = {"chi2_R": stats_tests.get("chi2_R"), "chi2_O": stats_tests.get("chi2_O")}
    else:
        payload["stats_tests"] = {}
    if correlations:
        payload["correlations"] = correlations[:20]
    else:
        payload["correlations"] = []

    prior: Dict[str, Any] = {}
    if report_memos:
        prior["memos"] = {}
        for block_id, m in report_memos.items():
            if not isinstance(m, dict):
                continue
            prior["memos"][block_id] = {
                "memo_text": (m.get("memo_text") or "").strip(),
                "hypotheses": m.get("hypotheses_to_check") or m.get("hypotheses") or [],
                "pitfalls": m.get("pitfalls") or [],
            }
    if synthesis and isinstance(synthesis, dict):
        prior["synthesis"] = {
            "synthesis_text": (synthesis.get("synthesis_text") or "").strip(),
            "structural_model": (synthesis.get("structural_model") or "").strip() if synthesis.get("structural_model") else None,
        }
        prior["synthesis"] = {k: v for k, v in prior["synthesis"].items() if v}
    if prior:
        payload["prior_interpretations"] = prior

    return _to_native(payload)


def generate_scientific_report(
    payload: Dict[str, Any],
    run_llm: bool = True,
    output_json_path: Optional[Path] = None,
    output_html_path: Optional[Path] = None,
    run_passport: Optional[Dict[str, Any]] = None,
    evidence_sample: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, str]:
    """
    Генерирует плотный научный отчёт по данным. Сохраняет scientific_report.json и scientific_report.html.
    run_passport и evidence_sample добавляют в HTML паспорт прогона и примеры source_pointer для верификации.
    """
    output_json_path = output_json_path or SCIENTIFIC_REPORT_JSON
    output_html_path = output_html_path or SCIENTIFIC_REPORT_HTML
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    section_keys = ["corpus", "distributions", "keyness", "essentialization", "networks", "indices", "ethnic_profiles", "conclusion"]

    if run_llm:
        system_prompt = get_system_prompt(use_knowledge=True)
        data_json = json.dumps(payload, ensure_ascii=False, indent=2)
        if "</script>" in data_json:
            data_json = data_json.replace("</script>", "<\\/script>")
        response = call_deepseek(system_prompt, SCIENTIFIC_REPORT_USER_PROMPT, data_json, call_id="scientific_report")
        parsed = _parse_llm_response(response)
        if parsed and any(parsed.get(k) for k in section_keys):
            sections = {k: (parsed.get(k) or "").strip() for k in section_keys}
            sections["_source"] = "llm"
        else:
            sections = _rule_based_scientific_sections()
    else:
        sections = _rule_based_scientific_sections()

    out = {"sections": sections, "payload_summary": {"n_docs": payload.get("corpus", {}).get("num_docs"), "n_mentions": payload.get("corpus", {}).get("n_clean")}}
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    html = _build_scientific_report_html(sections, run_passport=run_passport, evidence_sample=evidence_sample)
    output_html_path.write_text(html, encoding="utf-8")
    return sections


def _rule_based_scientific_sections() -> Dict[str, str]:
    """Заглушки секций при отключённом LLM."""
    return {
        "corpus": "Корпус травелогов о Сибири после предобработки и фильтрации шума используется как база для подсчёта упоминаний этнонимов и индексов. Объём и состав документов задают ограничения репрезентативности.",
        "distributions": "Распределения по этносам и по типам репрезентации (R) и ситуации (O) нормированы на 10k предложений для сопоставимости. Интерпретация требует учёта объёма подвыборок.",
        "keyness": "Keyness (G2) выделяет лексику, характерную для подкорпусов по R или по этносам. Топ-слова — кандидаты на маркеры ориентализирующего дискурса; необходима качественная проверка контекстов.",
        "essentialization": "Частоты эссенциализирующих конструкций по этносам отражают степень натурализации групп как «сущностей». Сопоставление с OI и EPS даёт комплексную картину.",
        "networks": "Графы co-mention и interaction показывают, кто с кем сополагается в дискурсе и кто выступает агентом/объектом. Индекс AS суммирует асимметрию агентности.",
        "indices": "Индексы OI, ED, EPS и AS рассчитаны по очищенной выборке. OI — доля негативной и экзотизирующей репрезентации; ED — доля эссенциализации; EPS — перекос негатив/позитив; AS — дисбаланс ролей в сетях взаимодействия.",
        "ethnic_profiles": "Профили этносов объединяют упоминания, нормализацию, индексы и дельты относительно взвешенного среднего. Статистические проверки (chi-square, Cramér's V) и корреляции дополняют описательную картину.",
        "conclusion": "Выводы по корпусу носят предварительный характер. Рекомендуется запуск с --run-llm для генерации полного аналитического отчёта на основе тех же данных.",
        "_source": "rule_based",
    }


def _build_scientific_report_html(
    sections: Dict[str, str],
    run_passport: Optional[Dict[str, Any]] = None,
    evidence_sample: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Собирает HTML научного отчёта из секций. Добавляет паспорт прогона и примеры source_pointer для верификации."""
    titles = {
        "corpus": "1. Корпус и данные",
        "distributions": "2. Распределения: этносы, репрезентация (R), ситуация (O)",
        "keyness": "3. Keyness и маркеры дискурса",
        "essentialization": "4. Эссенциализация",
        "networks": "5. Сети взаимодействий и co-mention",
        "indices": "6. Количественные индексы (OI, ED, EPS, AS)",
        "ethnic_profiles": "7. Профили этносов и статистические проверки",
        "conclusion": "8. Заключение",
    }
    proof_links = {
        "corpus": ("report.html#sec-corpus", "таблица корпуса"),
        "distributions": ("report.html#sec-distributions", "графики этносов, R, O"),
        "keyness": ("report.html#sec-keyness", "таблицы keyness и KWIC-примеры"),
        "essentialization": ("report.html#sec-essentialization", "таблица эссенциализации"),
        "networks": ("report.html#sec-networks", "таблицы рёбер и ограничения метода"),
        "indices": ("report.html#sec-indices", "таблица индексов (N, CI, надёжность)"),
        "ethnic_profiles": ("report.html#sec-ethnic-profile", "профили, χ², Cramér's V"),
        "conclusion": ("report.html#sec-evidence", "примеры предложений и evidence_base.xlsx"),
    }
    parts = []

    if run_passport:
        ts = run_passport.get("run_ts", "")
        if ts and "T" in str(ts):
            try:
                from datetime import datetime as dt
                t = dt.fromisoformat(str(ts).replace("Z", "+00:00"))
                ts = t.strftime("%d.%m.%Y %H:%M")
            except Exception:
                pass
        docs_str = ", ".join(run_passport.get("input_documents") or []) or "—"
        nf = run_passport.get("noise_filter") or {}
        filter_str = ", ".join(f"{k}={v}" for k, v in nf.items()) or "—"
        parts.append(
            "<section><h2>Паспорт прогона</h2>\n"
            f"<p>Дата/время: {ts}. Входные документы: {docs_str}. "
            f"Документов: {run_passport.get('num_docs', 0)}, предложений: {run_passport.get('num_sents', 0)}, "
            f"упоминаний raw: {run_passport.get('n_raw', 0)}, после очистки: {run_passport.get('n_clean', 0)}, "
            f"исключено шум: {run_passport.get('n_noise', 0)} ({run_passport.get('noise_pct', 0)}%). "
            f"Параметры фильтра: {filter_str}. Все цифры в отчёте соответствуют этому прогону.</p>\n</section>"
        )

    for key in ["corpus", "distributions", "keyness", "essentialization", "networks", "indices", "ethnic_profiles", "conclusion"]:
        if key.startswith("_"):
            continue
        text = (sections.get(key) or "").strip()
        if not text:
            continue
        title = titles.get(key, key)
        paras = [f"<p>{p.strip()}</p>" for p in text.split("\n\n") if p.strip()]
        proof = proof_links.get(key)
        if proof:
            url, desc = proof
            paras.append(f'<p class="proof-ref"><strong>Проверка:</strong> <a href="{url}">{desc}</a>; примеры предложений — в блоке «Доказательная база» ниже и в <a href="report.html#sec-evidence">report.html, Evidence Pack</a>.</p>')
        parts.append(f"<section><h2>{title}</h2>\n" + "\n".join(paras) + "\n</section>")

    # Методологические ограничения (явный раздел для статьи)
    limitations_html = """
    <section><h2>9. Методологические ограничения</h2>
    <p>При интерпретации результатов следует учитывать:</p>
    <ul>
    <li><strong>Единица наблюдения:</strong> одно упоминание этнонима в контексте предложения; агрегация по каноническому этнониму (alias→canonical).</li>
    <li><strong>Индексы при малом N:</strong> при N&lt;20 упоминаний индексы OI, ED, EPS, AS статистически неустойчивы; в отчёте такие случаи помечены как «низкая надёжность».</li>
    <li><strong>Keyness:</strong> пороги частоты и G2, лемматизация и фильтр OCR-мусора заданы в пайплайне; интерпретация опирается на таблицы keyness и KWIC-примеры в report.html.</li>
    <li><strong>Сети взаимодействий:</strong> разрежённость графа может отражать узкий словарь глаголов (resources/interaction_verbs.yml) и жёсткие правила извлечения SVO, а не фактическое отсутствие взаимодействий в текстах.</li>
    <li><strong>Репрезентация (R) и ситуация (O):</strong> классификация по лексиконам и при необходимости по LLM; возможны пропуски и шум.</li>
    <li><strong>Корпус:</strong> состав и объём документов задают границы репрезентативности; фильтр шума (header/footer, позиционная дедупликация) уменьшает артефакты, но может задеть краевые контексты.</li>
    </ul>
    <p>Рекомендуется верификация ключевых выводов по <code>source_pointer</code> в исходных текстах и выгрузке <code>output/evidence_base.xlsx</code>.</p>
    </section>
    """
    parts.append(limitations_html)

    if evidence_sample:
        def _h(s: str) -> str:
            if not s:
                return ""
            return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        rows = []
        for i, r in enumerate(evidence_sample[:10], 1):
            ptr = r.get("source_pointer") or (r.get("file_name", "") + "#" + str(r.get("sent_idx", "")))
            P = _h(r.get("P") or r.get("ethnos_norm", ""))
            R = _h(r.get("R", ""))
            sent = (r.get("sentence_text") or "")[:120]
            if len(r.get("sentence_text") or "") > 120:
                sent += "…"
            rows.append(f"<tr><td>{i}</td><td><code>{_h(ptr)}</code></td><td>{P}</td><td>{R}</td><td>{_h(sent)}</td></tr>")
        table = (
            "<table border=\"1\" cellpadding=\"6\" style=\"border-collapse:collapse; font-size:0.9em;\">"
            "<thead><tr><th>№</th><th>source_pointer</th><th>Этнос</th><th>R</th><th>Фрагмент</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>"
        )
        parts.append(
            "<section><h2>Доказательная база (примеры для верификации)</h2>\n"
            "<p>Ниже — 5–10 примеров с <code>source_pointer</code> для проверки выводов по тексту. "
            "Полная таблица и фильтры — в <a href=\"report.html#sec-evidence\">report.html, секция Evidence Pack</a>. "
            "Выгрузка: <code>output/evidence_base.xlsx</code>.</p>\n" + table + "\n</section>"
        )

    body = "\n".join(parts)
    source_note = " (сгенерировано ИИ по данным пайплайна)" if sections.get("_source") == "llm" else " (шаблонный текст, LLM не вызывался)"
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Научный отчёт: ориенталистский анализ травелогов</title>
<style>
body {{ font-family: Georgia, serif; margin: 0 auto; padding: 2rem; max-width: 800px; line-height: 1.6; color: #222; background: #fafafa; }}
h1 {{ color: #1a5276; border-bottom: 1px solid #1a5276; padding-bottom: 0.3rem; }}
h2 {{ color: #2c3e50; margin-top: 2rem; font-size: 1.15rem; }}
p {{ margin: 0.75rem 0; text-align: justify; }}
.note {{ font-size: 0.9rem; color: #666; margin-bottom: 2rem; }}
.proof-ref {{ font-size: 0.9rem; color: #1a5276; margin-top: 0.75rem; }}
</style>
</head>
<body>
<h1>Научный отчёт: ориенталистский анализ травелогов о Сибири</h1>
<p class="note">Плотный аналитический текст по данным пайплайна{source_note}. Таблицы, графики и доказательства — <a href="report.html">report.html</a>; паспорт прогона и примеры source_pointer приведены ниже.</p>
{body}
</body>
</html>
"""


def load_scientific_report(output_json_path: Optional[Path] = None) -> Optional[Dict[str, str]]:
    """Загружает сохранённые секции научного отчёта."""
    path = output_json_path or SCIENTIFIC_REPORT_JSON
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("sections")
    except Exception:
        return None
