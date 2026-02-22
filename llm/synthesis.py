"""
Межтабличный синтез: передача индексов OI, AS, ED, EPS и ключевых метрик в LLM.
Результат: структурная модель ориентализации, 5 исследовательских направлений, 3 комплексные гипотезы.
Сохраняется в output/llm_memos/synthesis.json.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from .deepseek_client import call_deepseek
from .knowledge_loader import get_system_prompt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEMOS_DIR = PROJECT_ROOT / "output" / "llm_memos"
SYNTHESIS_PATH = MEMOS_DIR / "synthesis.json"

SYNTHESIS_USER_PROMPT = """По данным количественных индексов ориентализации сформируй межтабличный синтез в рамках ориентализма, травелогов и новой имперской истории.

Требования:
1) Выяви структурную модель ориентализации (как связаны индексы OI, AS, ED, EPS между собой и с keyness/network).
2) Определи согласованность индексов; интерпретируй в свете репрезентации «других» и имперского взгляда в травелогах.
3) Сформулируй 5 исследовательских направлений (research_directions: массив строк).
4) Сформулируй 3 комплексные гипотезы (hypotheses: массив строк).
5) Укажи методологические ограничения (limitations: массив строк).
6) Краткая синтетическая интерпретация (synthesis_text: 1–2 абзаца) с теоретическим фреймингом.

Ответ — только JSON с ключами: structural_model (строка), consistency_note (строка), research_directions (массив), hypotheses (массив), limitations (массив), synthesis_text (строка)."""


def _parse_response(text: str) -> Dict[str, Any]:
    if not text or "LLM disabled" in text:
        return {}
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
    return {}


def generate_synthesis(
    oi: Dict[str, Any],
    as_scores: Dict[str, float],
    ed: Dict[str, float],
    eps: Dict[str, float],
    keyness_top_terms: Optional[Dict[str, List[str]]] = None,
    network_density: Optional[float] = None,
    run_llm: bool = True,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Формирует summary из индексов, вызывает DeepSeek, сохраняет output/llm_memos/synthesis.json.
    Возвращает полный ответ (structural_model, research_directions, hypotheses, limitations, synthesis_text).
    """
    output_path = output_path or SYNTHESIS_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keyness_top_terms = keyness_top_terms or {}
    summary = {
        "OI": oi,
        "AS": as_scores,
        "ED": ed,
        "EPS": eps,
        "keyness_top_terms": {k: (v[:20] if isinstance(v, list) else v) for k, v in list(keyness_top_terms.items())[:10]},
        "network_density": network_density,
    }
    data_json = json.dumps(summary, ensure_ascii=False, indent=2)
    system_prompt = get_system_prompt(use_knowledge=True)

    if run_llm:
        response = call_deepseek(system_prompt, SYNTHESIS_USER_PROMPT, data_json, call_id="synthesis")
        parsed = _parse_response(response)
        if parsed:
            out = {
                "structural_model": parsed.get("structural_model", ""),
                "consistency_note": parsed.get("consistency_note", ""),
                "research_directions": parsed.get("research_directions", []),
                "hypotheses": parsed.get("hypotheses", []),
                "limitations": parsed.get("limitations", []),
                "synthesis_text": parsed.get("synthesis_text", ""),
                "source": "llm",
            }
        else:
            out = _rule_synthesis()
    else:
        out = _rule_synthesis()
    out["inputs_summary"] = {"OI_ethnos": list((oi.get("raw_OI") or oi).keys())[:15], "AS_nodes": list(as_scores.keys())[:10]}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def _rule_synthesis() -> Dict[str, Any]:
    return {
        "structural_model": "Структурная модель строится по индексам OI, AS, ED, EPS. OI отражает долю негативной и экзотизирующей репрезентации; AS — дисбаланс активности в сети взаимодействий; ED — плотность эссенциализирующих конструкций; EPS — полярность оценок.",
        "consistency_note": "Согласованность между индексами требует ручной проверки на данном корпусе. Запустите с --run-llm для синтетической интерпретации.",
        "research_directions": [
            "Сопоставление OI с типологией источников (автор, издательство, период).",
            "Анализ связи AS с сюжетами взаимодействия (торговля, конфликт, подчинение).",
            "Корреляция ED с частотой упоминаний этноса.",
            "Сравнение EPS по подкорпусам (negative vs neutral keyness).",
            "Валидация кластеров эмбеддингов относительно индексов.",
        ],
        "hypotheses": [
            "Этносы с высоким OI чаще представлены в контекстах с высоким ED.",
            "Узлы с положительным AS (акторы) отличаются по keyness от узлов с отрицательным AS.",
            "Согласованность OI и EPS по этносам указывает на устойчивый оценочный паттерн.",
        ],
        "limitations": [
            "Индексы зависят от лексиконов и разметки R/O.",
            "Network density и AS чувствительны к объёму корпуса и словарю глаголов.",
            "Межтабличный синтез требует качественной верификации на выборке.",
        ],
        "synthesis_text": "Синтетический анализ отключён (LLM не вызывался). Включите --run-llm для генерации межтабличного синтеза по индексам.",
        "source": "rule_based",
    }


def load_synthesis(output_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Загружает сохранённый synthesis.json."""
    path = output_path or SYNTHESIS_PATH
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
