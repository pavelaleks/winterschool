"""
Движок мемо: генерация аналитического комментария по типу блока (representation, situation, keyness, network, essentialization, embeddings).
Конвейерно: summary_dict → LLM или rule-based → сохранение в output/llm_memos/{table_type}.json, возврат memo_text.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from .deepseek_client import call_deepseek
from .knowledge_loader import get_system_prompt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEMOS_DIR = PROJECT_ROOT / "output" / "llm_memos"
MAX_SAMPLE_EXAMPLES = 5

USER_PROMPTS = {
    "representation": (
        "Ниже — агрегированные данные по типу репрезентации (R): распределение по категориям (negative, exotic, positive, neutral, uncertain), нормированные частоты на 10k предложений, доля шума. "
        "Дай структурированную интерпретацию в свете ориентализма и репрезентации «других» в травелогах: memo_text, hypotheses_to_check, pitfalls. Ответ — только JSON."
    ),
    "situation": (
        "Ниже — агрегированные данные по типу ситуации (O): распределение по категориям, нормированные частоты, доля unknown/mixed. "
        "Дай структурированную интерпретацию в контексте контекстов упоминания этносов в травелогах (повседневность, власть, торговля и т.д.): memo_text, hypotheses_to_check, pitfalls. Ответ — только JSON."
    ),
    "keyness": (
        "Ниже — данные keyness: топ-слова по подкорпусам (negative vs neutral, exotic vs neutral и т.д.), частоты, G2. "
        "Интерпретируй в свете ориенталистских тропов и экзотизации: какие слова выступают маркерами негативизации/экзотизации/позитивной оценки, проверяемые гипотезы и риски. Ответ — только JSON: memo_text, hypotheses_to_check, pitfalls."
    ),
    "network": (
        "Ниже — данные по сетям: co-mention (совместные упоминания) и interaction (направленные рёбра), число рёбер, примеры. "
        "Интерпретируй с учётом имперского взгляда и асимметрии агентности (кто субъект/объект действия): memo_text, hypotheses_to_check, pitfalls. Ответ — только JSON."
    ),
    "essentialization": (
        "Ниже — данные по эссенциализации: частоты по этносам, примеры конструкций (не более 5). "
        "Интерпретируй в рамках концепции эссенциализации в ориенталистском дискурсе (сведение группы к «сущности»): memo_text, hypotheses_to_check, pitfalls. Ответ — только JSON."
    ),
    "embeddings": (
        "Ниже — метрики кластеризации контекстов (UMAP/HDBSCAN или KMeans): silhouette, purity по R/O, доля шума. "
        "Дай структурированную интерпретацию: что говорит валидация для различения типов репрезентации, гипотезы и риски. Ответ — только JSON: memo_text, hypotheses_to_check, pitfalls."
    ),
}


def _rule_based_memo(table_type: str, summary_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Rule-based мемо без вызова LLM."""
    memo_text = "Аналитический комментарий сформирован без вызова LLM. Рекомендуется ручная проверка цифр и выборки."
    hypotheses: List[str] = [
        "Проверить репрезентативность выборки по документам.",
        "Выборочно верифицировать разметку по контекстам.",
    ]
    pitfalls: List[str] = ["Классификация основана на лексиконах; возможны пропуски контекстов."]

    if table_type == "representation":
        memo_text = "Распределение R получено по лексиконам и эпитетам. Нормировка на 10k предложений позволяет сравнивать документы разного объёма."
        pitfalls = ["Лексиконы не покрывают все маркеры негатива/экзотизации; доля uncertain отражает низкую уверенность классификатора."]
    elif table_type == "situation":
        memo_text = "Распределение O (ситуация) по доменам из словаря. unknown/mixed — случаи низкой уверенности."
        pitfalls = ["Домены ситуации заданы лексиконами; возможны смешанные контексты."]
    elif table_type == "keyness":
        memo_text = "Keyness (G2) выделяет слова, характерные для подкорпуса. Топ-30 — кандидаты на маркеры; требуется качественный разбор контекстов."
        hypotheses = ["Проверить конкорданс по топ-10 keyness-слов вручную.", "Сравнить эпитетный слой (ADJ) с контентными словами."]
        pitfalls = ["Keyness чувствителен к объёму подкорпусов; стоп-слова и OCR-фильтр уже применены."]
    elif table_type == "network":
        memo_text = "Co-mention: совместное упоминание в контексте. Interaction: субъект–глагол–объект по словарю глаголов. Индексоподобные предложения исключены."
        pitfalls = ["Сеть interaction чувствительна к качеству разбора зависимостей (spaCy); возможны пропуски связей."]
    elif table_type == "essentialization":
        memo_text = "Частоты эссенциализирующих конструкций по этносам. Примеры дедуплицированы. Рекомендуется выборочная проверка по source_pointer."
        pitfalls = ["Паттерны (the X are, X by nature) могут давать ложные срабатывания на не-этнонимы."]
    elif table_type == "embeddings":
        memo_text = "Кластеризация контекстов — валидация расхождения негатив/экзотика по семантическим зонам. Интерпретация кластеров требует ручного просмотра примеров."
        pitfalls = ["Параметры UMAP/HDBSCAN влияют на число кластеров; размер выборки ограничен."]

    return {
        "memo_text": memo_text,
        "hypotheses_to_check": hypotheses,
        "pitfalls": pitfalls,
        "source": "rule_based",
    }


def _ensure_summary_structure(summary_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Ограничивает sample_examples не более чем MAX_SAMPLE_EXAMPLES элементами."""
    out = dict(summary_dict)
    if "sample_examples" in out and isinstance(out["sample_examples"], list):
        out["sample_examples"] = out["sample_examples"][:MAX_SAMPLE_EXAMPLES]
    return out


def _parse_llm_response(text: str) -> Dict[str, Any]:
    """Извлекает JSON из ответа LLM (убирает markdown, код-блоки)."""
    if not text or text.strip() == "LLM disabled":
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


def generate_memo(
    table_type: str,
    summary_dict: Dict[str, Any],
    run_llm: bool = True,
    output_dir: Optional[Path] = None,
) -> str:
    """
    Генерирует мемо для блока отчёта.
    table_type: representation | situation | keyness | network | essentialization | embeddings.
    summary_dict: структурированный JSON (counts, normalized, top_values, confidence, noise_share, sample_examples до 5).
    Если run_llm=True и API ключ есть — вызов DeepSeek, иначе — rule-based.
    Результат сохраняется в output/llm_memos/{table_type}.json.
    Возвращает memo_text (строка).
    """
    output_dir = output_dir or MEMOS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dict = _ensure_summary_structure(summary_dict)
    user_prompt = USER_PROMPTS.get(table_type, "Дай структурированную интерпретацию по переданным данным. Ответ — только JSON: memo_text, hypotheses_to_check, pitfalls.")
    system_prompt = get_system_prompt(use_knowledge=True)
    data_json = json.dumps(summary_dict, ensure_ascii=False, indent=2)

    if run_llm:
        response = call_deepseek(system_prompt, user_prompt, data_json, call_id=table_type)
        parsed = _parse_llm_response(response)
        if parsed:
            memo_text = parsed.get("memo_text", "")
            hypotheses = parsed.get("hypotheses_to_check", [])
            pitfalls = parsed.get("pitfalls", [])
        else:
            memo_text = response if response and response != "LLM disabled" else _rule_based_memo(table_type, summary_dict)["memo_text"]
            hypotheses = _rule_based_memo(table_type, summary_dict)["hypotheses_to_check"]
            pitfalls = _rule_based_memo(table_type, summary_dict)["pitfalls"]
            parsed = {"memo_text": memo_text, "hypotheses_to_check": hypotheses, "pitfalls": pitfalls}
        out = {
            "table_type": table_type,
            "memo_text": parsed.get("memo_text", memo_text),
            "hypotheses_to_check": parsed.get("hypotheses_to_check", hypotheses),
            "pitfalls": parsed.get("pitfalls", pitfalls),
            "source": "llm",
        }
    else:
        out = _rule_based_memo(table_type, summary_dict)
        out["table_type"] = table_type

    out_path = output_dir / f"{table_type}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out.get("memo_text", "")


def load_memo(table_type: str, output_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Загружает сохранённый мемо из output/llm_memos/{table_type}.json. Не вызывает API."""
    output_dir = output_dir or MEMOS_DIR
    path = output_dir / f"{table_type}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_fallback_memo(block_id: str, summary_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Возвращает rule-based мемо для блока, если нет сохранённого LLM-мемо. Используется, чтобы рубрики отчёта не были пустыми."""
    summary_dict = summary_dict or {}
    if block_id in ("corpus", "evidence", "limits"):
        return get_rule_only_memo(block_id, summary_dict)
    table_map = {
        "representations": "representation",
        "situations": "situation",
        "keyness": "keyness",
        "essentialization": "essentialization",
        "networks": "network",
        "embedding": "embeddings",
    }
    table_type = table_map.get(block_id)
    if table_type:
        return _rule_based_memo(table_type, summary_dict)
    return {"table_type": block_id, "memo_text": "", "hypotheses_to_check": [], "pitfalls": [], "source": "rule_based"}


def get_rule_only_memo(block_id: str, summary_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Rule-based мемо для блоков без вызова LLM (corpus, evidence, limits). Возвращает структуру как load_memo."""
    summary_dict = summary_dict or {}
    if block_id == "corpus":
        n_raw = summary_dict.get("raw", 0)
        n_clean = summary_dict.get("clean", 0)
        pct = round((n_raw - n_clean) / n_raw * 100, 1) if n_raw else 0
        return {
            "table_type": "corpus",
            "memo_text": f"Исключено как шум {n_raw - n_clean} из {n_raw} упоминаний ({pct}%). Метрики считаны по очищенной выборке.",
            "hypotheses_to_check": ["Проверить репрезентативность по документам.", "Просмотреть noise_with_ethnonyms.csv."],
            "pitfalls": ["Колонтитулы и индексные строки могли содержать этнонимы; исключение повышает релевантность."],
            "source": "rule_based",
        }
    if block_id == "evidence":
        return {
            "table_type": "evidence",
            "memo_text": "Таблица фрагментов для ручной разметки. Используйте source_pointer для верификации в тексте.",
            "hypotheses_to_check": ["Выборочная верификация R/O по контекстам.", "Проверка границ эссенциализации."],
            "pitfalls": ["Объём выборки ограничен; при необходимости расширьте экспорт из piro_fragments.xlsx."],
            "source": "rule_based",
        }
    if block_id == "limits":
        return {
            "table_type": "limits",
            "memo_text": "Ограничения: классификация R/O по лексиконам; keyness чувствителен к объёму подкорпусов; сети отражают только явные маркеры.",
            "hypotheses_to_check": ["Проверить долю uncertain/unknown.", "Сопоставить ключевые слова с конкордансом."],
            "pitfalls": ["Возможны артефакты OCR и повторяющихся формулировок в травелогах."],
            "source": "rule_based",
        }
    return {"table_type": block_id, "memo_text": "", "hypotheses_to_check": [], "pitfalls": [], "source": "rule_based"}


SYNTHESIS_USER_PROMPT = """По данным производных метрик (профили этносов, индексы OI/ED/EPS/AS, тесты, корреляции, кластеры) сформируй синтетическую интерпретацию на русском языке.

Структура ответа — только JSON с ключами:
- observation: краткое наблюдение по данным (что видно по топам OI/ED, по корреляциям и тестам).
- theoretical_framing: как это соотносится с теорией (ориентализм, репрезентация «других»).
- diagnostics: диагностика — какие индексы согласованы, какие расходятся, что говорит Cramér's V.
- hypotheses: массив ровно из 3 проверяемых гипотез.
- risks: массив рисков и ограничений (методология, объём выборки, шум).

Ответ — только JSON, без markdown и пояснений вне JSON."""


def generate_synthesis_memo(
    payload: Dict[str, Any],
    run_llm: bool = True,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Генерирует synthesis.json по payload производных метрик (топ OI/ED с CI, корреляции, кластеры, chi2, noise).
    Ответ на русском: observation → theoretical_framing → diagnostics → 3 гипотезы → risks.
    Сохраняет output/llm_memos/synthesis.json; в отчёт вставляется блок «что передано в LLM» (n, поля, noise_share).
    Если --run-llm отключен или нет API ключа — rule-based синтез с пометкой «LLM отключен».
    """
    output_path = output_path or MEMOS_DIR / "synthesis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Краткое описание payload для отчёта
    n_mentions = payload.get("n_mentions") or 0
    fields_used = payload.get("fields_used") or ["OI", "ED", "EPS", "AS", "mentions_norm"]
    noise_share = payload.get("noise_share")
    llm_payload_summary = {
        "n_mentions": n_mentions,
        "n_ethnos": payload.get("n_ethnos"),
        "fields_used": fields_used,
        "noise_share": noise_share,
        "has_chi2_R": "chi2_R" in payload,
        "has_chi2_O": "chi2_O" in payload,
        "has_correlations": "correlations" in payload,
        "has_clusters": "clusters" in payload,
    }

    def _to_native(obj):
        import numpy as np
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_native(x) for x in obj]
        return obj

    payload = _to_native(payload)
    if run_llm:
        system_prompt = get_system_prompt(use_knowledge=True)
        data_json = json.dumps(payload, ensure_ascii=False, indent=2)
        response = call_deepseek(system_prompt, SYNTHESIS_USER_PROMPT, data_json, call_id="synthesis")
        parsed = _parse_llm_response(response)
        if parsed and (parsed.get("observation") or parsed.get("hypotheses")):
            out = {
                "observation": parsed.get("observation", ""),
                "theoretical_framing": parsed.get("theoretical_framing", ""),
                "diagnostics": parsed.get("diagnostics", ""),
                "hypotheses": parsed.get("hypotheses", []),
                "risks": parsed.get("risks", []),
                "source": "llm",
                "llm_payload_summary": llm_payload_summary,
            }
        else:
            out = _rule_based_synthesis(llm_payload_summary)
    else:
        out = _rule_based_synthesis(llm_payload_summary)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def _rule_based_synthesis(llm_payload_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Короткий rule-based синтез при отключённом LLM."""
    return {
        "observation": "Производные индексы (OI, ED, EPS, AS) и профили этносов рассчитаны по очищенной выборке. Рекомендуется просмотр топов по OI и ED и корреляций в отчёте.",
        "theoretical_framing": "Индексы ориентированы на выявление ориентализирующих и эссенциализирующих паттернов репрезентации.",
        "diagnostics": "Согласованность индексов и значимость связей (Cramér's V, корреляции) требуют ручной интерпретации. Запустите с --run-llm для синтетической интерпретации.",
        "hypotheses": [
            "Этносы с высоким OI чаще имеют повышенный ED.",
            "Корреляция OI и mentions_norm может отражать объём репрезентации, а не только тон.",
            "Кластеры этносов различаются по комбинации OI/ED/EPS/AS.",
        ],
        "risks": ["Классификация R/O по лексиконам; малый объём упоминаний по части этносов искажает индексы."],
        "source": "rule_based",
        "llm_disabled_note": "LLM отключен",
        "llm_payload_summary": llm_payload_summary,
    }


def load_all_memos(output_dir: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Загружает все сохранённые мемо из output/llm_memos/*.json. Для встраивания в отчёт без повторных вызовов API."""
    output_dir = output_dir or MEMOS_DIR
    result = {}
    for table_type in USER_PROMPTS:
        memo = load_memo(table_type, output_dir)
        if memo:
            result[table_type] = memo
    return result


def _system_prompt_hash() -> str:
    """Хеш содержимого системного промпта для воспроизводимости."""
    import hashlib
    text = get_system_prompt(use_knowledge=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def write_llm_metadata(
    run_llm: bool,
    output_dir: Optional[Path] = None,
    seed: Optional[int] = None,
) -> None:
    """Записывает output/metadata/llm_mode.json и run_config.json для воспроизводимости."""
    from datetime import datetime
    output_dir = output_dir or PROJECT_ROOT / "output"
    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    prompt_hash = _system_prompt_hash()
    llm_mode = {
        "date": datetime.utcnow().isoformat() + "Z",
        "system_prompt_version": prompt_hash,
        "run_llm": run_llm,
        "model": "deepseek-chat",
        "seed": seed,
    }
    path_mode = meta_dir / "llm_mode.json"
    with open(path_mode, "w", encoding="utf-8") as f:
        json.dump(llm_mode, f, ensure_ascii=False, indent=2)
    run_config = {
        "date": datetime.utcnow().isoformat() + "Z",
        "run_llm": run_llm,
        "memos_dir": str(MEMOS_DIR),
    }
    path_config = meta_dir / "run_config.json"
    with open(path_config, "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
