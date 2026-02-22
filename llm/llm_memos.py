"""
Структурированные мемо для блоков отчёта: только по числам, проверяемые гипотезы, риски.
DeepSeek вызывается только по агрегатам; при отсутствии API — rule-based шаблоны.
Формат: memo_id, title, inputs_summary, memo_text, hypotheses_to_check[], pitfalls[]
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

RESEARCHER_SYSTEM = """Ты методолог качества данных и исследователь травелогов.
Твоя роль: осторожные формулировки, явные допущения, указание на необходимость ручной проверки.
ЗАПРЕЩЕНО: интерпретировать мир без опоры на переданные числа; добавлять внешние факты.
ОБЯЗАТЕЛЬНО: опираться только на переданные статистические данные; формулировать проверяемые гипотезы; указывать артефакты и риски данных.
Ответь строго в JSON с ключами: memo_text (3–6 предложений), hypotheses_to_check (массив строк), pitfalls (массив строк)."""


def _load_env_file() -> None:
    if os.environ.get("DEEPSEEK_API_KEY"):
        return
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k == "DEEPSEEK_API_KEY" and v:
                    os.environ["DEEPSEEK_API_KEY"] = v
                    break
    except Exception:
        pass


def call_deepseek_for_memo(payload: Dict[str, Any], api_key: Optional[str] = None) -> Dict[str, Any]:
    """Один запрос: payload = {block_id, title, inputs}. Возвращает {memo_text, hypotheses_to_check, pitfalls}."""
    _load_env_file()
    key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        return {}
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
        user_content = (
            "По следующим агрегированным данным для раздела отчёта сформируй JSON: "
            "memo_text (краткая аналитика строго по числам, 3–6 предложений), "
            "hypotheses_to_check (массив из 1–4 проверяемых гипотез для ручной верификации), "
            "pitfalls (массив из 1–3 рисков/артефактов данных). Только JSON.\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": RESEARCHER_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
        )
        text = resp.choices[0].message.content.strip()
        if "```" in text:
            for part in text.split("```"):
                if part.strip().startswith("json"):
                    part = part.strip()[4:]
                if part.strip().startswith("{"):
                    return json.loads(part)
        return json.loads(text)
    except Exception:
        return {}


def rule_based_memo(block_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Шаблонный мемо без API: по block_id и краткому inputs."""
    memo_text = "Аналитический комментарий сформирован без вызова LLM (нет ключа или ошибка). Рекомендуется ручная проверка цифр и выборки."
    hypotheses = ["Проверить репрезентативность выборки по документам.", "Выборочно верифицировать разметку R/O по контекстам."]
    pitfalls = ["Классификация R/O основана на лексиконах; возможны пропуски контекстов."]
    if block_id == "corpus":
        n_raw = inputs.get("raw", 0)
        n_clean = inputs.get("clean", 0)
        pct = round((n_raw - n_clean) / n_raw * 100, 1) if n_raw else 0
        memo_text = f"Исключено как шум {n_raw - n_clean} из {n_raw} упоминаний ({pct}%). Метрики ниже считаны по очищенной выборке. При высокой доле шума интерпретация частот может быть смещена."
        pitfalls = ["Колонтитулы и индексные строки могли содержать этнонимы; их исключение уменьшает объём, но повышает релевантность."]
    elif block_id == "representations":
        memo_text = "Распределение R (тип репрезентации) получено по лексиконам и эпитетам. Нормировка на 10k предложений позволяет сравнивать документы разного объёма."
        pitfalls = ["Лексиконы не покрывают все маркеры негатива/экзотизации; доля uncertain отражает низкую уверенность классификатора."]
    elif block_id == "keyness":
        memo_text = "Keyness (G2) выделяет слова, характерные для подкорпуса относительно референса. Фильтр по POS (NOUN/ADJ/VERB) и отсечение OCR-подобных токенов снижают шум. Интерпретация: топ-30 слов — кандидаты на маркеры ориенталистских сдвигов; требуется качественный разбор контекстов."
        hypotheses = ["Проверить конкорданс по топ-10 keyness-слов вручную.", "Сравнить эпитетный слой (ADJ) с контентными словами."]
    elif block_id == "essentialization":
        memo_text = "Частоты эссенциализирующих конструкций по этносам. Примеры дедуплицированы по тексту предложения. Рекомендуется выборочная проверка по source_pointer."
        pitfalls = ["Паттерны (the X are, X by nature) могут давать ложные срабатывания на не-этнонимы в отдельных контекстах."]
    elif block_id == "networks":
        memo_text = "Co-mention: совместное упоминание в контексте. Interaction: грамматическое отношение субъект–глагол–объект по глаголам из словаря. Строки, похожие на индекс/оглавление, исключены из interaction. Примеры на рёбрах — типичные предложения; при пагинации возможны повторы."
        pitfalls = ["Сеть interaction чувствительна к качеству разбора зависимостей (spaCy); возможны пропуски связей."]
    elif block_id == "embedding":
        memo_text = "Кластеризация контекстов по эмбеддингам — валидация: проверка, что негатив/экзотика расходятся по семантическим зонам, а не являются артефактом лексикона."
        pitfalls = ["Размер выборки и параметры UMAP/HDBSCAN влияют на число кластеров; интерпретация кластеров требует ручного просмотра примеров."]
    return {
        "memo_text": memo_text,
        "hypotheses_to_check": hypotheses,
        "pitfalls": pitfalls,
    }


def build_memo(
    memo_id: str,
    title: str,
    inputs: Dict[str, Any],
    use_llm: bool = True,
) -> Dict[str, Any]:
    """
    Строит один мемо: {memo_id, title, inputs_summary, memo_text, hypotheses_to_check, pitfalls}.
    inputs_summary — краткое описание переданных данных (числа, не сырые тексты).
    """
    inputs_summary = {}
    for k, v in list(inputs.items())[:15]:
        if isinstance(v, (dict, list)) and len(str(v)) > 500:
            inputs_summary[k] = f"<{type(v).__name__} len={len(v)}>"
        else:
            inputs_summary[k] = v
    payload = {"block_id": memo_id, "title": title, "inputs": inputs}
    if use_llm:
        llm_out = call_deepseek_for_memo(payload)
        if llm_out:
            return {
                "memo_id": memo_id,
                "title": title,
                "inputs_summary": inputs_summary,
                "memo_text": llm_out.get("memo_text", ""),
                "hypotheses_to_check": llm_out.get("hypotheses_to_check", []),
                "pitfalls": llm_out.get("pitfalls", []),
            }
    rb = rule_based_memo(memo_id, inputs)
    return {
        "memo_id": memo_id,
        "title": title,
        "inputs_summary": inputs_summary,
        "memo_text": rb.get("memo_text", ""),
        "hypotheses_to_check": rb.get("hypotheses_to_check", []),
        "pitfalls": rb.get("pitfalls", []),
    }


def run_all_memos(
    corpus_stats: Dict[str, Any],
    norm_ethnos: Dict,
    norm_R: Dict,
    norm_O: Dict,
    keyness_top: Dict[str, List[str]],
    essentialization_counts: Dict[str, int],
    noise_stats: Dict[str, int],
    interaction_summary: Optional[Dict] = None,
    cluster_validation: Optional[Dict] = None,
    use_llm: bool = True,
    output_path: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Генерирует мемо по каждому блоку отчёта, сохраняет output/llm_memos.json.
    Возвращает {memo_id: memo_dict}.
    """
    output_path = output_path or Path(__file__).resolve().parent.parent / "output" / "llm_memos.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    memos = {}
    blocks = [
        ("corpus", "Корпус и очистка", {"raw": noise_stats.get("raw", 0), "clean": noise_stats.get("clean", 0), "removed": noise_stats.get("removed", 0)}),
        ("representations", "Репрезентации (R)", {"R_counts": list(norm_R.keys()) if norm_R else [], "norm_R": norm_R}),
        ("situations", "Ситуации (O)", {"norm_O": norm_O}),
        ("keyness", "Keyness", {"top_keyness": keyness_top}),
        ("essentialization", "Эссенциализация", {"counts": essentialization_counts}),
        ("networks", "Сети", {"interaction_summary": interaction_summary or {}}),
        ("evidence", "Evidence Pack", {"note": "Таблица фрагментов для ручной разметки."}),
        ("embedding", "Embedding validation", {"cluster_validation": cluster_validation or {}}),
        ("limits", "Ограничения", {"note": "Автопроверки и ограничения метода."}),
    ]
    for memo_id, title, inputs in blocks:
        memo = build_memo(memo_id, title, inputs, use_llm=use_llm)
        memos[memo_id] = memo
    out_data = {"memos": memos, "blocks_order": [b[0] for b in blocks]}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    return memos
