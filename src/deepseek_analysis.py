"""
Модуль анализа через DeepSeek: только агрегаты и валидационные данные.
Использует теоретическую рамку (база знаний) для интерпретации в духе ориентализма и травелогов.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List


def _get_system_prompt() -> str:
    """Системный промпт с опорой на базу знаний (ориентализм, травелоги, имперский дискурс)."""
    try:
        from llm.knowledge_loader import get_system_prompt
        return get_system_prompt(use_knowledge=True)
    except Exception:
        return """Ты исследователь травелогов и ориентализма. Анализируй только переданные статистические данные. Формулируй наблюдения, вероятные паттерны и ограничения в контексте репрезентации «других» и ориенталистского дискурса."""


def build_payload(
    normalized_rates_ethnos: Optional[Dict[str, Dict[str, Any]]] = None,
    normalized_rates_R: Optional[Dict[str, Dict[str, Any]]] = None,
    normalized_rates_O: Optional[Dict[str, Dict[str, Any]]] = None,
    keyness_top20: Optional[Dict[str, List[Dict]]] = None,
    cluster_validation: Optional[Dict[str, Any]] = None,
    share_unknown_uncertain: Optional[Dict[str, float]] = None,
    noise_stats: Optional[Dict[str, Any]] = None,
    essentialization: Optional[Dict[str, int]] = None,
    interaction_summary: Optional[Dict[str, Any]] = None,
    # Обратная совместимость
    frequencies: Optional[Dict[str, int]] = None,
    heatmap_representation: Optional[Dict] = None,
    heatmap_situation: Optional[Dict] = None,
    interaction_matrix: Optional[Dict[str, Dict[str, int]]] = None,
    exoticization: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Формирует JSON для LLM: только агрегаты, без сырых текстов."""
    data = {}
    if normalized_rates_ethnos is not None:
        data["normalized_rates_by_ethnos"] = normalized_rates_ethnos
    if normalized_rates_R is not None:
        data["normalized_rates_by_R"] = normalized_rates_R
    if normalized_rates_O is not None:
        data["normalized_rates_by_O_situation"] = normalized_rates_O
    if keyness_top20 is not None:
        data["keyness_top20"] = keyness_top20
    if cluster_validation is not None:
        data["cluster_validation"] = cluster_validation
    if share_unknown_uncertain is not None:
        data["share_unknown_uncertain"] = share_unknown_uncertain
    if noise_stats is not None:
        data["noise_filter_stats"] = noise_stats
    if essentialization is not None:
        data["essentialization_counts"] = essentialization
    if interaction_summary is not None:
        data["interaction_summary"] = interaction_summary
    if frequencies is not None:
        data["frequencies"] = frequencies
    if heatmap_representation is not None:
        data["heatmap_representation"] = heatmap_representation
    if heatmap_situation is not None:
        data["heatmap_situation"] = heatmap_situation
    if interaction_matrix is not None:
        data["interaction_matrix"] = interaction_matrix
    if exoticization is not None:
        data["exoticization"] = exoticization
    return data


def _load_env_file() -> None:
    import os
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


def call_deepseek(payload: Dict[str, Any], api_key: Optional[str] = None) -> Dict[str, Any]:
    """Отправляет payload в DeepSeek, возвращает { observations, probable_patterns, limitations }."""
    import os
    _load_env_file()
    key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    try:
        from llm.deepseek_client import _record_token_usage
    except Exception:
        _record_token_usage = None
    if not key:
        if _record_token_usage:
            _record_token_usage("run_analysis", 0, 0, 0, error="no_api_key")
        return {
            "observations": ["Анализ LLM пропущен: DEEPSEEK_API_KEY не задан."],
            "probable_patterns": [],
            "limitations": ["Требуется API-ключ DeepSeek."],
        }
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
        system_prompt = _get_system_prompt()
        user_content = (
            "По переданным статистическим данным сформируй интерпретацию в рамках ориентализма и репрезентации «других» в травелогах. "
            "Верни JSON с ключами: observations (массив строк — что видно по данным), probable_patterns (массив — вероятные паттерны репрезентации), limitations (массив — ограничения метода и данных). Только JSON.\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
        )
        usage = getattr(resp, "usage", None)
        if _record_token_usage and usage is not None:
            pt = getattr(usage, "prompt_tokens", 0) or 0
            ct = getattr(usage, "completion_tokens", 0) or 0
            tt = getattr(usage, "total_tokens", None) or (pt + ct)
            _record_token_usage("run_analysis", pt, ct, tt, error=None)
        elif _record_token_usage:
            _record_token_usage("run_analysis", 0, 0, 0, error="no_usage_in_response")
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        result.setdefault("observations", [])
        result.setdefault("probable_patterns", [])
        result.setdefault("limitations", [])
        return result
    except Exception as e:
        if _record_token_usage:
            _record_token_usage("run_analysis", 0, 0, 0, error=str(e))
        return {"observations": [], "probable_patterns": [], "limitations": [str(e)]}


def run_analysis(
    frequencies: Optional[Dict[str, int]] = None,
    heatmap_rep: Optional[Dict] = None,
    heatmap_sit: Optional[Dict] = None,
    interaction_matrix: Optional[Dict[str, Dict[str, int]]] = None,
    essentialization: Optional[Dict[str, int]] = None,
    exoticization: Optional[Dict] = None,
    normalized_rates_ethnos: Optional[Dict] = None,
    normalized_rates_R: Optional[Dict] = None,
    normalized_rates_O: Optional[Dict] = None,
    keyness_top20: Optional[Dict[str, List[Dict]]] = None,
    cluster_validation: Optional[Dict] = None,
    share_unknown_uncertain: Optional[Dict[str, float]] = None,
    noise_stats: Optional[Dict] = None,
    interaction_summary: Optional[Dict[str, Any]] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Собирает payload из переданных агрегатов, вызывает DeepSeek, сохраняет llm_analysis.json."""
    payload = build_payload(
        frequencies=frequencies,
        heatmap_representation=heatmap_rep,
        heatmap_situation=heatmap_sit,
        interaction_matrix=interaction_matrix,
        essentialization=essentialization,
        exoticization=exoticization,
        normalized_rates_ethnos=normalized_rates_ethnos,
        normalized_rates_R=normalized_rates_R,
        normalized_rates_O=normalized_rates_O,
        keyness_top20=keyness_top20,
        cluster_validation=cluster_validation,
        share_unknown_uncertain=share_unknown_uncertain,
        noise_stats=noise_stats,
        interaction_summary=interaction_summary,
    )
    result = call_deepseek(payload)
    out_path = output_path or Path(__file__).resolve().parent.parent / "output" / "llm_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result
