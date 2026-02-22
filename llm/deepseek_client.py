"""
Клиент DeepSeek: загрузка системного промпта, вызов API с логированием и учётом токенов.
Режим dry_run при отсутствии API ключа — возврат заглушки "LLM disabled".
"""

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent / "system_prompt.txt"
LOG_PATH = PROJECT_ROOT / "output" / "logs" / "llm_calls.log"
TOKEN_COUNT_PATH = PROJECT_ROOT / "output" / "metadata" / "llm_token_count.json"
MODEL_NAME = "deepseek-chat"

# Цены DeepSeek API (USD за 1M токенов), deepseek-chat. Источник: https://api-docs.deepseek.com/quick_start/pricing-details-usd
DEEPSEEK_PRICE_INPUT_PER_1M = 0.27   # cache miss
DEEPSEEK_PRICE_OUTPUT_PER_1M = 1.10


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    """Оценка стоимости в USD по тарифам DeepSeek (deepseek-chat)."""
    return (prompt_tokens / 1_000_000) * DEEPSEEK_PRICE_INPUT_PER_1M + (completion_tokens / 1_000_000) * DEEPSEEK_PRICE_OUTPUT_PER_1M


def _load_env_key() -> Optional[str]:
    if os.environ.get("DEEPSEEK_API_KEY"):
        return os.environ.get("DEEPSEEK_API_KEY")
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return None
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k == "DEEPSEEK_API_KEY" and v:
                    return v
    except Exception:
        pass
    return None


def load_system_prompt(prompt_path: Optional[Path] = None) -> str:
    """Читает системный промпт из system_prompt.txt."""
    path = prompt_path or DEFAULT_PROMPT_PATH
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _log_call(user_prompt: str, data_json: str, response: str, error: Optional[str] = None) -> None:
    """Пишет в output/logs/llm_calls.log входные данные и ответ (размеры и метаданные, не полные тексты)."""
    log_dir = LOG_PATH.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "user_prompt_len": len(user_prompt),
        "data_json_len": len(data_json),
        "response_len": len(response),
        "error": error,
    }
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def reset_token_count() -> None:
    """Сбрасывает счётчик токенов для текущего запуска (вызывать в начале пайплайна с --run-llm)."""
    TOKEN_COUNT_PATH.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.utcnow().isoformat() + "Z"
    data = {
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "n_calls": 0,
        "n_calls_llm_used": 0,
        "n_calls_without_llm": 0,
        "estimated_cost_usd": 0.0,
        "last_updated": now,
        "calls": [],
        "note": "Счётчик сброшен в начале этого запуска. LLM участвовал в работе, если n_calls_llm_used > 0 и total_tokens > 0.",
    }
    try:
        TOKEN_COUNT_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _record_token_usage(
    call_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    error: Optional[str] = None,
) -> None:
    """Обновляет output/metadata/llm_token_count.json: добавляет вызов и пересчитывает итоги."""
    TOKEN_COUNT_PATH.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.utcnow().isoformat() + "Z"
    call_entry = {
        "call_id": call_id,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "timestamp": now,
        "llm_used": total_tokens > 0 and error is None,
        "error": error,
    }
    data: Dict[str, Any] = {
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "n_calls": 0,
        "n_calls_llm_used": 0,
        "n_calls_without_llm": 0,
        "last_updated": now,
        "calls": [],
    }
    try:
        if TOKEN_COUNT_PATH.exists():
            data = json.loads(TOKEN_COUNT_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    data["calls"] = data.get("calls") or []
    data["calls"].append(call_entry)
    data["total_prompt_tokens"] = data.get("total_prompt_tokens", 0) + prompt_tokens
    data["total_completion_tokens"] = data.get("total_completion_tokens", 0) + completion_tokens
    data["total_tokens"] = data.get("total_tokens", 0) + total_tokens
    data["n_calls"] = data.get("n_calls", 0) + 1
    if total_tokens > 0 and error is None:
        data["n_calls_llm_used"] = data.get("n_calls_llm_used", 0) + 1
        # Накопительная оценка стоимости (DeepSeek deepseek-chat, USD)
        data["estimated_cost_usd"] = data.get("estimated_cost_usd", 0.0) + estimate_cost_usd(prompt_tokens, completion_tokens)
    else:
        data["n_calls_without_llm"] = data.get("n_calls_without_llm", 0) + 1
    data["last_updated"] = now
    try:
        TOKEN_COUNT_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def call_deepseek(
    system_prompt: str,
    user_prompt: str,
    data_json: str,
    api_key: Optional[str] = None,
    call_id: Optional[str] = None,
) -> str:
    """
    Отправляет запрос в DeepSeek.
    system_prompt — текст из system_prompt.txt,
    user_prompt — инструкция,
    data_json — сериализованный JSON с данными (добавляется к user_prompt).
    Возвращает текст ответа. Логирует в output/logs/llm_calls.log.
    Режим dry_run: при отсутствии API ключа возвращает "LLM disabled".
    """
    cid = call_id or "unknown"
    key = api_key or _load_env_key()
    if not key:
        _log_call(user_prompt, data_json, "LLM disabled", error="no_api_key")
        _record_token_usage(cid, 0, 0, 0, error="no_api_key")
        return "LLM disabled"

    full_user = user_prompt + "\n\n" + data_json if data_json else user_prompt
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt or load_system_prompt()},
                {"role": "user", "content": full_user},
            ],
            temperature=0.2,
        )
        text = (resp.choices[0].message.content or "").strip()
        _log_call(user_prompt, data_json, text, error=None)
        # Учёт токенов (OpenAI-совместимый usage)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            pt = getattr(usage, "prompt_tokens", 0) or 0
            ct = getattr(usage, "completion_tokens", 0) or 0
            tt = getattr(usage, "total_tokens", None) or (pt + ct)
            _record_token_usage(cid, pt, ct, tt, error=None)
        else:
            _record_token_usage(cid, 0, 0, 0, error="no_usage_in_response")
        # Пауза между вызовами, чтобы реже упираться в rate limit API
        time.sleep(1.5)
        return text
    except Exception as e:
        err_msg = str(e)
        _log_call(user_prompt, data_json, "", error=err_msg)
        _record_token_usage(cid, 0, 0, 0, error=err_msg)
        return "LLM disabled"
