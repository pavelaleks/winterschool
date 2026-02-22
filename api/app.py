"""
Локальный Flask API для реактивной LLM-аналитики дашборда.
POST /api/analyze: body { block_id, summary_json } → memo + meta.
Кэш по hash(summary_json) в output/llm_cache/.
"""

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Корень проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = OUTPUT_DIR / "llm_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _prompt_hash() -> str:
    from llm.deepseek_client import load_system_prompt
    text = load_system_prompt()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _get_cached(cache_key: str) -> dict | None:
    path = CACHE_DIR / f"{cache_key}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(cache_key: str, data: dict) -> None:
    path = CACHE_DIR / f"{cache_key}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


USER_PROMPTS = {
    "corpus": "Ниже — сводка по корпусу и этносам. Дай JSON: memo_text, hypotheses_to_check, pitfalls.",
    "representation": "Ниже — агрегированные данные по типу репрезентации (R). Дай JSON: memo_text, hypotheses_to_check, pitfalls.",
    "representations": "Ниже — агрегированные данные по типу репрезентации (R). Дай JSON: memo_text, hypotheses_to_check, pitfalls.",
    "situation": "Ниже — агрегированные данные по типу ситуации (O). Дай JSON: memo_text, hypotheses_to_check, pitfalls.",
    "situations": "Ниже — агрегированные данные по типу ситуации (O). Дай JSON: memo_text, hypotheses_to_check, pitfalls.",
    "keyness": "Ниже — данные keyness. Дай JSON: memo_text, hypotheses_to_check, pitfalls.",
    "network": "Ниже — данные по сетям. Дай JSON: memo_text, hypotheses_to_check, pitfalls.",
    "networks": "Ниже — данные по сетям. Дай JSON: memo_text, hypotheses_to_check, pitfalls.",
    "essentialization": "Ниже — данные по эссенциализации. Дай JSON: memo_text, hypotheses_to_check, pitfalls.",
    "embeddings": "Ниже — метрики кластеризации. Дай JSON: memo_text, hypotheses_to_check, pitfalls.",
    "embedding": "Ниже — метрики кластеризации. Дай JSON: memo_text, hypotheses_to_check, pitfalls.",
    "evidence": "Ниже — данные Evidence Pack. Дай JSON: memo_text, hypotheses_to_check, pitfalls.",
    "limits": "Ниже — ограничения и проверки. Дай JSON: memo_text, hypotheses_to_check, pitfalls.",
}


def _rule_memo(block_id: str) -> dict:
    return {
        "memo_text": "LLM-анализ отключен или недоступен. Используется rule-based сводка. Включите --run-llm и задайте DEEPSEEK_API_KEY для реактивной аналитики.",
        "hypotheses_to_check": ["Проверить репрезентативность выборки.", "Верифицировать разметку по контекстам."],
        "pitfalls": ["Классификация по лексиконам; возможны пропуски."],
        "source": "rule_based",
    }


def _parse_llm_response(text: str) -> dict:
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


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Body: { "block_id": "representation"|"situation"|..., "summary_json": { ... } }.
    Returns: { memo_text, hypotheses_to_check, pitfalls, meta: { date, system_prompt_version, model, n_records, noise_share } }.
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
        block_id = body.get("block_id") or ""
        summary = body.get("summary_json") or {}
        summary_str = json.dumps(summary, sort_keys=True, ensure_ascii=False)
        cache_key = hashlib.sha256(summary_str.encode("utf-8")).hexdigest()[:32]
        n_records = summary.get("n_records") or summary.get("counts", {}).get("total") or 0
        if isinstance(n_records, dict):
            n_records = sum(n_records.values()) if n_records else 0
        noise_share = summary.get("noise_share")

        cached = _get_cached(cache_key)
        if cached:
            cached["meta"] = cached.get("meta") or {}
            cached["meta"]["cached"] = True
            return jsonify(cached)

        from llm.deepseek_client import load_system_prompt, call_deepseek
        system_prompt = load_system_prompt()
        user_prompt = USER_PROMPTS.get(block_id, "Дай JSON: memo_text, hypotheses_to_check, pitfalls.")
        response_text = call_deepseek(system_prompt, user_prompt, summary_str, call_id=block_id or "api_analyze")
        parsed = _parse_llm_response(response_text)
        if parsed:
            memo_text = parsed.get("memo_text", "")
            hypotheses = parsed.get("hypotheses_to_check", [])
            pitfalls = parsed.get("pitfalls", [])
        else:
            rb = _rule_memo(block_id)
            memo_text = rb["memo_text"]
            hypotheses = rb["hypotheses_to_check"]
            pitfalls = rb["pitfalls"]

        meta = {
            "date": datetime.utcnow().isoformat() + "Z",
            "system_prompt_version": _prompt_hash(),
            "model": "deepseek-chat",
            "n_records": n_records,
            "noise_share": noise_share,
            "cached": False,
        }
        out = {
            "memo_text": memo_text,
            "hypotheses_to_check": hypotheses,
            "pitfalls": pitfalls,
            "meta": meta,
        }
        if "LLM disabled" not in response_text and parsed:
            _save_cache(cache_key, out)
        return jsonify(out)
    except Exception as e:
        return jsonify({
            "memo_text": f"Ошибка: {e}.",
            "hypotheses_to_check": [],
            "pitfalls": [],
            "meta": {"error": str(e)},
        }), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "cache_dir": str(CACHE_DIR)})


def run_server(host: str = "127.0.0.1", port: int = 5000):
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    run_server(port=int(os.environ.get("PORT", 5000)))
