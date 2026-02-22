"""
Самопроверки после пайплайна: доля шума, ключевые слова на OCR-мусор, примеры interaction не индексные.
Логи: output/logs/run_*.log
"""

import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

NOISE_RATIO_WARN_THRESHOLD = 0.5  # если > 50% помечено как шум — предупреждение
INDEX_LIKE_RE = re.compile(
    r"^(?:[A-Z][A-Z\s]+\s+\d+)\s*$|^[\s\.,\d\-]+$|(?:,\s*\d+\s*){3,}",
    re.IGNORECASE,
)


def _is_likely_ocr(word: str) -> bool:
    """Эвристика: слово похоже на OCR-мусор."""
    if not word or len(word) < 3:
        return True
    if not word.isalpha():
        return True
    w = word.lower()
    if re.search(r"(.)\1{2,}", w):
        return True
    consonants = 0
    for c in w:
        if c in "aeiouy":
            consonants = 0
        else:
            consonants += 1
            if consonants >= 4:
                return True
    return False


def check_noise_ratio(raw_count: int, clean_count: int, log_path: Optional[Path] = None) -> List[str]:
    """Доля шума не должна превышать порог; иначе предупреждение."""
    warnings = []
    if raw_count <= 0:
        return warnings
    ratio = (raw_count - clean_count) / raw_count
    if ratio > NOISE_RATIO_WARN_THRESHOLD:
        warnings.append(
            f"Доля помеченных как шум упоминаний высока: {ratio:.1%} ({raw_count - clean_count} из {raw_count}). "
            "Рекомендуется проверить настройки фильтра (noise_filter) и выборочно просмотреть noise_with_ethnonyms.csv."
        )
    return warnings


def check_keyness_ocr(keyness_top_words: Dict[str, List[str]], log_path: Optional[Path] = None) -> List[str]:
    """Топ-20 keyness не должны содержать очевидный OCR-мусор."""
    warnings = []
    for name, words in (keyness_top_words or {}).items():
        for w in (words or [])[:20]:
            if _is_likely_ocr(w):
                warnings.append(
                    f"В топе keyness ({name}) встречено слово, похожее на OCR-мусор: «{w}». "
                    "Проверьте фильтр в keyness.py (_is_likely_ocr_garbage)."
                )
                break
    return warnings


def check_interaction_examples_index_like(interaction_edges: List[Dict], log_path: Optional[Path] = None) -> List[str]:
    """Примеры предложений на рёбрах interaction не должны быть индексоподобными."""
    warnings = []
    for e in (interaction_edges or []):
        for ex in (e.get("examples") or [])[:3]:
            if not ex:
                continue
            if INDEX_LIKE_RE.match(ex.strip()) or INDEX_LIKE_RE.search(ex.strip()):
                warnings.append(
                    f"В примерах interaction встречена строка, похожая на индекс/оглавление: «{ex[:80]}…». "
                    "Усильте фильтр is_normal_sentence_for_interaction в relations.py."
                )
                break
    return warnings


def run_sanity_checks(
    raw_count: int,
    clean_count: int,
    keyness_top_words: Optional[Dict[str, List[str]]] = None,
    interaction_edges: Optional[List[Dict]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Запускает все проверки, пишет лог в output/logs/run_YYYYMMDD_HHMMSS.log.
    Возвращает {passed: bool, warnings: [...], log_path: str}.
    """
    output_dir = output_dir or Path(__file__).resolve().parent / "output"
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"run_{stamp}.log"

    all_warnings = []
    all_warnings.extend(check_noise_ratio(raw_count, clean_count, log_path))
    all_warnings.extend(check_keyness_ocr(keyness_top_words or {}, log_path))
    all_warnings.extend(check_interaction_examples_index_like(interaction_edges or [], log_path))

    lines = [
        f"Sanity checks at {datetime.now().isoformat()}",
        f"Raw mentions: {raw_count}, clean: {clean_count}",
        f"Warnings: {len(all_warnings)}",
    ]
    for w in all_warnings:
        lines.append(f"  - {w}")
    log_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "passed": len(all_warnings) == 0,
        "warnings": all_warnings,
        "log_path": str(log_path),
    }
