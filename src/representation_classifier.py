"""
Классификация типа репрезентации R с scoring и uncertain.
R: negative | exotic | positive | neutral | uncertain.
Сохраняем R, R_scores (negative, exotic, positive), R_confidence.
Поддержка поиска по леммам: context_lemmas и lemmatize=True при загрузке лексиконов.
"""

import re
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional

RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"
LEXICON_PATH = RESOURCES_DIR / "representation_lexicons.yml"

# Порог разницы между top1 и top2 для назначения uncertain
DELTA_UNCERTAIN = 1
# Минимальный score для назначения категории (иначе neutral + low confidence)
MIN_SCORE_THRESHOLD = 0


def _lemmatize_word_list(words: List[str]) -> List[str]:
    """Приводит список слов к леммам через spaCy (если доступен). Иначе возвращает lower()."""
    if not words:
        return []
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        lemmas = []
        for w in words:
            w = (w or "").strip().lower()
            if not w:
                continue
            doc = nlp(w)
            lemmas.append(doc[0].lemma_ if doc else w)
        return lemmas
    except Exception:
        return [w.lower().strip() for w in words if (w or "").strip()]


def _ensure_string_list(lst) -> List[str]:
    """Приводит элементы списка к строкам (YAML может вернуть true/false как bool)."""
    if not lst or not isinstance(lst, list):
        return []
    return [str(w).strip() for w in lst if w is not None and str(w).strip()]


def load_representation_lexicons(lemmatize: bool = False) -> Dict[str, List[str]]:
    """Загружает representation_lexicons.yml. При lemmatize=True слова приводятся к леммам (spaCy)."""
    if not LEXICON_PATH.exists():
        return {"negative": [], "exotic": [], "positive": []}
    with open(LEXICON_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data = data or {"negative": [], "exotic": [], "positive": []}
    data = {k: _ensure_string_list(v) if isinstance(v, list) else [] for k, v in data.items()}
    if lemmatize:
        return {
            k: _lemmatize_word_list(v) if v else []
            for k, v in data.items()
        }
    return data


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-z]+", text.lower()))


def _proximity_weight(word_pos: int, eth_pos: int, total: int) -> float:
    """Вес по близости к этнониму (упрощённо: по позиции в контексте)."""
    if total <= 0:
        return 1.0
    dist = abs(word_pos - eth_pos)
    return max(0.3, 1.0 - dist / max(total, 1))


def score_representation(
    context_lower: str,
    epithets: List[str],
    lexicons: Dict[str, List[str]],
    context_lemmas: Optional[List[str]] = None,
) -> Tuple[float, float, float]:
    """
    Считает score_negative, score_exotic, score_positive по маркерам в контексте и эпитетах.
    Если передан context_lemmas, совпадения ищутся по леммам (иначе по словоформам из context_lower).
    Эпитеты по-прежнему сравниваются в lower; вес эпитетов 1.2.
    """
    if context_lemmas is not None:
        words = set(lem.lower().strip() for lem in context_lemmas if (lem or "").strip())
    else:
        words = _tokenize(context_lower)
    epithets_lower = [e.lower() for e in epithets]
    all_tokens = set(words) | set(epithets_lower)

    def _norm_lexicon(words):
        return set(str(w).strip().lower() for w in (words or []) if w is not None and str(w).strip())

    neg_set = _norm_lexicon(lexicons.get("negative"))
    exo_set = _norm_lexicon(lexicons.get("exotic"))
    pos_set = _norm_lexicon(lexicons.get("positive"))

    score_neg = sum(1.2 if w in epithets_lower else 1.0 for w in all_tokens if w in neg_set)
    score_exo = sum(1.2 if w in epithets_lower else 1.0 for w in all_tokens if w in exo_set)
    score_pos = sum(1.2 if w in epithets_lower else 1.0 for w in all_tokens if w in pos_set)

    return (score_neg, score_exo, score_pos)


def classify_representation(
    context_text: str,
    epithets: List[str],
    epithet_confidence: float = 0.5,
    delta_uncertain: float = 1.0,
    context_lemmas: Optional[List[str]] = None,
    use_lemma_lexicons: bool = False,
) -> Dict:
    """
    Назначает R и R_confidence.
    context_lemmas: при передаче совпадения с лексиконами ищутся по леммам контекста.
    use_lemma_lexicons: при True лексиконы загружаются и приводятся к леммам (spaCy).
    """
    lexicons = load_representation_lexicons(lemmatize=use_lemma_lexicons)
    context_lower = (context_text or "").lower()
    sn, se, sp = score_representation(
        context_lower, epithets or [], lexicons, context_lemmas=context_lemmas
    )
    scores = {"negative": sn, "exotic": se, "positive": sp}
    max_score = max(sn, se, sp)

    if max_score <= MIN_SCORE_THRESHOLD:
        return {
            "R": "neutral",
            "R_scores": scores,
            "R_confidence": 0.3,
        }

    sorted_cats = sorted(
        [("negative", sn), ("exotic", se), ("positive", sp)],
        key=lambda x: -x[1],
    )
    top1, top1_score = sorted_cats[0]
    top2_score = sorted_cats[1][1] if len(sorted_cats) > 1 else 0

    if top1_score - top2_score < delta_uncertain:
        return {
            "R": "uncertain",
            "R_scores": scores,
            "R_confidence": 0.4,
        }

    # Базовый confidence по силе top1 и эпитетам
    base_conf = 0.5 + 0.15 * min(top1_score, 3)  # cap contribution
    if epithet_confidence >= 0.7:
        base_conf = min(0.9, base_conf + 0.2)
    R_confidence = round(min(0.9, base_conf), 2)

    return {
        "R": top1,
        "R_scores": scores,
        "R_confidence": R_confidence,
    }
