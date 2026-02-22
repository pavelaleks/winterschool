"""
Классификация типа дискурсивной сцены (O_situation) по контексту.
С поддержкой unknown, mixed, confidence и O_scores.
Использует resources/situation_domains.yml.
Поддержка поиска по леммам: context_lemmas и lemmatize=True при загрузке доменов.
"""

import yaml
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter

DOMAINS_PATH = Path(__file__).resolve().parent.parent / "resources" / "situation_domains.yml"

# Порог: если top_score < threshold -> unknown
SCORE_THRESHOLD = 1
# Если разница между top1 и top2 мала -> mixed
COMPETITION_RATIO = 0.6  # top2 >= ratio * top1 -> mixed


def _lemmatize_word_list(words: List[str]) -> List[str]:
    """Приводит список слов к леммам через spaCy (если доступен)."""
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


def load_situation_domains(lemmatize: bool = False) -> Dict[str, List[str]]:
    """Загружает situation_domains.yml. При lemmatize=True маркеры доменов приводятся к леммам."""
    if not DOMAINS_PATH.exists():
        return {}
    with open(DOMAINS_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data = data or {}
    if lemmatize:
        return {
            k: _lemmatize_word_list(v) if isinstance(v, list) else _lemmatize_word_list([v])
            for k, v in data.items()
        }
    return data


def _tokenize_lower(text: str) -> List[str]:
    return text.lower().replace(".", " ").replace(",", " ").split()


def classify_situation(
    context_text: str,
    domains: Optional[Dict[str, List[str]]] = None,
    score_threshold: int = SCORE_THRESHOLD,
    context_lemmas: Optional[List[str]] = None,
    use_lemma_domains: bool = False,
) -> str:
    """
    По тексту контекста определяет доминирующий домен ситуации.
    context_lemmas / use_lemma_domains — для поиска по леммам.
    """
    domains = domains or load_situation_domains(lemmatize=use_lemma_domains)
    result = classify_situation_full(
        context_text, domains=domains, score_threshold=score_threshold,
        context_lemmas=context_lemmas,
    )
    return result["O_situation"]


def classify_situation_full(
    context_text: str,
    domains: Optional[Dict[str, List[str]]] = None,
    score_threshold: int = SCORE_THRESHOLD,
    context_lemmas: Optional[List[str]] = None,
) -> Dict:
    """
    Возвращает O_situation, O_confidence, O_scores.
    Если передан context_lemmas, совпадения с маркерами доменов ищутся по леммам.
    """
    domains = domains or load_situation_domains()
    if context_lemmas is not None:
        words = set(lem.lower().strip() for lem in context_lemmas if (lem or "").strip())
    else:
        words = set(_tokenize_lower(context_text or ""))
    scores = Counter()
    for domain_name, markers in domains.items():
        for m in markers:
            if m.lower() in words:
                scores[domain_name] += 1

    O_scores = dict(scores)
    if not scores:
        return {
            "O_situation": "unknown",
            "O_confidence": 0.2,
            "O_scores": O_scores,
        }

    sorted_domains = scores.most_common()
    top_domain, top_score = sorted_domains[0]
    second_score = sorted_domains[1][1] if len(sorted_domains) > 1 else 0

    if top_score < score_threshold:
        return {
            "O_situation": "unknown",
            "O_confidence": 0.2,
            "O_scores": O_scores,
        }

    if second_score >= COMPETITION_RATIO * top_score and second_score > 0:
        return {
            "O_situation": "mixed",
            "O_confidence": 0.4,
            "O_scores": O_scores,
        }

    # Сильный лидер
    O_confidence = round(0.5 + 0.1 * min(top_score, 4), 2)
    O_confidence = min(0.9, O_confidence)

    return {
        "O_situation": top_domain,
        "O_confidence": O_confidence,
        "O_scores": O_scores,
    }


def get_context_sentences(
    sentences: List[Dict],
    center_index: int,
    window: int = 4,
) -> List[Dict]:
    """Извлекает контекст ±window предложений вокруг center_index."""
    start = max(0, center_index - window)
    end = min(len(sentences), center_index + window + 1)
    return sentences[start:end]


def get_context_text(sentences: List[Dict], center_index: int, window: int = 4) -> str:
    """Склеивает текст контекста ±window предложений."""
    ctx = get_context_sentences(sentences, center_index, window)
    return " ".join(s["text"] for s in ctx)
