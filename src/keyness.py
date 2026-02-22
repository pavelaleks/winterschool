"""
Keyness: log-likelihood (G2) и effect size; POS-фильтры (content_words, adjectives_only, verbs_only, no_proper_nouns).
Таблицы сохраняются в output/tables/. Wordcloud — только в Appendix.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict

import pandas as pd
import numpy as np

# Режимы POS-фильтра (для ключевых слов)
POS_MODES = {
    "content_words": {"NOUN", "ADJ", "VERB"},
    "adjectives_only": {"ADJ"},
    "verbs_only": {"VERB"},
    "no_proper_nouns": None,  # исключить PROPN, остальное оставить
}
EN_STOP_KEYNESS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "as", "is", "was", "are", "were", "been", "be", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "can",
    "this", "that", "these", "those", "it", "its", "they", "them", "we", "our", "i", "you",
    "he", "she", "his", "her", "their", "my", "your", "not", "no", "so", "if", "than",
}

# Минимальная длина слова для keyness (отсекает часть OCR-мусора)
KEYNESS_MIN_WORD_LEN = 3
# Топ-N слов в отчёте (интерпретация по top-30)
KEYNESS_TOP_N_REPORT = 30


def _is_likely_ocr_garbage(word: str) -> bool:
    """
    Эвристика: слово похоже на OCR-мусор (scjuare, luxuiiant).
    - не только буквы a-z -> да;
    - длина < KEYNESS_MIN_WORD_LEN -> да;
    - 4+ подряд согласных без гласной -> да;
    - одна и та же буква 3+ раза подряд -> да.
    """
    if not word or len(word) < KEYNESS_MIN_WORD_LEN:
        return True
    if not word.isalpha() or not word.isascii():
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


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z]+", (text or "").lower())


def _corpus_freq(corpora: List[List[str]]) -> Counter:
    c = Counter()
    for words in corpora:
        c.update(words)
    return c


def log_likelihood(
    w_focus: int,
    w_ref: int,
    n_focus: int,
    n_ref: int,
) -> float:
    """
    G2 (Dunning log-likelihood) для одного слова.
    w_focus, w_ref — частоты слова в фокус- и референс-корпусе;
    n_focus, n_ref — объёмы корпусов (в токенах).
    """
    if n_focus <= 0 or n_ref <= 0:
        return 0.0
    e1 = n_focus * (w_focus + w_ref) / (n_focus + n_ref)
    e2 = n_ref * (w_focus + w_ref) / (n_focus + n_ref)
    if e1 <= 0 or e2 <= 0:
        return 0.0
    term1 = w_focus * np.log(w_focus / e1) if w_focus > 0 else 0
    term2 = w_ref * np.log(w_ref / e2) if w_ref > 0 else 0
    g2 = 2 * (term1 + term2)
    return max(0.0, g2)


def log_ratio_effect(
    w_focus: int,
    w_ref: int,
    n_focus: int,
    n_ref: int,
    prior: float = 0.5,
) -> float:
    """Log-odds ratio с prior (упрощённый effect size)."""
    if n_focus <= 0 or n_ref <= 0:
        return 0.0
    p_f = (w_focus + prior) / (n_focus + 2 * prior)
    p_r = (w_ref + prior) / (n_ref + 2 * prior)
    if p_f <= 0 or p_r <= 0:
        return 0.0
    return np.log2(p_f / p_r)


def _tokenize_with_pos(
    texts: List[str],
    nlp: Any,
    pos_mode: str = "content_words",
) -> List[str]:
    """
    Токенизация с POS-фильтром: удаляются стоп-слова, числительные, служебные.
    pos_mode: content_words (NOUN/ADJ/VERB), adjectives_only, verbs_only, no_proper_nouns.
    """
    allow_pos = POS_MODES.get(pos_mode)
    exclude_propn = pos_mode == "no_proper_nouns" or (allow_pos and "PROPN" not in str(allow_pos))
    tokens = []
    for text in (texts or [])[:5000]:
        if not (text and isinstance(text, str)):
            continue
        try:
            doc = nlp(text[:100000])
            for t in doc:
                if not t.is_alpha or t.like_num:
                    continue
                w = t.lemma_.lower()
                if w in EN_STOP_KEYNESS or len(w) < 2:
                    continue
                if allow_pos and t.pos_ not in allow_pos:
                    continue
                if exclude_propn and t.pos_ == "PROPN":
                    continue
                tokens.append(w)
        except Exception:
            tokens.extend(w for w in _tokenize(text) if w not in EN_STOP_KEYNESS and len(w) >= 2)
    return tokens


def compute_keyness(
    focus_texts: List[str],
    ref_texts: List[str],
    min_freq_focus: int = 2,
    nlp: Optional[Any] = None,
    pos_mode: Optional[str] = None,
) -> pd.DataFrame:
    """
    Считает keyness (G2 и log_ratio). Если nlp и pos_mode заданы — применяется POS-фильтр.
    Возвращает DataFrame: word, freq_focus, freq_ref, G2, log_ratio.
    """
    if nlp and pos_mode:
        focus_tokens = _tokenize_with_pos(focus_texts, nlp, pos_mode)
        ref_tokens = _tokenize_with_pos(ref_texts, nlp, pos_mode)
    else:
        focus_tokens = []
        for t in focus_texts:
            focus_tokens.extend(w for w in _tokenize(t) if w not in EN_STOP_KEYNESS and len(w) >= 2)
        ref_tokens = []
        for t in ref_texts:
            ref_tokens.extend(w for w in _tokenize(t) if w not in EN_STOP_KEYNESS and len(w) >= 2)

    n_focus = len(focus_tokens)
    n_ref = len(ref_tokens)
    cf = Counter(focus_tokens)
    cr = Counter(ref_tokens)
    all_words = set(cf.keys()) | set(cr.keys())

    rows = []
    for w in all_words:
        if _is_likely_ocr_garbage(w):
            continue
        wf = cf.get(w, 0)
        wr = cr.get(w, 0)
        if wf < min_freq_focus:
            continue
        g2 = log_likelihood(wf, wr, n_focus, n_ref)
        lr = log_ratio_effect(wf, wr, n_focus, n_ref)
        rows.append({
            "word": w,
            "freq_focus": wf,
            "freq_ref": wr,
            "G2": round(g2, 4),
            "log_ratio": round(lr, 4),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("G2", ascending=False).reset_index(drop=True)
    return df


def keyness_by_representation(
    piro_records: List[Dict],
    output_dir: Optional[Path] = None,
    tables_dir: Optional[Path] = None,
    nlp: Optional[Any] = None,
    pos_mode: Optional[str] = "content_words",
) -> Dict[str, pd.DataFrame]:
    """
    Сравнения: negative vs neutral, exotic vs neutral, positive vs neutral.
    Сохраняет CSV в tables_dir (или output/tables/). pos_mode: content_words, adjectives_only, verbs_only, no_proper_nouns.
    """
    base = Path(__file__).resolve().parent.parent / "output"
    output_dir = output_dir or base
    tables_dir = tables_dir or base / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    by_r = defaultdict(list)
    for r in piro_records:
        ctx = r.get("context_text") or r.get("context") or ""
        R = r.get("R", "neutral")
        by_r[R].append(ctx)

    results = {}
    suffix = f"_{pos_mode}" if (nlp and pos_mode) else ""
    for focus_label, ref_label in [
        ("negative", "neutral"),
        ("exotic", "neutral"),
        ("positive", "neutral"),
    ]:
        focus_texts = by_r.get(focus_label, [])
        ref_texts = by_r.get("neutral", [])
        if not focus_texts:
            continue
        df = compute_keyness(focus_texts, ref_texts, nlp=nlp, pos_mode=pos_mode)
        name = f"keyness_{focus_label}_vs_{ref_label}{suffix}"
        results[name] = df
        path = tables_dir / f"{name}.csv"
        df.head(200).to_csv(path, index=False, encoding="utf-8")
    return results


def keyness_by_ethnos(
    piro_records: List[Dict],
    output_dir: Optional[Path] = None,
    tables_dir: Optional[Path] = None,
    nlp: Optional[Any] = None,
    pos_mode: Optional[str] = "content_words",
) -> Dict[str, pd.DataFrame]:
    """Для каждого этноса: контексты этого этноса vs все остальные. CSV в output/tables/."""
    base = Path(__file__).resolve().parent.parent / "output"
    output_dir = output_dir or base
    tables_dir = tables_dir or base / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    by_ethnos = defaultdict(list)
    for r in piro_records:
        ctx = r.get("context_text") or r.get("context") or ""
        P = r.get("P") or r.get("ethnos_norm", "")
        if P:
            by_ethnos[P].append(ctx)
    all_ctx = []
    for ctx_list in by_ethnos.values():
        all_ctx.extend(ctx_list)

    results = {}
    suffix = f"_{pos_mode}" if (nlp and pos_mode) else ""
    for eth, focus_texts in by_ethnos.items():
        ref_texts = [c for e, ctx_list in by_ethnos.items() for c in ctx_list if e != eth]
        if not ref_texts:
            ref_texts = all_ctx
        if not focus_texts:
            continue
        df = compute_keyness(focus_texts, ref_texts, nlp=nlp, pos_mode=pos_mode)
        name = f"keyness_{eth}_vs_rest{suffix}"
        results[name] = df
        path = tables_dir / f"{name}.csv"
        df.head(200).to_csv(path, index=False, encoding="utf-8")
    return results


def plot_keyness_barplot(
    df: pd.DataFrame,
    title: str,
    output_path: Path,
    top_n: int = 20,
    xlabel_ru: str = "Keyness (G2)",
) -> None:
    """Строит barplot топ-N слов по G2. Подписи по-русски где нужно."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = df.head(top_n)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(df)), df["G2"].values, color="#4a6fa5", alpha=0.8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["word"].tolist(), fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel_ru)
    ax.set_title(title)
    ax.set_facecolor("white")
    fig.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def run_keyness_visualizations(
    piro_records: List[Dict],
    output_dir: Optional[Path] = None,
    tables_dir: Optional[Path] = None,
    nlp: Optional[Any] = None,
) -> Dict[str, str]:
    """Запускает keyness по R и по этносам (с POS content_words при наличии nlp), таблицы в output/tables/, barplot'ы в output. Возвращает {название: путь}."""
    base = Path(__file__).resolve().parent.parent / "output"
    output_dir = output_dir or base
    tables_dir = tables_dir or base / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_paths = {}
    pos_mode = "content_words" if nlp else None

    by_r = keyness_by_representation(piro_records, output_dir, tables_dir=tables_dir, nlp=nlp, pos_mode=pos_mode)
    for name, df in by_r.items():
        if df.empty or len(df) < 2:
            continue
        path = output_dir / f"{name}_top20.png"
        title_ru = name.replace("keyness_", "").replace("_vs_", " vs ")
        plot_keyness_barplot(df, title_ru, path, top_n=20, xlabel_ru="Keyness (G2)")
        out_paths[name] = str(path)

    by_eth = keyness_by_ethnos(piro_records, output_dir, tables_dir=tables_dir, nlp=nlp, pos_mode=pos_mode)
    for name, df in by_eth.items():
        if df.empty or len(df) < 2:
            continue
        path = output_dir / f"{name}_top20.png"
        plot_keyness_barplot(df, name.replace("keyness_", ""), path, top_n=20, xlabel_ru="Keyness (G2)")
        out_paths[name] = str(path)

    return out_paths
