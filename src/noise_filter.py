"""
Фильтр шума: ядро — позиционная дедупликация по (sentence_norm + position_bucket).
Повторяющиеся строки на одинаковых позициях в документе/корпусе маркируются как header/footer.
Дополнительно: regex/частотные маркеры (is_probable_header_footer).
raw_df — все строки с флагами; clean_df — исключены is_noise == True.
Сохраняет: noise_top_repeats.csv, noise_position_summary.csv, noise_with_ethnonyms.csv.
"""

import logging
import re
from pathlib import Path
from typing import Optional, Any

import pandas as pd

logger = logging.getLogger(__name__)

# Маркеры страниц / колонтитулов
PAGE_MARKERS = re.compile(
    r"^(?:page\s*)?\d+\s*$|^p\.\s*\d+\s*$|^pp\.\s*\d+\s*$|"
    r"^[IVXLCDM]+\s*$|^[ivxlcdm]+\s*$|^chapter\s+[IVXLCDM\d]+\s*$",
    re.IGNORECASE,
)
HEADER_WORDS = {"chapter", "part", "book", "section", "volume", "page", "contents", "index", "the"}

# Индексные/оглавленческие строки: "THE WEDDING 17", "..., 66", списки номеров страниц/станций
INDEX_LIKE_PATTERN = re.compile(
    r"^(?:[A-Z][A-Z\s]+\s+\d+)\s*$|"  # THE WEDDING 17
    r"^[\s\.,\d\-]+$|"                  # только цифры, точки, запятые, дефисы
    r"(?:,\s*\d+\s*){3,}|"             # ..., 66, 67, 68 (списки страниц)
    r"^\d+(?:\s*[\-–]\s*\d+)?\s*$",    # одна строка = номер или диапазон
    re.IGNORECASE,
)
# doc_position_bucket: шаг 2% (0, 2, 4, ... 100) для позиционной дедупликации
POSITION_BUCKET_PCT_STEP = 2

# Английские стоп-слова (частые, без NLTK)
EN_STOP = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "as", "is", "was", "are", "were", "been", "be", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "must",
    "can", "this", "that", "these", "those", "it", "its", "they", "them", "we", "our",
    "i", "you", "he", "she", "his", "her", "their", "my", "your",
}

# Границы корзин позиции в документе (0–100%)
BUCKET_BOUNDS = [
    (0, 5, "bucket_start"),
    (5, 20, "bucket_early"),
    (20, 80, "bucket_middle"),
    (80, 95, "bucket_late"),
    (95, 100.01, "bucket_end"),
]
POSITION_HEADER_FOOTER_BUCKETS = {"bucket_start", "bucket_end"}
DEFAULT_K_POSITION = 2  # один и тот же (sentence_norm, position_bucket) в доке повторяется >= K → header/footer
DEFAULT_M_GLOBAL = 5
DEFAULT_N_CROSS_DOC = 3  # (sentence_norm, position_bucket_pct) встречается в >= N документов → cross-doc header/footer
MAX_LEMMAS = 15


def normalize_sentence(
    text: str,
    nlp: Optional[Any] = None,
    max_lemmas: int = MAX_LEMMAS,
    stop_words: Optional[set] = None,
) -> str:
    """
    Нормализация текста для дедупликации:
    нижний регистр, удаление цифр и пунктуации, лишних пробелов;
    опционально лемматизация (spaCy) и удаление стоп-слов;
    обрезка до max_lemmas лемм.
    """
    if not text or not isinstance(text, str):
        return ""
    stop = stop_words if stop_words is not None else EN_STOP
    text = text.lower().strip()
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    if nlp is not None:
        try:
            doc = nlp(text[:5000])
            lemmas = []
            for t in doc:
                if t.is_alpha and t.lemma_.lower() not in stop:
                    lemmas.append(t.lemma_.lower())
                if len(lemmas) >= max_lemmas:
                    break
            return " ".join(lemmas[:max_lemmas])
        except Exception:
            pass

    words = [w for w in text.split() if w.isalpha() and w not in stop]
    return " ".join(words[:max_lemmas])


def doc_position_to_bucket(doc_position_percent: float) -> str:
    """Преобразует doc_position_percent (0–100) в имя корзины."""
    p = float(doc_position_percent) if doc_position_percent is not None else 50.0
    for lo, hi, name in BUCKET_BOUNDS:
        if lo <= p < hi:
            return name
    return "bucket_middle"


def doc_position_bucket_pct(doc_position_percent: float) -> float:
    """Корзина в процентах с шагом POSITION_BUCKET_PCT_STEP (2%): 0, 2, 4, ... 100. Для дедупликации."""
    p = float(doc_position_percent) if doc_position_percent is not None else 50.0
    step = POSITION_BUCKET_PCT_STEP
    return round(p / step) * step


def _is_index_like(sentence: str) -> bool:
    """Строка похожа на элемент оглавления/индекса: заголовок + номер, списки страниц."""
    if not sentence or not isinstance(sentence, str):
        return False
    s = sentence.strip()
    if not s:
        return False
    if INDEX_LIKE_PATTERN.search(s) or INDEX_LIKE_PATTERN.match(s):
        return True
    # "THE WEDDING 17" — несколько слов в верхнем регистре + число
    words = s.split()
    if len(words) >= 2 and words[-1].isdigit() and sum(1 for w in words[:-1] if w.isalpha()) >= 1:
        if _caps_ratio(s) > 0.6:
            return True
    return False


def _word_count(s: str) -> int:
    return len(s.split()) if isinstance(s, str) else 0


def _caps_ratio(s: str) -> float:
    if not s or not s.strip():
        return 0.0
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def _is_mostly_digits_punct(s: str) -> bool:
    if not s or not s.strip():
        return True
    letters = sum(1 for c in s if c.isalpha())
    return letters < 0.3 * len(s.strip())


def _fuzzy_match_title(sentence: str, file_name: str, threshold: int = 3) -> bool:
    if not sentence or not file_name:
        return False
    stem = Path(file_name).stem.lower()
    words_stem = set(stem.replace("_", " ").replace("-", " ").split())
    words_sent = set(sentence.lower().split())
    common = len(words_stem & words_sent)
    return common >= min(threshold, len(words_stem))


def flag_probable_header_footer(
    df: pd.DataFrame,
    repeat_threshold: int = 10,
    min_words: int = 5,
    doc_position_low: float = 2.0,
    doc_position_high: float = 98.0,
    sentence_col: str = "sentence_text",
    file_col: str = "file_name",
    doc_position_col: str = "doc_position_percent",
) -> pd.DataFrame:
    """
    Добавляет колонку is_probable_header_footer.
    """
    out = df.copy()
    if sentence_col not in out.columns:
        out["is_probable_header_footer"] = False
        return out

    sent_counts = out[sentence_col].fillna("").astype(str).value_counts().to_dict()
    flags = []
    for _, row in out.iterrows():
        sent = row.get(sentence_col, "") or ""
        fname = row.get(file_col, "") or ""
        pos = float(row.get(doc_position_col, 50))
        this_repeat = sent_counts.get(str(sent), 0)
        is_noise = False
        if this_repeat >= repeat_threshold:
            is_noise = True
        if _word_count(sent) < min_words:
            is_noise = True
        if _caps_ratio(sent) > 0.8 or _is_mostly_digits_punct(sent):
            is_noise = True
        if PAGE_MARKERS.match(sent.strip()):
            is_noise = True
        words_lower = set(sent.lower().split())
        if words_lower & HEADER_WORDS and _word_count(sent) < 8:
            is_noise = True
        if _fuzzy_match_title(sent, fname):
            is_noise = True
        if (pos <= doc_position_low or pos >= doc_position_high) and this_repeat >= 2:
            is_noise = True
        if _is_index_like(sent):
            is_noise = True
        flags.append(is_noise)
    out["is_probable_header_footer"] = flags
    return out


def add_position_deduplication_flags(
    df: pd.DataFrame,
    sentence_norm_col: str = "sentence_norm",
    position_bucket_col: str = "position_bucket",
    position_bucket_pct_col: Optional[str] = "position_bucket_pct",
    doc_id_col: str = "file_name",
    K: int = DEFAULT_K_POSITION,
    M: int = DEFAULT_M_GLOBAL,
    N_cross_doc: int = DEFAULT_N_CROSS_DOC,
) -> pd.DataFrame:
    """
    Ядро noise-removal: дедупликация по (sentence_norm + position_bucket).
    - (sentence_norm, position_bucket_pct) в одном документе повторяется ≥ K → is_position_header_footer
    - (sentence_norm, position_bucket_pct) встречается в ≥ N_cross_doc документов → is_cross_doc_header_footer
    - (sentence_norm, position_bucket) в одном документе повторяется ≥ M в любом bucket → is_global_repeat
    """
    out = df.copy()
    if sentence_norm_col not in out.columns or position_bucket_col not in out.columns:
        out["is_position_header_footer"] = False
        out["is_global_repeat"] = False
        out["is_cross_doc_header_footer"] = False
        return out

    doc_col = doc_id_col
    if doc_col not in out.columns:
        doc_col = "doc_id"
    if doc_col not in out.columns:
        out["is_position_header_footer"] = False
        out["is_global_repeat"] = False
        out["is_cross_doc_header_footer"] = False
        return out

    # Ключ для позиционной дедупликации: 2% корзина, если есть
    pct_col = position_bucket_pct_col if position_bucket_pct_col and position_bucket_pct_col in out.columns else position_bucket_col

    # Внутри документа: (sentence_norm, position_bucket_pct) → число повторов
    out["_repeat_count"] = 0
    for doc_key, group in out.groupby(doc_col, dropna=False):
        cnts = group.groupby([sentence_norm_col, pct_col], dropna=False).size()
        for (sn, bucket), c in cnts.items():
            mask = (
                (out[doc_col] == doc_key)
                & (out[sentence_norm_col].fillna("") == sn)
                & (out[pct_col].fillna("") == bucket)
            )
            out.loc[mask, "_repeat_count"] = c
    repeat = out["_repeat_count"]
    sn_nonempty = out[sentence_norm_col].fillna("").astype(bool)
    # Один и тот же текст на одной позиции повторяется в доке ≥ K → header/footer (ядро)
    out["is_position_header_footer"] = sn_nonempty & (repeat >= K)
    out["is_global_repeat"] = sn_nonempty & (repeat >= M)

    # Кросс-документ: (sentence_norm, position_bucket_pct) встречается в ≥ N документов → типичный колонтитул
    out["is_cross_doc_header_footer"] = False
    cross = out.groupby([sentence_norm_col, pct_col], dropna=False)[doc_col].nunique()
    for (sn, bucket), n_docs in cross.items():
        if n_docs >= N_cross_doc and sn and str(sn).strip():
            mask = (
                (out[sentence_norm_col].fillna("") == sn)
                & (out[pct_col].fillna("") == bucket)
            )
            out.loc[mask, "is_cross_doc_header_footer"] = True
    out.drop(columns=["_repeat_count"], inplace=True)
    return out


def run_noise_filter(
    df: pd.DataFrame,
    repeat_threshold: int = 10,
    output_dir: Optional[Path] = None,
    nlp: Optional[Any] = None,
    K_position: int = DEFAULT_K_POSITION,
    M_global: int = DEFAULT_M_GLOBAL,
    N_cross_doc: int = DEFAULT_N_CROSS_DOC,
    sentence_col: str = "sentence_text",
    doc_position_col: str = "doc_position_percent",
    doc_id_col: str = "file_name",
) -> tuple:
    """
    Применяет флаги шума. Ядро — позиционная дедупликация по (sentence_norm + position_bucket_pct).
    raw_df содержит: sentence_norm, position_bucket, position_bucket_pct,
      is_position_header_footer, is_cross_doc_header_footer, is_global_repeat,
      is_probable_header_footer, is_noise, noise_reason.
    clean_df = raw_df[~raw_df["is_noise"]].
    """
    output_dir = output_dir or Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = df.copy()
    if sentence_col not in raw_df.columns:
        raw_df["sentence_norm"] = ""
        raw_df["position_bucket"] = "bucket_middle"
        raw_df["position_bucket_pct"] = 50.0
        raw_df["is_probable_header_footer"] = False
        raw_df["is_position_header_footer"] = False
        raw_df["is_cross_doc_header_footer"] = False
        raw_df["is_global_repeat"] = False
        raw_df["is_noise"] = False
        raw_df["noise_reason"] = ""
        return raw_df, raw_df.loc[~raw_df["is_noise"]].copy()

    # 1) Нормализация и корзины (именованная + 2% для дедупликации)
    raw_df["sentence_norm"] = raw_df[sentence_col].fillna("").apply(
        lambda t: normalize_sentence(str(t), nlp=nlp, max_lemmas=MAX_LEMMAS)
    )
    raw_df["position_bucket"] = raw_df[doc_position_col].apply(doc_position_to_bucket)
    raw_df["position_bucket_pct"] = raw_df[doc_position_col].apply(doc_position_bucket_pct)

    # 2) Позиционная дедупликация — ядро: (sentence_norm, position_bucket_pct)
    raw_df = add_position_deduplication_flags(
        raw_df,
        sentence_norm_col="sentence_norm",
        position_bucket_col="position_bucket",
        position_bucket_pct_col="position_bucket_pct",
        doc_id_col=doc_id_col,
        K=K_position,
        M=M_global,
        N_cross_doc=N_cross_doc,
    )

    # 3) Дополнительно: regex/частотные маркеры (колонтитулы по шаблонам)
    raw_df = flag_probable_header_footer(
        raw_df,
        repeat_threshold=repeat_threshold,
        sentence_col=sentence_col,
        doc_position_col=doc_position_col,
    )

    # 4) Итоговый флаг: сначала позиционная дедупликация, затем regex
    raw_df["is_noise"] = (
        raw_df["is_position_header_footer"].fillna(False)
        | raw_df["is_cross_doc_header_footer"].fillna(False)
        | raw_df["is_global_repeat"].fillna(False)
        | raw_df["is_probable_header_footer"].fillna(False)
    )
    reasons = []
    for _, row in raw_df.iterrows():
        r = []
        if row.get("is_position_header_footer"):
            r.append("position_repeat")
        if row.get("is_cross_doc_header_footer"):
            r.append("cross_doc_position")
        if row.get("is_global_repeat"):
            r.append("global_repeat")
        if row.get("is_probable_header_footer"):
            r.append("header_footer")
        reasons.append("; ".join(r) if r else "")
    raw_df["noise_reason"] = reasons
    clean_df = raw_df.loc[~raw_df["is_noise"]].copy()

    # 5) Диагностика: noise_position_summary.csv
    summary_rows = []
    doc_col = doc_id_col if doc_id_col in raw_df.columns else "doc_id"
    for (doc_key, sn, bucket), group in raw_df.groupby(
        [doc_col, "sentence_norm", "position_bucket"], dropna=False
    ):
        if not str(sn).strip():
            continue
        cnt = len(group)
        any_pos_hf = group["is_position_header_footer"].any()
        any_cross = group["is_cross_doc_header_footer"].any()
        any_global = group["is_global_repeat"].any()
        summary_rows.append({
            "sentence_norm": sn[:500],
            "position_bucket": bucket,
            "repeat_count": cnt,
            "doc_id": doc_key,
            "is_position_header_footer": any_pos_hf,
            "is_cross_doc_header_footer": any_cross,
            "is_global_repeat": any_global,
        })
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_dir / "noise_position_summary.csv", index=False, encoding="utf-8")

    # Лог
    n_pos = raw_df["is_position_header_footer"].sum()
    n_cross = raw_df["is_cross_doc_header_footer"].sum()
    n_glob = raw_df["is_global_repeat"].sum()
    n_prob = raw_df["is_probable_header_footer"].sum()
    n_noise = raw_df["is_noise"].sum()
    logger.info(
        "Noise filter (positional dedup core): is_position_header_footer=%s, is_cross_doc_header_footer=%s, "
        "is_global_repeat=%s, is_probable_header_footer=%s, is_noise=%s",
        int(n_pos), int(n_cross), int(n_glob), int(n_prob), int(n_noise),
    )
    print(
        f"   Шум: позиционная дедупликация {int(n_pos)}, кросс-док {int(n_cross)}, "
        f"глобальный повтор {int(n_glob)}, regex/частотность {int(n_prob)}, всего is_noise {int(n_noise)}"
    )

    # Топ повторяющихся (как раньше)
    if sentence_col in raw_df.columns:
        top_repeats = (
            raw_df[sentence_col].fillna("").astype(str).value_counts().head(200).reset_index()
        )
        top_repeats.columns = ["sentence_text", "repeat_count"]
        top_repeats.to_csv(output_dir / "noise_top_repeats.csv", index=False, encoding="utf-8")

    # 6) Строки с этнонимом, помеченные как шум (для верификации)
    ethnos_col = "ethnos_norm" if "ethnos_norm" in raw_df.columns else ("P" if "P" in raw_df.columns else None)
    noise_with_eth = raw_df.loc[raw_df["is_noise"]]
    if ethnos_col and not noise_with_eth.empty:
        has_eth = noise_with_eth[ethnos_col].notna() & (noise_with_eth[ethnos_col].astype(str).str.strip().str.len() > 0)
        noise_with_ethnonyms = noise_with_eth.loc[has_eth]
        if not noise_with_ethnonyms.empty:
            want = ["mention_id", "file_name", "doc_id", "sent_idx", "sentence_text", "sentence_norm", "position_bucket", "position_bucket_pct", "is_position_header_footer", "is_cross_doc_header_footer", "is_noise", "noise_reason", "ethnos_norm", "ethnos_raw"]
            cols = [c for c in want if c in noise_with_ethnonyms.columns]
            if "ethnos_norm" not in cols and "ethnos_raw" not in cols and "P" in noise_with_ethnonyms.columns:
                cols.append("P")
            noise_with_ethnonyms[cols].to_csv(output_dir / "noise_with_ethnonyms.csv", index=False, encoding="utf-8")

    return raw_df, clean_df
