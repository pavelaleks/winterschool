"""
Evidence-first слой: corpus.db → evidence pack.
Читает output/corpus.db, извлекает все предложения с этнонимом, строит evidence dataset,
позиционная дедупликация (колонтитулы), сохраняет parquet, sample JSON и noise_top_repeats.csv.
Не меняет индексы, архитектуру пайплайна, не использует LLM.
"""

import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Корень проекта: на уровень выше tools/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
DERIVED_DIR = OUTPUT_DIR / "derived"

# Путь к corpus.db
CORPUS_DB = OUTPUT_DIR / "corpus.db"

# Минимум слов для "не index_like"
MIN_WORDS_INDEX_LIKE = 5
# Порог позиционной дедупликации: одно и то же sentence_text_norm в >= N разных bucket в доке → шум
POSITIONAL_REPEAT_BUCKETS = 3
# Размер bucket по sentence_id: bucket = sentence_id // 50
BUCKET_SIZE = 50
# Маркеры оглавления/индекса
INDEX_LIKE_PATTERN = re.compile(
    r"\b(chapter|index|vol\.|volume|part|book|section|contents|page|pp\.)\b",
    re.IGNORECASE,
)


def _norm_text(s: str) -> str:
    """lower + strip + unicode normalize."""
    if not s or not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s.strip().lower())
    return s.strip()


def _word_count(s: str) -> int:
    return len(s.split()) if isinstance(s, str) else 0


def _caps_ratio(s: str) -> float:
    if not s or not s.strip():
        return 0.0
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def _digits_ratio(s: str) -> float:
    if not s or not s.strip():
        return 0.0
    return sum(1 for c in s if c.isdigit()) / max(len(s), 1)


def _is_index_like(sentence_text_norm: str, sentence_text: str) -> bool:
    """Слишком короткая, преимущественно цифры/заглавные, или chapter/index/vol. и т.п."""
    if not sentence_text_norm.strip():
        return True
    if _word_count(sentence_text) < MIN_WORDS_INDEX_LIKE:
        return True
    if _caps_ratio(sentence_text) > 0.7:
        return True
    if _digits_ratio(sentence_text) > 0.4:
        return True
    if INDEX_LIKE_PATTERN.search(sentence_text_norm) or INDEX_LIKE_PATTERN.search(sentence_text):
        return True
    return False


def load_corpus(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Загружает корпус из corpus.db (тот же формат, что и src.corpus_db.load_corpus)."""
    db_path = Path(db_path or CORPUS_DB)
    if not db_path.exists():
        return []
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        docs = conn.execute("SELECT id, filename, raw_text FROM documents ORDER BY id").fetchall()
        if not docs:
            return []
        corpus = []
        for row in docs:
            doc_id = row["id"]
            filename = row["filename"] or ""
            raw_text = row["raw_text"] or ""
            sents = conn.execute(
                "SELECT sentence_index, text, start_char, end_char FROM sentences WHERE doc_id = ? ORDER BY sentence_index",
                (doc_id,),
            ).fetchall()
            sentences = []
            pos = 0
            for s in sents:
                text = s["text"] or ""
                start_char = s["start_char"] if s["start_char"] is not None else pos
                end_char = s["end_char"] if s["end_char"] is not None else (pos + len(text))
                pos = end_char
                sentences.append({
                    "sentence_index": s["sentence_index"],
                    "text": text,
                    "start_char": start_char,
                    "end_char": end_char,
                })
            corpus.append({
                "doc_id": doc_id,
                "filename": filename,
                "raw_text": raw_text,
                "sentences": sentences,
            })
        return corpus
    finally:
        conn.close()


def get_ethnonym_patterns() -> List[Tuple[str, re.Pattern]]:
    """Загружает ethnonyms.yml и строит regex-паттерны."""
    from src.ethnonym_extractor import load_ethnonyms, build_ethnonym_patterns
    ethnonyms = load_ethnonyms()
    return build_ethnonym_patterns(ethnonyms)


def extract_mentions_from_doc(
    doc: Dict,
    patterns: List[Tuple[str, re.Pattern]],
) -> List[Dict]:
    """Извлекает упоминания этнонимов в документе, маппит на sentence_index."""
    from src.ethnonym_extractor import extract_ethnonym_mentions, map_mentions_to_sentences
    raw_text = doc.get("raw_text", "")
    sentences = doc.get("sentences", [])
    if not raw_text or not sentences:
        return []
    mentions = extract_ethnonym_mentions(raw_text, patterns)
    # Нужны start_char/end_char у предложений для map_mentions_to_sentences
    mentions = map_mentions_to_sentences(mentions, sentences, doc_start_offset=0)
    return mentions


def context_prev_next(sentences: List[Dict], center_index: int, window: int = 2) -> Tuple[str, str]:
    """Возвращает (context_prev, context_next) — по window предложений до и после."""
    prev_sents = []
    next_sents = []
    for i in range(max(0, center_index - window), center_index):
        if i < len(sentences):
            prev_sents.append(sentences[i].get("text", ""))
    for i in range(center_index + 1, min(len(sentences), center_index + 1 + window)):
        next_sents.append(sentences[i].get("text", ""))
    return " ".join(prev_sents), " ".join(next_sents)


def build_evidence_rows(corpus: List[Dict]) -> List[Dict[str, Any]]:
    """Строит список строк evidence из корпуса (все предложения с этнонимом)."""
    patterns = get_ethnonym_patterns()
    rows = []
    for doc in corpus:
        doc_id = doc.get("doc_id", 0)
        doc_name = doc.get("filename", "")
        sentences = doc.get("sentences", [])
        mentions = extract_mentions_from_doc(doc, patterns)
        for m in mentions:
            sent_idx = m.get("sentence_index")
            if sent_idx is None or sent_idx >= len(sentences):
                continue
            sent = sentences[sent_idx]
            sentence_text = sent.get("text", "")
            sentence_text_norm = _norm_text(sentence_text)
            ctx_prev, ctx_next = context_prev_next(sentences, sent_idx, window=2)
            doc_position_bucket = sent_idx // BUCKET_SIZE
            source_pointer = f"{doc_name}#sent={sent_idx}"
            rows.append({
                "doc_id": doc_id,
                "doc_name": doc_name,
                "sentence_id": sent_idx,
                "sentence_text": sentence_text,
                "sentence_text_norm": sentence_text_norm,
                "context_prev": ctx_prev,
                "context_next": ctx_next,
                "ethnos_raw": m.get("match_text", ""),
                "ethnos_norm": m.get("ethnonym", ""),
                "R_label": "",
                "O_label": "",
                "is_noise": False,
                "noise_reason": "",
                "doc_position_bucket": doc_position_bucket,
                "source_pointer": source_pointer,
            })
    return rows


def apply_noise_flags(rows: List[Dict]) -> List[Dict]:
    """
    Позиционная дедупликация: (sentence_text_norm + doc_position_bucket) в доке;
    если одно и то же sentence_text_norm в >= 3 разных bucket → is_noise, noise_reason = positional_repeat_header_footer.
    Дополнительно: index_like (короткие, цифры/заглавные, chapter/index/vol.).
    """
    # Группируем по (doc_id, sentence_text_norm) и считаем число различных doc_position_bucket
    from collections import defaultdict
    doc_norm_to_buckets: Dict[Tuple[int, str], set] = defaultdict(set)
    for r in rows:
        key = (r["doc_id"], r["sentence_text_norm"])
        if r["sentence_text_norm"].strip():
            doc_norm_to_buckets[key].add(r["doc_position_bucket"])

    positional_noise = set()
    for (doc_id, sn), buckets in doc_norm_to_buckets.items():
        if len(buckets) >= POSITIONAL_REPEAT_BUCKETS:
            positional_noise.add((doc_id, sn))

    for r in rows:
        if (r["doc_id"], r["sentence_text_norm"]) in positional_noise:
            r["is_noise"] = True
            r["noise_reason"] = "positional_repeat_header_footer"
        if _is_index_like(r["sentence_text_norm"], r["sentence_text"]):
            r["is_noise"] = True
            r["noise_reason"] = (r["noise_reason"] + "; index_like").lstrip("; ")
    return rows


def build_noise_top_repeats(rows: List[Dict]) -> pd.DataFrame:
    """Агрегат для noise_top_repeats.csv: sentence_text_norm, count, doc_ids."""
    from collections import defaultdict
    agg: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "doc_ids": set()})
    for r in rows:
        sn = (r.get("sentence_text_norm") or "").strip()
        if not sn:
            continue
        agg[sn]["count"] += 1
        agg[sn]["doc_ids"].add(r.get("doc_name") or r.get("doc_id") or "")
    out = []
    for sn, v in sorted(agg.items(), key=lambda x: -x[1]["count"]):
        out.append({
            "sentence_text_norm": sn[:500],
            "count": v["count"],
            "doc_ids": "|".join(sorted(v["doc_ids"]))[:2000],
        })
    return pd.DataFrame(out)


def build_sample(
    rows: List[Dict],
    max_rows: int = 1000,
    max_context_len: int = 1000,
    only_clean: bool = True,
) -> List[Dict]:
    """Выборка: только не noise, равномерно по топ-этносам, укороченный context."""
    if only_clean:
        rows = [r for r in rows if not r.get("is_noise")]
    if not rows:
        return []
    # Топ этносов по количеству упоминаний
    from collections import Counter
    ethnos_counts = Counter(r.get("ethnos_norm") or "" for r in rows)
    top_ethnos = [e for e, _ in ethnos_counts.most_common(30) if e]
    # Равномерная выборка по топ-этносам
    by_ethnos: Dict[str, List[Dict]] = {e: [] for e in top_ethnos}
    other = []
    for r in rows:
        e = r.get("ethnos_norm") or ""
        if e in by_ethnos:
            by_ethnos[e].append(r)
        else:
            other.append(r)
    per_ethnos = max(1, max_rows // max(len(top_ethnos), 1))
    sample = []
    for e in top_ethnos:
        sample.extend(by_ethnos[e][:per_ethnos])
    sample.extend(other[: max_rows - len(sample)])
    sample = sample[:max_rows]
    # Укоротить context до max_context_len
    for r in sample:
        for key in ("context_prev", "context_next", "sentence_text"):
            if key in r and isinstance(r[key], str) and len(r[key]) > max_context_len:
                r[key] = r[key][:max_context_len] + "…"
    return sample


def main() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Evidence layer: чтение corpus.db...")
    corpus = load_corpus()
    if not corpus:
        print("  corpus.db не найден или пуст. Запустите пайплайн до шага сохранения корпуса.")
        return
    print(f"  Документов: {len(corpus)}, предложений: {sum(len(d.get('sentences', [])) for d in corpus)}")

    print("  Извлечение упоминаний этнонимов...")
    rows = build_evidence_rows(corpus)
    if not rows:
        print("  Нет предложений с этнонимом.")
        return
    print(f"  Упоминаний: {len(rows)}")

    print("  Позиционная дедупликация и маркеры шума...")
    rows = apply_noise_flags(rows)
    n_noise = sum(1 for r in rows if r.get("is_noise"))
    print(f"  Помечено как шум: {n_noise}")

    # DataFrame для parquet (все колонки уже в rows)
    df = pd.DataFrame(rows)
    parquet_path = DERIVED_DIR / "evidence_pack.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"  Сохранено: {parquet_path}")

    noise_df = build_noise_top_repeats(rows)
    noise_path = OUTPUT_DIR / "noise_top_repeats.csv"
    noise_df.to_csv(noise_path, index=False, encoding="utf-8")
    print(f"  Сохранено: {noise_path}")

    sample = build_sample(rows, max_rows=1000, max_context_len=1000, only_clean=True)
    sample_path = DERIVED_DIR / "evidence_pack_sample.json"
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=0)
    print(f"  Сохранено: {sample_path} ({len(sample)} строк)")
    print("Готово.")


if __name__ == "__main__":
    main()
