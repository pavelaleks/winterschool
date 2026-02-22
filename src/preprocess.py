"""
Предобработка текстов: сегментация на предложения, токенизация, POS, dependency parsing.
Использует spaCy. Поддерживает параллельную обработку документов (n_jobs > 1).
"""

import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

_spacy_import_error = None
try:
    import spacy
except (ImportError, OSError) as _e:
    # OSError: часто на Windows при ошибке загрузки DLL PyTorch (c10.dll)
    spacy = None
    _spacy_import_error = _e
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Для воркеров при параллельной обработке
_worker_nlp = None


# Лимит длины текста для spaCy (по умолчанию 1e6). Увеличиваем для длинных травелогов.
SPACY_MAX_LENGTH = 2_500_000
# Размер чанка при разбиении очень длинных текстов (символов)
CHUNK_SIZE = 900_000

# Регулярка для разбиения на предложения без spaCy (fallback)
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*$", re.MULTILINE)


def _sentences_fallback(text: str) -> List[Dict[str, Any]]:
    """
    Fallback: разбиение на предложения по .!? и простые токены (без POS/deps).
    Используется, когда spaCy недоступен (например, ошибка DLL PyTorch на Windows).
    Структура предложения совместима с downstream (tokens, pos_tags, deps, token_objects).
    """
    text = (text or "").strip()
    if not text:
        return []
    # Разбить по концам предложений; оставшийся хвост без .!? — одно предложение
    parts = _SENTENCE_END_RE.split(text)
    # Объединить нечётные "хвосты" (между разделителями может быть пустая строка)
    cleaned = [p.strip() for p in parts if p.strip()]
    sentences = []
    char_offset = 0
    for sent_idx, sent_text in enumerate(cleaned):
        if not sent_text:
            continue
        tokens = sent_text.split()
        pos_tags = ["X"] * len(tokens)  # placeholder
        deps = ["dep"] * len(tokens)
        token_objects = [
            {
                "text": w,
                "lemma": w.lower(),
                "pos": "X",
                "dep": "dep",
                "head": "",
                "head_i": 0,
                "i": i,
            }
            for i, w in enumerate(tokens)
        ]
        start_char = char_offset
        end_char = char_offset + len(sent_text)
        char_offset = end_char + 1
        sentences.append({
            "sentence_index": sent_idx,
            "text": sent_text,
            "tokens": tokens,
            "pos_tags": pos_tags,
            "deps": deps,
            "token_objects": token_objects,
            "start_char": start_char,
            "end_char": end_char,
        })
    return sentences


def _get_nlp():
    """Загружает модель spaCy (en_core_web_sm) и увеличивает лимит длины текста."""
    if spacy is None:
        hint = (
            "Установите: pip install spacy && python -m spacy download en_core_web_sm. "
            "Если при импорте spacy появляется ошибка загрузки DLL (c10.dll на Windows), "
            "установите PyTorch для CPU: pip install torch --index-url https://download.pytorch.org/whl/cpu "
            "или установите Visual C++ Redistributable (https://aka.ms/vs/17/release/vc_redist.x64.exe)."
        )
        raise RuntimeError(f"spacy недоступен. {hint} Ошибка: {_spacy_import_error}") from _spacy_import_error
    try:
        nlp = spacy.load("en_core_web_sm")
        nlp.max_length = SPACY_MAX_LENGTH
        return nlp
    except OSError as e:
        raise OSError(
            "Модель en_core_web_sm не найдена. Выполните: python -m spacy download en_core_web_sm"
        ) from e


def _doc_to_sentences(doc: Any, char_offset: int = 0) -> List[Dict[str, Any]]:
    """Преобразует spaCy doc в список предложений (словари)."""
    sentences = []
    for sent_idx, sent in enumerate(doc.sents):
        tokens = []
        pos_tags = []
        deps = []
        token_objects = []
        for t in sent:
            tokens.append(t.text)
            pos_tags.append(t.pos_)
            deps.append(t.dep_)
            token_objects.append(
                {
                    "text": t.text,
                    "lemma": t.lemma_,
                    "pos": t.pos_,
                    "dep": t.dep_,
                    "head": t.head.text,
                    "head_i": t.head.i,
                    "i": t.i,
                }
            )
        sentences.append({
            "sentence_index": sent_idx,
            "text": sent.text,
            "tokens": tokens,
            "pos_tags": pos_tags,
            "deps": deps,
            "token_objects": token_objects,
            "start_char": char_offset + sent.start_char,
            "end_char": char_offset + sent.end_char,
        })
    return sentences


def _split_into_chunks(text: str, max_chars: int) -> List[str]:
    """Разбивает текст на куски не больше max_chars, по границам абзацев (\\n\\n)."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            sep = text.rfind("\n\n", start, end + 1)
            if sep > start:
                end = sep + 2
        chunks.append(text[start:end])
        start = end
    return chunks


def preprocess_document(
    text: str,
    nlp: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Обрабатывает один документ: предложения с токенами, POS, deps.
    Если текст длиннее лимита spaCy — обрабатывается по частям.
    Returns: список предложений; каждое — dict с keys:
      sentence_index, text, tokens, pos_tags, deps, token_objects (для доступа к head и т.д.)
    """
    if nlp is None:
        nlp = _get_nlp()
    max_len = getattr(nlp, "max_length", SPACY_MAX_LENGTH)
    if len(text) <= max_len:
        doc = nlp(text)
        return _doc_to_sentences(doc, 0)
    # Очень длинный текст: разбиваем на чанки
    chunks = _split_into_chunks(text, CHUNK_SIZE)
    all_sentences = []
    offset = 0
    for chunk in chunks:
        doc = nlp(chunk)
        sents = _doc_to_sentences(doc, offset)
        for i, s in enumerate(sents):
            s["sentence_index"] = len(all_sentences) + i
        all_sentences.extend(sents)
        offset += len(chunk)
    return all_sentences


def _init_worker() -> None:
    """Инициализация воркера: загрузка модели spaCy (один раз на процесс)."""
    global _worker_nlp
    _worker_nlp = _get_nlp()


def _process_one_doc(item: Tuple[str, str]) -> Dict[str, Any]:
    """Обрабатывает один документ (filename, text). Вызывается в дочернем процессе."""
    filename, text = item
    sentences = preprocess_document(text, nlp=_worker_nlp)
    return {
        "filename": filename,
        "sentences": sentences,
        "raw_text": text,
    }


def preprocess_corpus(
    documents: List[tuple],
    nlp: Optional[Any] = None,
    n_jobs: int = 1,
) -> List[Dict[str, Any]]:
    """
    Обрабатывает корпус: список (filename, text).
    n_jobs: 1 — последовательно; >1 — параллельно (столько процессов). Рекомендуется 2–4.
    Если spaCy недоступен (DLL/ImportError), используется fallback: разбиение по .!? и простые токены.
    Returns: список документов, каждый — dict с keys:
      filename, sentences (результат preprocess_document или fallback), raw_text.
    """
    use_fallback = spacy is None
    if use_fallback:
        # Без spaCy: простая сегментация и токены-заглушки (POS=X, dep=dep)
        corpus = []
        for filename, text in tqdm(documents, desc="Предобработка (fallback, без spaCy)", unit="файл"):
            sentences = _sentences_fallback(text)
            corpus.append({
                "filename": filename,
                "sentences": sentences,
                "raw_text": text,
            })
        return corpus

    n_workers = max(1, min(n_jobs, len(documents), os.cpu_count() or 4))
    if n_workers <= 1:
        # Последовательно
        if nlp is None:
            nlp = _get_nlp()
        corpus = []
        for filename, text in tqdm(documents, desc="Предобработка документов", unit="файл"):
            sentences = preprocess_document(text, nlp=nlp)
            corpus.append({
                "filename": filename,
                "sentences": sentences,
                "raw_text": text,
            })
        return corpus
    # Параллельно: каждый процесс загружает свою копию модели
    corpus = [None] * len(documents)
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as executor:
        future_to_idx = {executor.submit(_process_one_doc, item): i for i, item in enumerate(documents)}
        for future in tqdm(as_completed(future_to_idx), total=len(documents), desc="Предобработка документов", unit="файл"):
            i = future_to_idx[future]
            try:
                corpus[i] = future.result()
            except Exception as e:
                raise RuntimeError(f"Ошибка при обработке документа {documents[i][0]}: {e}") from e
    return corpus
