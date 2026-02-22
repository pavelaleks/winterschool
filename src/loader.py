"""
Загрузка текстов травелогов из data/texts/.
Поддержка выбора одного или нескольких документов (--documents) для анализа подкорпуса.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional


def get_texts_dir() -> Path:
    """Путь к каталогу с текстами относительно корня проекта."""
    base = Path(__file__).resolve().parent.parent
    return base / "data" / "texts"


def doc_matches(filename: str, selectors: Optional[List[str]]) -> bool:
    """
    Проверяет, подходит ли документ под список селекторов.
    selectors is None или пустой — подходит любой документ.
    Иначе: имя файла (без пути) сопоставляется с каждым селектором без учёта регистра:
    - если селектор похож на имя файла (содержит точку или заканчивается на .txt) — точное совпадение;
    - иначе — вхождение селектора в имя файла (например, "michie" подходит к "michie_1859_travel.txt").
    """
    if not selectors:
        return True
    fn_lower = (filename or "").lower()
    for sel in selectors:
        if not sel:
            continue
        s = sel.strip().lower()
        if not s:
            continue
        if "." in s or s.endswith(".txt"):
            if fn_lower == s or fn_lower.endswith(s):
                return True
        if s in fn_lower:
            return True
    return False


def load_texts(include_docs: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """
    Загружает .txt файлы из data/texts/.

    include_docs: если задан, загружаются только документы, подходящие под селекторы.
      Селектор — подстрока в имени файла (например, "michie") или точное имя ("michie_1859_travel.txt").
      Несколько селекторов — ИЛИ (достаточно совпадения с одним).

    Returns: список кортежей (имя_файла, содержимое_текста).
    """
    texts_dir = get_texts_dir()
    if not texts_dir.exists():
        texts_dir.mkdir(parents=True, exist_ok=True)
        return []

    result = []
    for path in sorted(texts_dir.glob("*.txt")):
        if not doc_matches(path.name, include_docs):
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            result.append((path.name, content))
        except Exception as e:
            raise RuntimeError(f"Ошибка чтения {path}: {e}") from e
    return result


def filter_corpus_by_docs(corpus: List[dict], include_docs: Optional[List[str]]) -> List[dict]:
    """
    Оставляет в корпусе только документы, чьё имя файла подходит под include_docs.
    corpus: список dict с ключом "filename".
    include_docs: как в load_texts; None или пустой — без фильтра (вернуть как есть).
    """
    if not include_docs:
        return corpus
    return [d for d in corpus if doc_matches(d.get("filename", ""), include_docs)]
