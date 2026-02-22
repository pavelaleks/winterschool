"""
Очистка текстов от типичных OCR-ошибок перед анализом.
- Unicode-нормализация, кавычки/дефисы
- Словарные замены из ocr_replacements.yml
- Схлопывание повторяющихся букв (boook -> book)
- Опционально: исправление по словарю (SymSpell), если установлен symspellpy
"""

import re
import unicodedata
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set

import yaml

RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"
OCR_REPLACEMENTS_PATH = RESOURCES_DIR / "ocr_replacements.yml"
ETHNONYMS_PATH = RESOURCES_DIR / "ethnonyms.yml"
SENTIMENT_PATH = RESOURCES_DIR / "sentiment_lexicon.yml"
SITUATION_PATH = RESOURCES_DIR / "situation_domains.yml"


def load_replacements() -> Dict[str, str]:
    """Загружает словарь замен wrong -> right из ocr_replacements.yml."""
    if not OCR_REPLACEMENTS_PATH.exists():
        return {}
    with open(OCR_REPLACEMENTS_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data:
        return {}
    if isinstance(data, dict):
        return {str(k).strip(): str(v).strip() for k, v in data.items() if k and v}
    return {}


def _apply_replacements(text: str, replacements: Dict[str, str]) -> str:
    """Применяет целословные замены с сохранением регистра первой буквы."""
    if not replacements:
        return text
    for wrong, right in replacements.items():
        if not wrong or wrong == right:
            continue
        pattern = re.compile(r"\b" + re.escape(wrong) + r"\b", re.IGNORECASE)

        def repl(m):
            s = m.group(0)
            if len(s) > 0 and s[0].isupper():
                return right[0].upper() + right[1:] if len(right) > 1 else right.upper()
            return right

        text = pattern.sub(repl, text)
    return text


def _normalize_whitespace(text: str) -> str:
    """Несколько пробелов/переносов — в один пробел, обрезка по краям."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n +", "\n", text)
    return text.strip()


def _fix_hyphenation(text: str) -> str:
    """Убирает перенос слова через дефис в конце строки (word-\\nword -> wordword)."""
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    return text


def _remove_control_chars(text: str) -> str:
    """Удаляет управляющие и непечатаемые символы, кроме \\n, \\t."""
    return "".join(c for c in text if c in "\n\t\r" or (c.isprintable() and ord(c) >= 32))


def _normalize_unicode(text: str) -> str:
    """Нормализация Unicode (NFKC): совмещение составных символов, замена визуальных дубликатов."""
    return unicodedata.normalize("NFKC", text)


def _normalize_quotes_and_hyphens(text: str) -> str:
    """Кавычки и дефисы разных типов — к обычным ASCII ' \" -."""
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2010", "-").replace("\u2011", "-")
    text = text.replace("\u2012", "-").replace("\u2013", "-").replace("\u2014", "-")
    return text


def _collapse_repeated_letters(text: str) -> str:
    """Повтор одной буквы 3+ раз подряд — в 2 (типичная ошибка OCR: savagggc -> savagc)."""
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def _get_known_words() -> Set[str]:
    """Собирает множество «правильных» слов из ресурсов проекта для опциональной проверки правописания."""
    known: Set[str] = set()
    for path in (ETHNONYMS_PATH, SENTIMENT_PATH, SITUATION_PATH):
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                continue
            for v in data.values():
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, str):
                            known.add(item.lower())
                elif isinstance(v, str):
                    known.add(v.lower())
        except Exception:
            pass
    if OCR_REPLACEMENTS_PATH.exists():
        try:
            with open(OCR_REPLACEMENTS_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, str):
                        known.add(v.lower())
        except Exception:
            pass
    return known


def _apply_symspell_if_available(text: str, known_words: Optional[Set[str]] = None) -> str:
    """
    Если установлен symspellpy: исправляет слова с одной опечаткой (редакционное расстояние 1),
    только если исправление есть в словаре. Иначе возвращает text без изменений.
    """
    try:
        from symspellpy import SymSpell
    except ImportError:
        return text
    if known_words is None:
        known_words = _get_known_words()
    if not known_words:
        return text
    sym = SymSpell(max_dictionary_edit_distance=1)
    for w in known_words:
        if w.isalpha() and len(w) > 1:
            sym.create_dictionary_entry(w, 1)
    words = re.findall(r"\b([a-zA-Z']+)\b", text)
    result = text
    for w in words:
        if not w or w.lower() in known_words:
            continue
        suggestions = sym.lookup(w.lower(), max_edit_distance=1, verbosity=1)
        if len(suggestions) == 1 and suggestions[0].term != w.lower():
            correct = suggestions[0].term
            if w[0].isupper():
                correct = correct[0].upper() + correct[1:]
            result = re.sub(r"\b" + re.escape(w) + r"\b", correct, result)
    return result


def clean_text(
    text: str,
    replacements: Optional[Dict[str, str]] = None,
    use_symspell: bool = False,
) -> str:
    """
    Полная очистка одного текста.
    use_symspell: если True и установлен symspellpy — дополнительная коррекция по словарю (этнонимы, лексиконы).
    """
    if replacements is None:
        replacements = load_replacements()
    text = _normalize_unicode(text)
    text = _normalize_quotes_and_hyphens(text)
    text = _remove_control_chars(text)
    text = _fix_hyphenation(text)
    text = _collapse_repeated_letters(text)
    text = _apply_replacements(text, replacements)
    if use_symspell:
        known = _get_known_words()
        text = _apply_symspell_if_available(text, known)
    text = _normalize_whitespace(text)
    return text


def clean_corpus(
    documents: List[Tuple[str, str]],
    replacements: Optional[Dict[str, str]] = None,
    show_progress: bool = True,
    use_symspell: bool = False,
) -> List[Tuple[str, str]]:
    """
    Очищает корпус: список (filename, text) -> список (filename, cleaned_text).
    use_symspell: дополнительная коррекция по словарю (нужен pip install symspellpy).
    """
    if replacements is None:
        replacements = load_replacements()
    try:
        from tqdm import tqdm
        iterator = tqdm(documents, desc="Очистка OCR", unit="файл") if show_progress else documents
    except ImportError:
        iterator = documents
    result = []
    for filename, text in iterator:
        result.append((filename, clean_text(text, replacements, use_symspell=use_symspell)))
    return result
