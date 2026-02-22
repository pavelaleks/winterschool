"""
Загрузка англоязычных травелогов о Сибири с archive.org (1850–1917).
Поиск, дедупликация, подтверждение, скачивание в data/texts/, Excel-библиография.
"""

import logging
import re
import time
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import requests
from tqdm import tqdm

# Лимит года для репрезентативной базы
YEAR_MIN = 1850
YEAR_MAX = 1917
TEXTS_DIR = Path(__file__).resolve().parent.parent / "data" / "texts"
LOG_DIR = Path(__file__).resolve().parent.parent / "output" / "logs"
SEARCH_URL = "https://archive.org/advancedsearch.php"
METADATA_URL = "https://archive.org/metadata/{identifier}"
DOWNLOAD_BASE = "https://archive.org/download/{identifier}/{filename}"
# Пагинация
ROWS_PER_PAGE = 100
MAX_PAGES_PER_QUERY = 20


def _setup_logging(log_path: Optional[Path] = None) -> logging.Logger:
    """Логи в файл и в консоль."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = log_path or LOG_DIR / "archive_download.log"
    log = logging.getLogger("archive_downloader")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(ch)
    return log


def get_search_queries() -> List[str]:
    """
    Запросы для репрезентативной базы англоязычных травелогов о Сибири (1850–1917).
    """
    return [
        "Siberia travel",
        "Siberian travel",
        "travel Siberia",
        "travelogue Siberia",
        "journey Siberia",
        "expedition Siberia",
        "voyage Siberia",
        "narrative Siberia",
        "travelling Siberia",
        "Russia Siberia travel",
        "Siberia description travel",
        "Siberian journey narrative",
        "Siberia expedition narrative",
    ]


def _normalize_for_dedup(s: str) -> str:
    """Нормализация строки для сравнения (латиница, нижний регистр, без лишних пробелов)."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[^a-z0-9\s]", "", s.lower())
    return " ".join(s.split())


def _normalize_author(creator: Any) -> str:
    if isinstance(creator, list):
        creator = creator[0] if creator else ""
    return (creator or "").strip()


def _normalize_title(title: Any) -> str:
    if isinstance(title, list):
        title = title[0] if title else ""
    return (title or "").strip()


def _normalize_year(year: Any) -> Optional[int]:
    if year is None:
        return None
    if isinstance(year, list):
        year = year[0] if year else None
    if year is None:
        return None
    try:
        y = int(re.sub(r"\D", "", str(year))[:4] or "0") or None
        return y if y else None
    except Exception:
        return None


# Паттерны для выявления тома/части в названии и описании (чтобы не считать дубликатом)
VOLUME_PATTERNS = [
    re.compile(r"\bvol\.?\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\bvolume\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\bpart\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\bpart\s+(i+v?|iv|v?i{1,3})\b", re.IGNORECASE),
    re.compile(r"\btome\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\bbook\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\bv\.?\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\bpt\.?\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\b(second|third|fourth|first)\s+volume\b", re.IGNORECASE),
    re.compile(r"\bvolume\s+(one|two|three|four)\b", re.IGNORECASE),
]
VOLUME_WORD_TO_NUM = {"first": "1", "second": "2", "third": "3", "fourth": "4", "one": "1", "two": "2", "three": "3", "four": "4"}


def extract_volume_signature(title: str, description: Optional[str] = None) -> str:
    """
    Извлекает указание на том/часть из названия и описания.
    Возвращает строку вида "vol2", "part1", "" — если пусто, считаем одной книгой (возможный дубликат).
    """
    text = (title or "") + " " + (description or "")
    text = (text or "").lower()
    for pat in VOLUME_PATTERNS:
        m = pat.search(text)
        if m:
            g = m.group(1).lower()
            num = VOLUME_WORD_TO_NUM.get(g) or re.sub(r"\D", "", g) or g
            if "vol" in pat.pattern or "volume" in pat.pattern or "v\\.?" in pat.pattern:
                return f"vol{num}"
            if "part" in pat.pattern or "pt" in pat.pattern:
                return f"part{num}"
            if "book" in pat.pattern:
                return f"book{num}"
            if "tome" in pat.pattern:
                return f"tome{num}"
            return f"vol{num}"
    return ""


def extract_surname(creator: str) -> str:
    """Фамилия автора: при 'Фамилия, Имя' — первая часть, иначе последнее слово."""
    if not creator or not creator.strip():
        return ""
    creator = creator.strip()
    if "," in creator:
        return creator.split(",")[0].strip()
    parts = creator.split()
    return parts[-1] if parts else ""


def title_start_normalized(title: str, num_words: int = 4) -> str:
    """Первые num_words слов названия, нормализованные для сравнения (без знаков, нижний регистр)."""
    n = _normalize_for_dedup(title or "")
    words = n.split()[:num_words]
    return " ".join(words) if words else ""


def search_archive(
    query: str,
    year_min: int = YEAR_MIN,
    year_max: int = YEAR_MAX,
    log: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    Поиск на archive.org (mediatype:texts). Год фильтруется по полю year в ответе.
    """
    log = log or logging.getLogger("archive_downloader")
    all_results = []
    page = 0
    while page < MAX_PAGES_PER_QUERY:
        params = {
            "q": f"({query}) AND mediatype:texts",
            "fl[]": ["identifier", "title", "creator", "year", "publisher", "language"],
            "output": "json",
            "rows": ROWS_PER_PAGE,
            "page": page,
        }
        try:
            r = requests.get(SEARCH_URL, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log.warning("Search request failed for %s: %s", query, e)
            break
        response = data.get("response", {})
        docs = response.get("docs", [])
        if not docs:
            break
        for doc in docs:
            year = _normalize_year(doc.get("year"))
            if year is not None and (year < year_min or year > year_max):
                continue
            lang = doc.get("language") or []
            if isinstance(lang, str):
                lang = [lang]
            if lang:
                lang_str = " ".join(str(x).lower() for x in lang)
                if "eng" not in lang_str and "english" not in lang_str:
                    continue
            all_results.append({
                "identifier": doc.get("identifier", ""),
                "title": _normalize_title(doc.get("title")),
                "creator": _normalize_author(doc.get("creator")),
                "year": year,
                "publisher": doc.get("publisher") or "",
                "language": lang,
            })
        if len(docs) < ROWS_PER_PAGE:
            break
        page += 1
        time.sleep(0.5)
    return all_results


def fetch_item_metadata(identifier: str, log: Optional[logging.Logger] = None) -> Optional[Dict[str, Any]]:
    """Получение полных метаданных и списка файлов по identifier."""
    log = log or logging.getLogger("archive_downloader")
    try:
        r = requests.get(METADATA_URL.format(identifier=identifier), timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.debug("Metadata failed %s: %s", identifier, e)
        return None


def find_txt_file(metadata: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """
    Ищет в item файл в формате Plain Text (.txt).
    Returns: (filename, download_url) или None.
    """
    identifier = metadata.get("metadata", {}).get("identifier") or metadata.get("identifier", "")
    if not identifier:
        return None
    files = metadata.get("files", [])
    candidates = []
    for f in files:
        name = f.get("name", "")
        fmt = (f.get("format") or "").lower()
        if name.endswith(".txt") or "plain text" in fmt or fmt == "plain text":
            candidates.append((name, DOWNLOAD_BASE.format(identifier=identifier, filename=name)))
    if not candidates:
        return None
    prefer = [c for c in candidates if not c[0].endswith("_meta.txt")]
    return (prefer[0] if prefer else candidates[0])


def get_main_page_url(identifier: str) -> str:
    """Ссылка на основную страницу книги на archive.org (не на txt)."""
    return f"https://archive.org/details/{identifier}"


def deduplicate_items(
    items: List[Dict[str, Any]],
    log: Optional[logging.Logger] = None,
    title_words_for_dedup: int = 4,
) -> List[Dict[str, Any]]:
    """
    Дедупликация с учётом томов:
    - Если в метаданных есть указание на том/часть (vol, part, book и т.д.) — разные тома считаются разными книгами, все сохраняем.
    - Если указаний на том нет — дубликат определяем по: одна фамилия автора + совпадение первых 3–4 слов названия + год. Оставляем одну запись (приоритет — наличие txt_url).
    """
    log = log or logging.getLogger("archive_downloader")
    key_to_items: Dict[tuple, List[Dict[str, Any]]] = {}
    for it in items:
        surname = _normalize_for_dedup(extract_surname(it.get("creator") or ""))
        title = it.get("title") or ""
        year = it.get("year")
        volume_sig = (it.get("volume_sig") or "").strip()
        if volume_sig:
            key = (surname, title_start_normalized(title, 10), year, volume_sig)
        else:
            key = (surname, title_start_normalized(title, title_words_for_dedup), year, "")
        key_to_items.setdefault(key, []).append(it)
    result = []
    for key, group in key_to_items.items():
        with_txt = [x for x in group if x.get("txt_url")]
        chosen = with_txt[0] if with_txt else group[0]
        result.append(chosen)
        if len(group) > 1:
            log.debug(
                "Dedup: kept %s, dropped %d duplicates (key: surname + title_start + year + volume)",
                chosen.get("identifier"),
                len(group) - 1,
            )
    return result


def sanitize_filename(name: str, max_len: int = 80) -> str:
    """Имя файла: латиница, цифры, подчёркивание; длина ограничена."""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if c.isalnum() or c in " _-")
    name = re.sub(r"[\s\-]+", "_", name).strip("_")
    return name[:max_len] or "unknown"


def run_search_and_collect(
    year_min: int = YEAR_MIN,
    year_max: int = YEAR_MAX,
    log: Optional[logging.Logger] = None,
) -> Tuple[List[Dict[str, Any]], logging.Logger]:
    """
    Запуск всех поисковых запросов, сбор результатов, проверка наличия .txt, дедупликация.
    Каждый item: identifier, title, creator, year, publisher, page_url, txt_url, txt_filename.
    """
    log = log or _setup_logging()
    queries = get_search_queries()
    seen_ids = set()
    raw_items = []
    for q in tqdm(queries, desc="Поиск по запросам", unit="запрос"):
        hits = search_archive(q, year_min=year_min, year_max=year_max, log=log)
        for h in hits:
            if h["identifier"] in seen_ids:
                continue
            seen_ids.add(h["identifier"])
            raw_items.append(h)
        time.sleep(0.3)
    log.info("Всего найдено записей (без дедупликации по контенту): %s", len(raw_items))

    # Проверяем наличие .txt у каждого item
    items_with_txt = []
    for it in tqdm(raw_items, desc="Проверка наличия TXT", unit="item"):
        meta = fetch_item_metadata(it["identifier"], log=log)
        if not meta:
            continue
        md = meta.get("metadata", {})
        if md:
            it["title"] = _normalize_title(md.get("title") or it.get("title"))
            it["creator"] = _normalize_author(md.get("creator") or it.get("creator"))
            it["year"] = _normalize_year(md.get("year")) or it.get("year")
            pub = md.get("publisher")
            it["publisher"] = _normalize_title(pub) if pub else (it.get("publisher") or "")
        desc = md.get("description") if md else None
        if isinstance(desc, list):
            desc = " ".join(str(x) for x in desc) if desc else ""
        it["volume_sig"] = extract_volume_signature(it.get("title") or "", desc)
        it["page_url"] = get_main_page_url(it["identifier"])
        found = find_txt_file(meta)
        if found:
            it["txt_filename"] = found[0]
            it["txt_url"] = found[1]
            items_with_txt.append(it)
        time.sleep(0.2)
    log.info("Записей с доступным TXT: %s", len(items_with_txt))

    deduped = deduplicate_items(items_with_txt, log=log, title_words_for_dedup=4)
    n_volumes = sum(1 for i in deduped if i.get("volume_sig"))
    log.info(
        "После дедупликации (фамилия + первые 4 слова названия + год; тома учтены): %s (из них с указанием тома: %s)",
        len(deduped),
        n_volumes,
    )
    return deduped, log


def download_one(
    item: Dict[str, Any],
    dest_dir: Path,
    log: logging.Logger,
) -> bool:
    """Скачивает один txt в dest_dir с именем author_year_title.txt (при томах — author_year_title_vol2.txt)."""
    url = item.get("txt_url")
    if not url:
        return False
    author = sanitize_filename(item.get("creator") or "unknown")
    year = item.get("year") or 0
    title = sanitize_filename(item.get("title") or "unknown")
    vol = (item.get("volume_sig") or "").strip()
    if vol:
        fname = f"{author}_{year}_{title}_{vol}.txt"
    else:
        fname = f"{author}_{year}_{title}.txt"
    if len(fname) > 200:
        fname = f"{author}_{year}_{title[:150]}{'_' + vol if vol else ''}.txt"
    path = dest_dir / fname
    if path.exists():
        log.debug("Already exists: %s", path.name)
        return True
    try:
        r = requests.get(url, timeout=120, stream=True)
        r.raise_for_status()
        path.write_bytes(r.content)
        log.info("Downloaded: %s -> %s", item.get("identifier"), path.name)
        return True
    except Exception as e:
        log.warning("Download failed %s: %s", item.get("identifier"), e)
        return False


def get_output_filename(item: Dict[str, Any]) -> str:
    """Имя файла, под которым сохранён или будет сохранён текст (как в download_one)."""
    author = sanitize_filename(item.get("creator") or "unknown")
    year = item.get("year") or 0
    title = sanitize_filename(item.get("title") or "unknown")
    vol = (item.get("volume_sig") or "").strip()
    if vol:
        fname = f"{author}_{year}_{title}_{vol}.txt"
    else:
        fname = f"{author}_{year}_{title}.txt"
    if len(fname) > 200:
        fname = f"{author}_{year}_{title[:150]}{'_' + vol if vol else ''}.txt"
    return fname


def build_bibliography_table(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Таблица для Excel: автор, название, год, издательство, ссылка на страницу, имя файла."""
    return [
        {
            "author": item.get("creator") or "",
            "title": item.get("title") or "",
            "year": str(item.get("year") or ""),
            "publisher": item.get("publisher") or "",
            "url": item.get("page_url") or "",
            "filename": get_output_filename(item),
        }
        for item in items
    ]


def save_bibliography_excel(
    items: List[Dict[str, Any]],
    output_path: Path,
    log: Optional[logging.Logger] = None,
) -> None:
    """Сохраняет Excel-таблицу библиографии."""
    import pandas as pd
    log = log or logging.getLogger("archive_downloader")
    table = build_bibliography_table(items)
    df = pd.DataFrame(table)
    df.columns = ["Автор", "Название книги", "Год издания", "Издательство", "Ссылка на страницу", "Имя файла"]
    df.to_excel(output_path, index=False, engine="openpyxl")
    log.info("Bibliography saved: %s", output_path)


def interactive_download(
    items: List[Dict[str, Any]],
    dest_dir: Optional[Path] = None,
    excel_path: Optional[Path] = None,
    log: Optional[logging.Logger] = None,
) -> None:
    """
    Показывает статистику, запрашивает подтверждение, скачивает файлы и сохраняет Excel.
    """
    dest_dir = dest_dir or TEXTS_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    excel_path = excel_path or (Path(__file__).resolve().parent.parent / "output" / "archive_bibliography.xlsx")
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    log = log or _setup_logging()

    total = len(items)
    with_txt = sum(1 for i in items if i.get("txt_url"))
    log.info("Всего найдено текстов: %s", total)
    log.info("С доступным TXT (недублирующихся после дедупликации): %s", with_txt)

    n_volumes = sum(1 for i in items if i.get("volume_sig"))
    print("\n--- Статистика ---")
    print(f"Всего найдено записей (с TXT): {total}")
    print(f"После дедупликации (фамилия + первые 4 слова названия + год; разные тома сохранены): {total}")
    print(f"Из них с указанием на том/часть в метаданных: {n_volumes}")
    print(f"Будет загружено в: {dest_dir}")
    print(f"Таблица-библиография: {excel_path}")
    try:
        answer = input("\nЗагружать эти тексты? (y/n): ").strip().lower()
    except EOFError:
        answer = "n"
    if answer != "y" and answer != "yes":
        print("Загрузка отменена.")
        log.info("Download cancelled by user")
        return

    save_bibliography_excel(items, excel_path, log=log)
    for it in tqdm(items, desc="Скачивание", unit="файл"):
        download_one(it, dest_dir, log=log)
        time.sleep(0.3)
    print(f"Готово. Файлы в {dest_dir}, библиография в {excel_path}")


def main():
    """Точка входа: поиск -> статистика -> вопрос -> скачивание и Excel."""
    log = _setup_logging()
    log.info("Start archive.org download (Siberian travelogues, %s-%s)", YEAR_MIN, YEAR_MAX)
    items, log = run_search_and_collect(year_min=YEAR_MIN, year_max=YEAR_MAX, log=log)
    if not items:
        print("По запросам ничего не найдено (или нет записей с TXT в указанном периоде).")
        log.warning("No items with TXT found")
        return
    interactive_download(items, dest_dir=TEXTS_DIR, log=log)


if __name__ == "__main__":
    main()
