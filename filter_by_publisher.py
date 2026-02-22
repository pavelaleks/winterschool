"""
Скрипт фильтрации скачанных текстов по издательству.
Копирует в data/texts_london_paris_westminster только те файлы, у которых в таблице
библиографии в графе «Издательство» указаны London, Paris или Westminster, либо ячейка пустая.

Запуск из корня проекта: python filter_by_publisher.py
"""

import shutil
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

# Папка со скачанными текстами и папка для отфильтрованных (рядом с ней)
TEXTS_DIR = (PROJECT_ROOT / "data" / "texts").resolve()
OUT_DIR = (PROJECT_ROOT / "data" / "texts_london_paris_westminster").resolve()
BIBLIOGRAPHY_PATH = (PROJECT_ROOT / "output" / "archive_bibliography.xlsx").resolve()

ALLOWED_PUBLISHER_KEYWORDS = ("london", "paris", "westminster")


def publisher_matches(publisher) -> bool:
    """True, если издательство пустое или содержит London, Paris или Westminster."""
    if publisher is None or (isinstance(publisher, float) and pd.isna(publisher)):
        return True
    s = str(publisher).strip().lower()
    if not s:
        return True
    return any(kw in s for kw in ALLOWED_PUBLISHER_KEYWORDS)


def sanitize(s: str) -> str:
    """Латиница/цифры/подчёркивание для сравнения имён файлов."""
    if not s:
        return ""
    s = str(s).strip().lower()
    return "".join(c for c in s if c.isalnum() or c in " _-").replace(" ", "_")


def extract_surname(creator: str) -> str:
    if not creator or not str(creator).strip():
        return ""
    creator = str(creator).strip()
    if "," in creator:
        return sanitize(creator.split(",")[0])
    parts = str(creator).split()
    return sanitize(parts[-1]) if parts else ""


def find_file_for_row(row, col_author, col_year, col_title, col_file, all_txt_files: list) -> Optional[Path]:
    """Находит файл в списке all_txt_files, соответствующий строке таблицы."""
    fname = None
    if col_file is not None:
        try:
            v = row[col_file]
            if v is not None and not (isinstance(v, float) and pd.isna(v)) and str(v).strip():
                fname = str(v).strip()
                if not fname.endswith(".txt"):
                    fname += ".txt"
        except Exception:
            pass
    if fname:
        for p in all_txt_files:
            if p.name == fname or p.name == fname.strip():
                return p
        return TEXTS_DIR / fname if (TEXTS_DIR / fname).exists() else None

    author = str(row.get(col_author, "") or "").strip()
    year_raw = row.get(col_year, "")
    year = str(year_raw).strip().replace(".0", "").split(".")[0] if year_raw is not None else ""
    title = str(row.get(col_title, "") or "").strip()
    surname = extract_surname(author)
    title_start = sanitize(title)[:60] if title else ""

    candidates = []
    for p in all_txt_files:
        stem = p.stem.lower().replace("-", "_")
        stem_flat = stem.replace("_", "")
        if not surname and not year:
            continue
        if year and year not in stem:
            continue
        if surname and surname.lower() not in stem:
            continue
        if title_start:
            title_flat = title_start[:40].replace("_", "")
            if title_flat and title_flat not in stem_flat:
                continue
        candidates.append(p)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return candidates[0]
    return None


def main():
    if not BIBLIOGRAPHY_PATH.exists():
        print(f"Таблица библиографии не найдена: {BIBLIOGRAPHY_PATH}")
        print("Сначала выполните download_archive.py и сохраните Excel.")
        return
    if not TEXTS_DIR.exists():
        print(f"Папка с текстами не найдена: {TEXTS_DIR}")
        return

    df = pd.read_excel(BIBLIOGRAPHY_PATH, engine="openpyxl")
    cols = list(df.columns)
    col_pub = None
    col_file = None
    col_author = None
    col_year = None
    col_title = None
    for i, c in enumerate(cols):
        cn = str(c).lower().strip()
        if "издатель" in cn or cn == "publisher":
            col_pub = c
        if "имя файла" in cn or cn == "filename" or "файл" in cn:
            col_file = c
        if "автор" in cn or cn == "author":
            col_author = c
        if "год" in cn or cn == "year":
            col_year = c
        if "название" in cn or "title" in cn:
            col_title = c
    if col_pub is None and len(cols) > 3:
        col_pub = cols[3]
    if col_file is None and len(cols) > 5:
        col_file = cols[5]
    if col_author is None and len(cols) > 0:
        col_author = cols[0]
    if col_year is None and len(cols) > 2:
        col_year = cols[2]
    if col_title is None and len(cols) > 1:
        col_title = cols[1]

    filtered = df[df[col_pub].apply(publisher_matches)].copy()
    if filtered.empty:
        print("Нет записей с издательством London, Paris, Westminster или пустым.")
        return

    all_txt = list(TEXTS_DIR.glob("*.txt"))
    print(f"В папке {TEXTS_DIR.name} найдено .txt файлов: {len(all_txt)}")
    print(f"Записей после фильтра по издательству: {len(filtered)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    copied_paths = set()
    missing = []
    for _, row in filtered.iterrows():
        src = find_file_for_row(row, col_author, col_year, col_title, col_file, all_txt)
        if src is None or not src.exists():
            missing.append(str(row.get(col_author, ""))[:40])
            continue
        src = src.resolve()
        if src in copied_paths:
            continue
        dest = OUT_DIR / src.name
        try:
            shutil.copy2(src, dest)
            copied_paths.add(src)
            copied += 1
        except Exception as e:
            print(f"Ошибка копирования {src.name}: {e}")

    print(f"Скопировано в {OUT_DIR}: {copied} файлов")
    if missing:
        print(f"Не удалось сопоставить с файлом: {len(missing)} записей")


if __name__ == "__main__":
    main()
