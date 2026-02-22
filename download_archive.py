"""
Скрипт загрузки травелогов с archive.org.
Запуск из корня проекта: python download_archive.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.archive_downloader import main

if __name__ == "__main__":
    main()
