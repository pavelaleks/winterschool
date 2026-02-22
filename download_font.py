"""Одноразовый скрипт: создать resources/fonts и скачать DejaVuSans.ttf."""
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent
FONTS_DIR = BASE / "resources" / "fonts"
URL = "https://github.com/prawnpdf/prawn-manual_builder/raw/master/data/fonts/DejaVuSans.ttf"
OUT = FONTS_DIR / "DejaVuSans.ttf"

def main():
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading DejaVuSans.ttf...")
    urllib.request.urlretrieve(URL, OUT)
    print("Saved to", OUT)

if __name__ == "__main__":
    main()
