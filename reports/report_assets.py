"""
Скачивание локальных ресурсов для офлайн-отчёта: jQuery, DataTables, Plotly.
Сохраняет в output/assets/ (или переданную папку).
"""

import urllib.request
from pathlib import Path
from typing import Optional

ASSETS = [
    ("https://code.jquery.com/jquery-3.7.1.min.js", "jquery-3.7.1.min.js"),
    ("https://cdn.plot.ly/plotly-2.27.0.min.js", "plotly-2.27.0.min.js"),
    ("https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js", "jquery.dataTables.min.js"),
    ("https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css", "jquery.dataTables.min.css"),
]


def ensure_report_assets(output_dir: Optional[Path] = None) -> Path:
    """
    Скачивает в output_dir/assets/ недостающие файлы. Возвращает путь к папке assets.
    """
    output_dir = output_dir or Path(__file__).resolve().parent.parent / "output"
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    for url, name in ASSETS:
        path = assets_dir / name
        if path.exists() and path.stat().st_size > 100:
            continue
        try:
            urllib.request.urlretrieve(url, path)
        except Exception:
            pass
    return assets_dir
