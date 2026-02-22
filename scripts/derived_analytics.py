"""
Скрипт второго аналитического слоя: читает output/pipeline.db (или переданный путь),
не трогает corpus.db, пишет артефакты в output/derived/ и output/tables/.
Запуск: python scripts/derived_analytics.py [--pipeline output/pipeline.db] [--out output]
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Derived analytics: профили этносов, тесты, корреляции, кластеры")
    parser.add_argument("--pipeline", type=Path, default=PROJECT_ROOT / "output" / "pipeline.db", help="Путь к pipeline.db")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "output", help="Корневая папка вывода (derived/, tables/)")
    args = parser.parse_args()
    if not args.pipeline.exists():
        print(f"Файл не найден: {args.pipeline}")
        print("Сначала выполните main.py (без --no-derived) или укажите --pipeline путь к существующему pipeline.db.")
        sys.exit(1)
    from src.pipeline_db import load_pipeline
    pipeline_data = load_pipeline(args.pipeline)
    if not pipeline_data:
        print("Не удалось загрузить пайплайн (версия БД или формат).")
        sys.exit(1)
    from analysis.derived_analytics import run_derived_analytics
    out_derived = args.out / "derived"
    out_tables = args.out / "tables"
    result = run_derived_analytics(
        pipeline_data=pipeline_data,
        output_derived=out_derived,
        output_tables=out_tables,
    )
    if result:
        print("Готово. Артефакты: output/derived/*, output/tables/ethnic_profiles.*")
    else:
        print("Нет данных для профилей (пустой piro_clean).")
        sys.exit(1)


if __name__ == "__main__":
    main()
