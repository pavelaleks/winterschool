"""
Проверка исходных данных: распределение R (репрезентация) по этносам в piro_clean.
Помогает понять, почему OI = 0 для некоторых этносов: OI = (negative + exotic) / total.
Запуск: из корня проекта: python tools/inspect_r_by_ethnos.py [ethnos1 ethnos2 ...]
По умолчанию: japanese, ostyak, tungus.
"""
from pathlib import Path
from collections import Counter
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_ETHNOS = ("japanese", "ostyak", "tungus")


def main():
    db_path = PROJECT_ROOT / "output" / "pipeline.db"
    if not db_path.exists():
        print(f"Файл не найден: {db_path}")
        print("Запустите пайплайн: python main.py ...")
        return 1

    try:
        from src.pipeline_db import load_pipeline
    except Exception as e:
        print(f"Ошибка импорта pipeline_db: {e}")
        return 1

    data = load_pipeline(db_path)
    if not data:
        print("Не удалось загрузить пайплайн.")
        return 1

    piro_clean = data.get("piro_clean") or []
    if not piro_clean:
        print("piro_clean пуст.")
        return 1

    ethnos_filter = sys.argv[1:] if len(sys.argv) > 1 else list(DEFAULT_ETHNOS)
    ethnos_set = set(e.lower() for e in ethnos_filter)

    # Собираем по этносу: счётчик R и примеры записей
    by_ethnos = {}
    for r in piro_clean:
        eth = (r.get("P") or r.get("ethnos_norm") or "").strip().lower()
        if not eth or (ethnos_set and eth not in ethnos_set):
            continue
        if eth not in by_ethnos:
            by_ethnos[eth] = {"R": Counter(), "records": []}
        rr = (r.get("R") or "neutral").strip().lower() or "neutral"
        by_ethnos[eth]["R"][rr] += 1
        # храним до 3 примеров на этнос для контекста
        if len(by_ethnos[eth]["records"]) < 3:
            by_ethnos[eth]["records"].append({
                "R": rr,
                "sentence_preview": (r.get("sentence_text") or "")[:120] + ("…" if len(r.get("sentence_text") or "") > 120 else ""),
            })

    print("Распределение R по выбранным этносам (piro_clean)")
    print("OI = (negative + exotic) / total  →  OI=0, если нет ни negative, ни exotic\n")
    for eth in sorted(by_ethnos.keys()):
        info = by_ethnos[eth]
        total = sum(info["R"].values())
        neg = info["R"].get("negative", 0)
        exo = info["R"].get("exotic", 0)
        oi = (neg + exo) / total if total else 0.0
        print(f"  {eth}: total={total}, R={dict(info['R'])}, raw_OI={oi:.4f}")
        if neg == 0 and exo == 0:
            print(f"       → OI=0: в выборке нет меток negative/exotic (все neutral/positive/uncertain).")
        for ex in info["records"][:2]:
            print(f"       Пример R={ex['R']}: {ex['sentence_preview']}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
