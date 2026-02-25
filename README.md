# Исследовательская платформа: анализ травелогов о Сибири (DH, ориентализм)

**Версия:** 1.0  
**Автор:** П.В. Алексеев — [pavel.alekseev.gasu@gmail.com](mailto:pavel.alekseev.gasu@gmail.com)

Цифровой гуманитарный пайплайн для анализа англоязычных травелогов о Сибири: извлечение упоминаний этнонимов, классификация репрезентации (R) и ситуации (O), эссенциализация, keyness, сети взаимодействий, производные индексы ориентализации. Результат — интерактивный HTML-отчёт с опциональной интерпретацией через LLM (DeepSeek) и учётом токенов.

---

## Быстрый старт

```bash
# 1. Установка
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Положить тексты травелогов в data/texts/*.txt

# 3. Один из двух режимов:

# Режим A — полный академический отчёт с LLM (мемо, синтез, токены в отчёте)
python main.py --run-llm
# или с кэшем корпуса:
python main.py --from-cache --run-llm

# Режим B — только аналитика по уже посчитанной базе (без пересчёта, без LLM)
python main.py --from-db
```

Для режима A задайте API-ключ DeepSeek: переменная окружения `DEEPSEEK_API_KEY` или файл `.env` в корне проекта с строкой `DEEPSEEK_API_KEY=ваш_ключ`.

Главный артефакт: **`output/report.html`**.

---

## Режимы scientific report (LLM)

| Режим | Команда | Скорость/стоимость | Качество scientific_report |
|------|---------|--------------------|----------------------------|
| Один вызов (базовый) | `python main.py --run-llm` | Быстрее, дешевле | Хороший общий текст, но возможна поверхностность отдельных секций |
| Поэтапный (staged) | `python main.py --run-llm --scientific-report-staged` | Медленнее, дороже (несколько LLM-вызовов) | Обычно глубже и аккуратнее по секциям (corpus/distributions, keyness/essentialization, networks/indices, profiles/conclusion) |
| Полный прогон + staged scientific report | `python main.py --full --run-llm --scientific-report-staged` | Самый долгий и дорогой режим | Максимально полный набор артефактов + более глубокий scientific_report по секциям |
| Только scientific report из готовых данных | `python main.py --scientific-report-only --run-llm` | Быстро (без полного пересчёта пайплайна) | Обновляет только scientific report по текущим данным из `pipeline.db` и `output/derived` |
| Только scientific report, staged | `python main.py --scientific-report-only --run-llm --scientific-report-staged` | Чуть дольше, но без полного прогона | Лучший баланс для доработки текста перед публикацией |

**Когда использовать staged:** перед «почти финальной» версией отчёта/статьи, когда важна глубина интерпретации и согласованность секций больше, чем экономия токенов.

---

## Содержание

1. [Описание проекта](#1-описание-проекта)
2. [Требования и установка](#2-требования-и-установка)
3. [Структура проекта](#3-структура-проекта)
4. [Входные и выходные данные](#4-входные-и-выходные-данные)
5. [Пайплайн (шаги)](#5-пайплайн-шаги)
6. [Режимы запуска](#6-режимы-запуска)
7. [Модули и компоненты](#7-модули-и-компоненты)
8. [LLM (DeepSeek) и учёт токенов](#8-llm-deepseek-и-учёт-токенов)
9. [Выходные артефакты](#9-выходные-артефакты)
10. [Устранение неполадок](#10-устранение-неполадок)

---

## 1. Описание проекта

**dh_orientalism_project** — исследовательская платформа для цифрового анализа травелогов о Сибири в контексте исследований ориентализма (Э. Саид) и репрезентации «других». Пайплайн:

- загружает тексты из `data/texts/`;
- выполняет очистку OCR, предобработку (spaCy), извлечение упоминаний этнонимов;
- классифицирует тип репрезентации (**R**: negative, exotic, positive, neutral, uncertain) и тип ситуации (**O**) по лексиконам;
- строит PIRO-разметку (эпитеты, R, O, эссенциализация);
- считает нормированные частоты, **keyness** (G2), сети **co-mention** и **interaction**, производные индексы (**OI**, **AS**, **ED**, **EPS**);
- при флаге `--run-llm` вызывает DeepSeek для мемо по блокам, межтабличного синтеза и обзора для PDF, с записью расхода токенов.

Итог: интерактивный **HTML-отчёт** (`output/report.html`), при необходимости — краткий PDF и Excel-экспорты.

---

## 2. Требования и установка

**Требования:**

- Python 3.10+
- Зависимости из `requirements.txt` (см. ниже)
- Модель spaCy для английского: **en_core_web_sm**

**Установка:**

```bash
cd dh_orientalism_project
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Опционально (для LLM):** в корне проекта создайте файл `.env` или задайте переменную окружения:

```
DEEPSEEK_API_KEY=ваш_ключ
```

**Основные зависимости (requirements.txt):**

- spacy, PyYAML, pandas, numpy, matplotlib, seaborn, networkx, tqdm  
- reportlab, openai, scipy, requests, flask, flask-cors, openpyxl  
- umap-learn, sentence-transformers, pyarrow, scikit-learn  
- опционально: wordcloud, symspellpy, hdbscan  

---

## 3. Структура проекта

```
dh_orientalism_project/
├── main.py                 # Точка входа, пайплайн (шаги 1–17)
├── requirements.txt
├── sanity_checks.py        # Пост-прогонные проверки
├── README.md
│
├── data/
│   └── texts/              # Входные тексты травелогов (*.txt)
│
├── output/                 # Создаётся при запуске
│   ├── report.html         # Главный интерактивный отчёт
│   ├── pipeline.db         # Кэш пайплайна (для --from-db)
│   ├── corpus.db           # Кэш корпуса (для --from-cache)
│   ├── llm_memos/          # Мемо по блокам и synthesis.json
│   ├── metadata/           # llm_token_count.json, llm_mode.json
│   ├── tables/             # Таблицы keyness и др.
│   ├── figures/            # Графики, сети, UMAP
│   ├── assets/             # jQuery, Plotly, DataTables для отчёта
│   └── logs/               # llm_calls.log и др.
│
├── src/                    # Ядро пайплайна
│   ├── loader.py           # Загрузка текстов из data/texts/
│   ├── ocr_cleaner.py      # Очистка OCR (опционально SymSpell)
│   ├── preprocess.py       # spaCy: предложения, токены, POS, deps
│   ├── corpus_db.py        # Кэш корпуса (corpus.db)
│   ├── pipeline_db.py      # Кэш пайплайна (pipeline.db)
│   ├── db_introspect.py    # Таблица упоминаний (mentions_df)
│   ├── noise_filter.py     # Фильтр шума (колонтитулы, дедупликация)
│   ├── ethnonym_extractor.py
│   ├── epithet_extractor.py
│   ├── representation_classifier.py   # Класс R
│   ├── situation_classifier.py        # Класс O
│   ├── essentialization.py
│   ├── piro.py             # PIRO-разметка на каждое упоминание
│   ├── discourse_patterns.py
│   ├── normalization.py    # Нормировка на 10k предложений
│   ├── keyness.py          # Keyness (G2), визуализации
│   ├── relations.py        # Сети co-mention и interaction
│   ├── visualization.py    # Графики, heatmap, wordcloud
│   ├── embeddings_clusters.py
│   ├── report.py           # PDF-отчёт
│   ├── deepseek_analysis.py # Обзор для PDF (observations, limitations)
│   ├── evidence_pack.py
│   ├── export_excel.py
│   └── ...
│
├── llm/                    # Модуль DeepSeek
│   ├── deepseek_client.py  # Вызов API, логи, учёт токенов
│   ├── memo_engine.py      # Мемо по блокам (R, O, keyness, …)
│   ├── synthesis.py        # Межтабличный синтез (OI, AS, ED, EPS)
│   ├── system_prompt.txt   # Системный промпт для LLM
│   └── README.md           # Подробнее про LLM в проекте
│
├── api/                    # Flask API для реактивного дашборда
│   └── app.py              # POST /api/analyze, кэш в llm_cache/
│
├── reports/
│   ├── build_report.py     # Сборка output/report.html
│   └── report_assets.py    # Ресурсы для офлайн-отчёта
│
├── analysis/
│   ├── derived_indices.py  # OI, AS, ED, EPS → derived_indices.json
│   └── derived_analytics.py # Второй слой: профили, тесты, корреляции, кластеры → output/derived/, output/tables/
│
├── scripts/
│   └── derived_analytics.py # Запуск только derived analytics: python scripts/derived_analytics.py --pipeline output/pipeline.db
│
├── exporters/
│   └── export_excel.py     # PIRO fragments, full database
│
├── docs/
│   └── ANALYSIS_AND_IMPROVEMENTS.md  # Анализ слабых мест и план улучшения научного отчёта
│
└── resources/              # Конфиги (YAML) и база знаний
    ├── ethnonyms.yml
    ├── representation_lexicons.yml
    ├── situation_domains.yml
    ├── interaction_verbs.yml
    ├── ocr_replacements.yml
    ├── sentiment_lexicon.yml
    └── knowledge/
        └── domain_briefing.txt   # Теоретическая рамка для LLM (ориентализм, травелоги)
```

---

## 4. Входные и выходные данные

**Вход:**

- **Тексты:** каталог `data/texts/` — файлы `*.txt` (травелоги). Имена файлов используются как идентификаторы документов.

**Выход (каталог `output/`):**

| Файл / каталог | Описание |
|----------------|----------|
| `report.html` | Основной интерактивный отчёт (графики, таблицы, аналитика, раздел по токенам). |
| `pipeline.db` | Кэш пайплайна: корпус, raw/clean df, PIRO, keyness, нормы, сети, индексы. Нужен для `--from-db`. |
| `corpus.db` | Кэш предобработанного корпуса. Нужен для `--from-cache`. |
| `llm_memos/*.json` | Мемо по блокам (representation, situation, keyness, …) и `synthesis.json`. |
| `metadata/llm_token_count.json` | Учёт токенов: prompt/completion/total, число вызовов с LLM и без. |
| `metadata/llm_mode.json` | Режим и метаданные запуска LLM. |
| `llm_analysis.json` | Сводный обзор для PDF (observations, limitations). |
| `derived_indices.json` | Индексы OI, AS, ED, EPS (из шага 8b). |
| **Второй слой (derived, шаг 8c)** | *Ниже перечисленные артефакты создаются поверх готового пайплайна; **corpus.db и структура БД не изменяются**.* |
| `derived/derived_indices.json` | Индексы + формулы, normalization base, CI. |
| `derived/stats_tests.json` | Chi-square Ethnos×R, Ethnos×O, Cramér's V; bootstrap 95% CI по топ-этносам. |
| `derived/correlations.csv` | Pearson и Spearman для OI, ED, EPS, AS, mentions_norm. |
| `derived/ethnos_clusters.csv` | Кластеры KMeans (k по silhouette). |
| `derived/run_config.json` | Дата запуска, выбор нормализации (per 10k предложений). |
| `tables/ethnic_profiles.csv`, `tables/ethnic_profiles.xlsx` | Профили по этносам: mentions_raw, mentions_norm, OI, EPS, ED, AS, доли, deltas. |
| `llm_memos/synthesis.json` | Синтез по производным метрикам (при `--run-llm` или rule-based). |
| `logs/derived_analytics.log` | Лог шага derived analytics. |
| `piro_fragments.xlsx` | Фрагменты PIRO для верификации. |
| `evidence_pack.xlsx` | При `--build-evidence-pack` или `--full`. |
| `piro_full_database.xlsx` | При `--export-excel` или `--full`. |
| `final_report.pdf` | При `--report-pdf` или `--full`. |
| `tables/`, `figures/` | Таблицы keyness, графики, сети, UMAP. |
| `assets/` | Скрипты и стили для офлайн просмотра отчёта. |
| `logs/llm_calls.log` | Лог вызовов LLM (размеры запросов/ответов). |

---

## 5. Пайплайн (шаги)

По `main.py` выполняются (в зависимости от флагов) следующие шаги:

| Шаг | Описание |
|-----|----------|
| 1 | Загрузка текстов из `data/texts/`. |
| 2 | Очистка OCR (`ocr_cleaner`); при `--ocr-symspell` — SymSpell. |
| 3 | Предобработка spaCy (предложения, токены, POS, deps); кэш в `corpus.db`. |
| 4 | Построение таблицы упоминаний этнонимов (`mentions_df`). |
| 5 | Фильтр шума (колонтитулы, дедупликация) → `raw_df`, `clean_df`. |
| 6 | PIRO: эпитеты, R, O, эссенциализация; экспорт `piro_fragments.xlsx`. |
| 7 | Нормировка (на 10k предложений) и keyness (G2, таблицы, графики). |
| 8 | Сети: co-mention и interaction; визуализации. |
| 8b | Производные индексы (OI, AS, ED, EPS) → `derived_indices.json`. |
| **8c** | **Derived analytics** (по умолчанию вкл., отключение: `--no-derived`): профили этносов, тесты (chi-square, Cramér's V, bootstrap CI), корреляции, кластеры → `output/derived/`, `output/tables/ethnic_profiles.*`. Не изменяет `corpus.db`. |
| 9 | Evidence pack (при флаге). |
| 10 | Экспорт полной базы в Excel (при флаге). |
| 11 | Embeddings и кластеризация (при `--run-embeddings` или `--full`). |
| 12 | **LLM:** мемо по блокам, синтез, обзор для PDF; сброс и запись счётчика токенов. |
| 13 | Загрузка мемо для отчёта (из файлов или rule-based fallback). |
| 14 | Подготовка ресурсов отчёта (assets). |
| 15 | Сборка HTML-отчёта (`report.html`), в т.ч. раздел по токенам. |
| 16 | Самопроверки (sanity_checks). |
| 17 | Генерация PDF (при `--report-pdf` или `--full`). |

При запуске с `--from-db` шаги 1–8b не пересчитываются: данные берутся из `pipeline.db`. Шаг 12 (LLM) при `--from-db` не выполняется.

---

## 6. Режимы запуска

### Режим A — полный академический отчёт с LLM

Максимально полный отчёт: мемо по блокам, синтез, обзор для PDF, учёт токенов в отчёте.

```bash
python main.py --run-llm
```

С уже сохранённым корпусом (без повторной загрузки и spaCy):

```bash
python main.py --from-cache --run-llm
```

«Всё сразу» (embeddings, LLM, PDF, Excel, evidence pack):

```bash
python main.py --full
```

Перед этим задайте `DEEPSEEK_API_KEY` (переменная окружения или `.env`). В начале запуска выведется: *«Режим: полный прогон с LLM — максимально полный академический отчёт и учёт токенов.»*

##### Режим C — анализ по одному или нескольким документам

Полная аналитика (и отчёт с LLM) только по выбранным травелогам — например, только по тексту А. Мичи для статьи.

**Селектор** — подстрока в имени файла (без учёта регистра) или точное имя файла. В `data/texts/` должны лежать нужные `.txt`; имена файлов используются для отбора.

Примеры:

```bash
# Только документы, в имени которых есть "michie" (напр. michie_1859_travel.txt)
python main.py --documents michie --run-llm

# Несколько авторов: подходят файлы с "michie" или "atkinson"
python main.py --documents michie atkinson --run-llm
# или через запятую:
python main.py --documents "michie,atkinson" --run-llm

# С кэшем корпуса: из corpus.db берутся только документы с "michie"
python main.py --from-cache --documents michie --run-llm
```

При указании `--documents` загрузка из `pipeline.db` (`--from-db`) отключается: пайплайн всегда считается заново по выбранному подкорпусу (из файлов или из `corpus.db`). Результат — тот же набор артефактов (`report.html`, мемо, синтез, научный отчёт), но только по выбранным документам.

### Режим B — аналитика по готовой базе

Без пересчёта пайплайна и без вызовов LLM. Требуется уже созданный `output/pipeline.db` (один раз выполните полный прогон без `--from-db`).

```bash
python main.py --from-db
```

В начале запуска выведется: *«Режим: загрузка из базы (output/pipeline.db), аналитика без пересчёта и без LLM.»* Для этого режима spaCy не нужен.

### Основные флаги

| Флаг | Описание |
|------|----------|
| `--from-cache` | Загрузить корпус из `output/corpus.db` (пропуск шагов 1–3). |
| `--from-db` | Загрузить всё из `output/pipeline.db` (аналитика без пересчёта и без LLM). |
| `--run-llm` | Включить LLM: мемо, синтез, обзор для PDF, учёт токенов. |
| `--run-deepseek` | То же, что `--run-llm`. |
| `--full` | Полный цикл: embeddings, LLM, report-pdf, export-excel, build-evidence-pack. |
| `--report-pdf` | Дополнительно сгенерировать `final_report.pdf`. |
| `--export-excel` | Экспорт `piro_full_database.xlsx`. |
| `--build-evidence-pack` | Собрать `evidence_pack.xlsx`. |
| `--no-derived` | Отключить второй слой аналитики (профили, тесты, корреляции, кластеры). По умолчанию derived включён. |
| `--documents NAME [NAME ...]` | Анализ только по выбранным документам: подстрока имени файла (напр. `michie`) или точное имя; несколько — через пробел или запятую. |
| `--use-lemmas` | Поиск этнонимов и совпадения лексиконов (репрезентация R, ситуация O) по **леммам** токенов (spaCy). **По умолчанию (без флага) везде используется поиск по словоформам.** |
| `--run-embeddings` | Кластеризация и UMAP. |
| `--ocr-symspell` | Усиленная очистка OCR (SymSpell). |
| `--jobs N` | Число процессов для spaCy (по умолчанию 1). |

### Слова и леммы

**По умолчанию поиск везде идёт по словоформам** (regex по тексту для этнонимов, совпадение слов из контекста с лексиконами R/O). Проект **не заменяет** этот режим: с флагом `--use-lemmas` включается **дополнительный** режим, в котором:
- этнонимы ищутся по леммам токенов (spaCy);
- совпадения с лексиконами репрезентации (R) и доменами ситуации (O) считаются по леммам.

Режим по леммам действует только в **основном пайплайне** (загрузка текстов → mentions_df → PIRO из mentions_df). По словоформам остаются: **сети** (co-mention, interaction в `relations.py`), **эссенциализация** (`discourse_patterns.py`), а также альтернативный путь `run_piro_on_corpus()` (если вызывать его напрямую, без `main.py`).

---

## 7. Модули и компоненты

**src/** — ядро пайплайна:

- **loader** — загрузка `.txt` из `data/texts/`; опция `include_docs` для выбора подкорпуса (один или несколько документов).
- **ethnonym_extractor** — поиск этнонимов: по умолчанию regex по словоформам (варианты из `ethnonyms.yml`); при `--use-lemmas` — по леммам токенов (каноническое имя = лемма).
- **representation_classifier** / **situation_classifier** — при `--use-lemmas` совпадения с лексиконами R и доменами O ищутся по леммам контекста; сами лексиконы приводятся к леммам через spaCy.
- **ocr_cleaner** — очистка OCR-ошибок.
- **preprocess** — сегментация, токены, POS, deps (spaCy).
- **corpus_db** / **pipeline_db** — кэш корпуса и пайплайна в SQLite.
- **db_introspect** — таблица упоминаний по корпусу.
- **noise_filter** — фильтр шума (колонтитулы, дедупликация).
- **ethnonym_extractor**, **epithet_extractor** — извлечение упоминаний и эпитетов.
- **representation_classifier** / **situation_classifier** — классы R и O.
- **piro** — PIRO-разметка (контекст, эпитеты, R, O, эссенциализация).
- **normalization** — нормировка на 10k предложений по этносам, R, O.
- **keyness** — keyness (G2), POS-фильтры, таблицы и графики.
- **relations** — сети co-mention и interaction.
- **visualization** — графики частот, heatmap, wordcloud, сети.
- **report** — итоговый PDF; **deepseek_analysis** — обзор для PDF через LLM.

**llm/** — DeepSeek:

- **deepseek_client** — вызов API, логи, учёт токенов (`llm_token_count.json`).
- **memo_engine** — мемо по блокам (R, O, keyness, network, essentialization, embeddings).
- **synthesis** — межтабличный синтез по индексам OI, AS, ED, EPS.

**api/** — Flask API для кнопки «Пересчитать аналитическую интерпретацию» в отчёте: `POST /api/analyze`, кэш в `output/llm_cache/`. Запуск: `python -m api.app` (порт 5000).

**reports/** — сборка `report.html` и загрузка assets (jQuery, Plotly, DataTables).

**analysis/** — производные индексы (`derived_indices.py`: OI, AS, ED, EPS) и второй слой (`derived_analytics.py`: профили этносов, тесты, корреляции, кластеры). Второй слой **не изменяет** `corpus.db` и не переписывает пайплайн; читает данные из `pipeline.db` (или из памяти при вызове из `main.py`) и пишет в `output/derived/` и `output/tables/`.

**resources/** — YAML: этнонимы, лексиконы R/O, домены ситуации, глаголы взаимодействия, OCR-замены, sentiment.

---

## 8. LLM (DeepSeek) и учёт токенов

- **Модель:** deepseek-chat (OpenAI-совместимый API).
- **Ключ:** `DEEPSEEK_API_KEY` в окружении или в `.env` в корне проекта.
- **Использование:** мемо по блокам, синтез (`synthesis.json`), обзор для PDF (`llm_analysis.json`). Системный промпт — `llm/system_prompt.txt`.
- **Без ключа:** возвращается «LLM disabled», в отчёте подставляются rule-based мемо.

**Учёт токенов:**

- Файл: **`output/metadata/llm_token_count.json`**.
- Содержит: `total_prompt_tokens`, `total_completion_tokens`, `total_tokens`, `n_calls`, `n_calls_llm_used`, `n_calls_without_llm`, `last_updated`, список вызовов с `call_id`.
- При запуске с `--run-llm` счётчик в начале шага 12 сбрасывается (учёт только текущего запуска).
- В HTML-отчёте раздел **«Использование LLM (учёт токенов)»** выводит эти данные; при отсутствии LLM в запуске — пояснительный текст.

Подробнее: **`llm/README.md`**.

---

## 9. Выходные артефакты

- **report.html** — главный отчёт (резюме, корпус, распределения, keyness, эссенциализация, сети, индексы, evidence, embedding, синтез, ограничения, **учёт токенов**, вспомогательные визуализации).
- **pipeline.db** — для повторных запусков с `--from-db`.
- **corpus.db** — для запусков с `--from-cache`.
- **llm_memos/** — JSON-мемо и synthesis; **metadata/llm_token_count.json** — токены.
- **derived_indices.json**, **llm_analysis.json** — индексы и обзор для PDF.
- **piro_fragments.xlsx**, при флагах — **evidence_pack.xlsx**, **piro_full_database.xlsx**, **final_report.pdf**.
- **tables/**, **figures/**, изображения сетей, UMAP, логи в **logs/**.

### 9.1 Второй слой (derived): формулы и тесты

Слой **не меняет** `corpus.db` и не выполняет миграций; читает данные из пайплайна и пишет в `output/derived/` и `output/tables/`.

**Нормализация:** единый стандарт — **на 10k предложений** (число предложений берётся из корпуса). Фиксируется в `output/derived/run_config.json`.

**Формулы индексов (по этносу):**

| Индекс | Формула | Смысл |
|--------|---------|--------|
| **OI** | (negative + exotic) / total_mentions | Доля негативной и экзотизирующей репрезентации. |
| **EPS** | (negative − positive) / total_mentions | Перекос негатив/позитив. |
| **ED** | essentializing_mentions / total_mentions | Доля эссенциализирующих конструкций (если нет разметки — в профиле NaN). |
| **AS** | (outgoing − incoming) / total_edges (по узлу) | Дисбаланс агентности в сети interaction (если нет рёбер — NaN). |

**Effect sizes:** OI_delta, ED_delta, EPS_delta, AS_delta — отклонение от взвешенного среднего по этносам (вес = mentions_raw).

**Статистические проверки (stats_tests.json):**

- **Chi-square** для таблиц сопряжённости Ethnos×R и Ethnos×O.
- **Cramér's V** — сила связи (0 — нет связи, 1 — максимальная). Интерпретация: &lt;0.1 слабая, 0.1–0.3 умеренная, &gt;0.3 сильная.
- **Bootstrap 95% CI** для OI, EPS, ED по топ-10 этносов по числу упоминаний.

**Запуск только derived (без полного пайплайна):**  
`python scripts/derived_analytics.py --pipeline output/pipeline.db` (требуется уже собранный `pipeline.db`).

---

## 10. Устранение неполадок

| Проблема | Решение |
|----------|---------|
| Нет текстов | Положите файлы `*.txt` в `data/texts/`. |
| Ошибка spaCy / модель не найдена | Выполните `python -m spacy download en_core_web_sm`. |
| Нет pipeline.db при `--from-db` | Сначала выполните полный прогон без `--from-db` (создастся `pipeline.db`). |
| Нет упоминаний этнонимов | Проверьте `resources/ethnonyms.yml` и соответствие текстов (английский, этнонимы из списка). |
| Пустые блоки «Расширенная интерпретация» | При отсутствии файлов в `llm_memos/` подставляются rule-based мемо. Для LLM-текста запустите с `--run-llm`. |
| LLM не вызывается | Проверьте `DEEPSEEK_API_KEY` (или `.env`). Просмотрите `output/metadata/llm_token_count.json`: при `n_calls_llm_used == 0` и `error: "no_api_key"` ключ не задан или не подхвачен. |
| Реактивная аналитика в отчёте не работает | Запустите API: `python -m api.app`. В отчёте используется `DASHBOARD_API_BASE` (по умолчанию `http://127.0.0.1:5000`). |
| Много шума в выборке | Проверьте `output/noise_with_ethnonyms.csv` и при необходимости настройки в `noise_filter`. |

---

## Лицензия и контекст

Проект предназначен для исследовательского использования в области цифровых гуманитарных наук и исследований ориентализма. Корпус — англоязычные травелоги о Сибири; метаданные и методология описаны в отчёте и в коде.
