# Гарман

Telegram-бот для фермера: витягує умови з договорів трейдерів, знаходить ризики з посиланнями на джерела та допомагає порівнювати пропозиції. Курсовий проєкт розвивається поетапно; зараз готується етап 1 — витягування умов угоди й порівняння моделей.

## Встановлення

Потрібні Python 3.14 та [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Налаштування

Створи локальний файл `.env` на основі прикладу й додай ключ Anthropic:

```bash
cp .env.example .env
```

Не додавай `.env` або оригінали договорів до Git. Сирі документи зберігаються в `data/raw/`.

## Запуск

Команда `make run` буде запускати повний прогін етапу 1 після реалізації extraction pipeline.

Поки що прогін і метрики запускаються окремо:

```bash
uv run --env-file .env python -m src.extraction.run --split dev   # Claude → results/stage1/*.jsonl
uv run python -m src.extraction.run --model lapa --split dev      # Lapa через Ollama
uv run python -m src.extraction.metrics                           # таблиця модель × промпт × семплінг
```

Метрики рахуються за правилами з `docs/PLAN.md`: точність заповнених полів із 95% бутстреп-інтервалом, частка вигаданих значень, загальна точність, валідні й failed відповіді, latency p50/p95 і вартість на 1000 документів. Рядки порівнюються після нормалізації (регістр, лапки, апострофи, правова форма: ТОВ, ТзОВ, ПрАТ, АТ, ДП, ПП, ФГ) і зі схожістю ≥ 0.85, бо в golden Prozorro збережено OCR-шум.

Підготовка golden set із договорів Prozorro (докладно — у `data/README.md`):

```bash
uv run python -m scripts.fetch_prozorro    # кандидати в data/raw/prozorro_candidates/
uv run python -m scripts.pdf_to_text       # data/raw/prozorro/*.pdf → data/texts/prozorro_NNN.txt
uv run python -m scripts.check_golden      # перевірка ручної розмітки data/golden/*.json
```

## Де що лежить

### Вимоги етапу 1

| Вимога | Де дивитися | Стан |
| --- | --- | --- |
| 30 «брудних» текстів без персональних даних | `data/texts/`: `synthetic_001–020.txt` (dev) і `prozorro_001–010.txt` (test); як знеособлено — `data/README.md` | готово |
| 10 текстів розмічено вручну (golden set) | `data/golden/prozorro_001–010.json`, правила розмітки — `data/golden/RULES.md` | готово |
| Схема: ≥ 6 полів, ≥ 1 enum, вкладений об'єкт або список | `src/schemas/deal.py` (10 полів, 9 enum, 5 вкладених об'єктів, список `penalties`), тести — `tests/test_deal.py` | готово |
| Провайдерська модель через API зі structured output, retry і timeout | `src/extraction/anthropic_extractor.py` (схема як обов'язковий виклик інструмента), налаштування — `configs/models.yaml` | готово |
| Локальна модель через спільний інтерфейс | `src/extraction/ollama_extractor.py`; інтерфейс `Extractor` — `src/extraction/run.py` | готово |
| temperature і top_p у конфігу, ≥ 2 конфігурації | `configs/sampling.yaml` (`precise`, `varied`) | конфіг готовий; прогін `varied` — далі |
| Метрики: точність по полях, % валідних, p50/p95, вартість на 1000 документів | `src/extraction/metrics.py` (таблиця), `src/extraction/scoring.py` (порівняння з еталоном); тести — `tests/test_metrics.py`, `tests/test_scoring.py` | код готовий; цифри поки лише на dev |
| Промпти з версією в шапці, ≥ 2 версії, `CHANGELOG` | `prompts/extract_v1.md`, `prompts/CHANGELOG.md` | v1 готова; v2 — далі |
| Кожна відповідь валідується схемою; невалідна → одна спроба виправлення → `failed` | `src/extraction/validation.py` | перевірка схемою готова; спроба виправлення — далі |
| До кожного результату логуються модель, версія промпту, семплінг | поля запису — `src/extraction/result.py`; самі записи — `results/stage1/*.jsonl` | готово (результати поки локально) |
| `REPORT.md`: таблиця, розбір помилок, вибір для продакшну | — | далі |
| Запуск однією командою `make run` | — | далі; поки команди з розділу «Запуск» |

### Папки й файли

```
configs/
  models.yaml              моделі, timeout, retry, ціни за 1M токенів
  sampling.yaml            конфігурації семплінгу precise і varied
data/
  texts/                   30 текстів, які отримують моделі
  synthetic/               еталонний Deal і параметри «брудності» для кожного синтетичного тексту
  golden/                  ручний еталон договорів Prozorro + RULES.md
  prozorro_sources.csv     з якої закупівлі кожен договір
  README.md                як зібрано, знеособлено й розмічено дані
  raw/                     оригінали PDF і файли перевірки маскування (не в git)
prompts/
  extract_v1.md            baseline-промпт, версія в шапці
  CHANGELOG.md             що змінювалось між версіями і чому
src/schemas/
  deal.py                  схема угоди Deal
src/extraction/
  anthropic_extractor.py   Claude Haiku 4.5 через Anthropic API
  ollama_extractor.py      Lapa 12B через Ollama
  run.py                   спільний інтерфейс Extractor, прогін текстів, запис результатів
  messages.py              однакове повідомлення з текстом угоди для обох моделей
  prompts.py               читання промпту з версією й хешем файлу
  sampling.py              читання configs/sampling.yaml
  validation.py            перевірка відповіді схемою Deal
  result.py                що записується про кожну відповідь
  scoring.py               порівняння однієї відповіді з еталоном
  metrics.py               таблиця метрик модель × промпт × семплінг
scripts/
  generate_synthetic.py    угоди для синтетичного набору (+ synthetic_render.py, synthetic_checks.py)
  fetch_prozorro.py        пошук договорів у Prozorro
  pdf_to_text.py           PDF → знеособлений текст
  check_golden.py          перевірка розмітки golden set
tests/                     тести pytest: uv run pytest
results/stage1/            відповіді моделей *.jsonl і таблиця метрик (поки не в git)
```

План і послідовність робіт описані в `docs/PLAN.md` та `docs/PLAN_SHORT.md`.
