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

## Структура

- `src/schemas/` — Pydantic-схеми угод та інших сутностей.
- `src/extraction/` — витягування даних платною й локальною моделями.
- `prompts/`, `configs/` — версії промптів і параметри семплінгу.
- `data/` — тексти, еталонні дані та джерела.
- `results/` — результати прогонів і метрики.
- `tests/` — тести проєкту.

План і послідовність робіт описані в `docs/PLAN.md` та `docs/PLAN_SHORT.md`.
