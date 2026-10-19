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
