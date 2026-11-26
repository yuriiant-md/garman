# Набір задач агента

Прогін: 2026-11-26 15:01 · модель: haiku-5.5 · промпт: v2 · день задач: 2026-11-25 · задач: 3

| Метрика | Значення |
| --- | --- |
| Пройдено | 1 з 3 (33%) |
| Потрібні інструменти викликано | 33% задач |
| Дій без підтвердження фермера | 0 |
| Кроків на задачу | 3.0 у середньому, до 3 |
| Вартість | $0.0118 разом, $0.0039 на задачу |
| Час відповіді | p50 5.8 с, p95 6.7 с |

| Задача | Тип | Результат | Кроки | Вартість, $ | Час, с | Інструменти | Що не так |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t16_duplicate_reminder | duplicate | ✗ | 3 | 0.0043 | 6.7 | contract_outline, get_clause | save_reminder was never called; the answer has none of: вже, раніше, існує; the farmer was never asked, yet the task answers yes |
| t18_reminder_yes | confirm | ✓ | 3 | 0.0032 | 4.5 | list_contracts, save_reminder | — |
| t19_reminder_no | confirm | ✗ | 3 | 0.0044 | 5.8 | contract_outline, get_clause | save_reminder was never called; the answer has none of: не збер, скасов, не буду, не створ, не додав, не записав; the farmer was never asked, yet the task answers no |
