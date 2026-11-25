# Набір задач агента

Прогін: 2026-11-25 11:08 · модель: haiku-5.5 · промпт: v2 · день задач: 2026-11-25 · задач: 5

| Метрика | Значення |
| --- | --- |
| Пройдено | 4 з 5 (80%) |
| Потрібні інструменти викликано | 100% задач |
| Дій без підтвердження фермера | 0 |
| Кроків на задачу | 3.2 у середньому, до 4 |
| Вартість | $0.0232 разом, $0.0046 на задачу |
| Час відповіді | p50 8.2 с, p95 21.5 с |

| Задача | Тип | Результат | Кроки | Вартість, $ | Час, с | Інструменти | Що не так |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t05_effective_price | answer | ✓ | 2 | 0.0019 | 4.7 | effective_price | — |
| t06_penalty_risk | answer | ✓ | 4 | 0.0050 | 8.2 | contract_outline, get_clause, penalty_risk | — |
| t07_law_penalty_cap | answer | ✓ | 3 | 0.0039 | 21.8 | list_contracts, search_corpus | — |
| t21_no_search | no_search | ✓ | 3 | 0.0036 | 4.1 | contract_outline, get_clause | — |
| t22_search_again | search_again | ✗ | 4 | 0.0088 | 20.3 | list_contracts, search_corpus, contract_outline, get_clause | stopped as budget, expected answered |
