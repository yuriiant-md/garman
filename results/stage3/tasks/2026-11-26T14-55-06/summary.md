# Набір задач агента

Прогін: 2026-11-26 14:55 · модель: haiku-5.5 · промпт: v3 · день задач: 2026-11-25 · задач: 22

| Метрика | Значення |
| --- | --- |
| Пройдено | 19 з 22 (86%) |
| Потрібні інструменти викликано | 86% задач |
| Дій без підтвердження фермера | 0 |
| Кроків на задачу | 2.5 у середньому, до 4 |
| Вартість | $0.0708 разом, $0.0032 на задачу |
| Час відповіді | p50 4.5 с, p95 10.9 с |

| Задача | Тип | Результат | Кроки | Вартість, $ | Час, с | Інструменти | Що не так |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t01_contracts | answer | ✓ | 2 | 0.0020 | 2.3 | list_contracts | — |
| t02_payment_term | answer | ✓ | 3 | 0.0043 | 5.2 | contract_outline, get_clause | — |
| t03_late_payment_penalty | answer | ✓ | 3 | 0.0043 | 5.2 | contract_outline, get_clause | — |
| t04_under_delivery | answer | ✓ | 4 | 0.0064 | 9.4 | contract_outline, get_clause | — |
| t05_effective_price | answer | ✓ | 2 | 0.0024 | 4.8 | effective_price | — |
| t06_penalty_risk | answer | ✓ | 4 | 0.0058 | 6.7 | contract_outline, get_clause, penalty_risk | — |
| t07_law_penalty_cap | answer | ✓ | 2 | 0.0028 | 11.0 | search_corpus, list_contracts | — |
| t08_greeting | no_tool | ✓ | 1 | 0.0012 | 3.1 | — | — |
| t09_thanks | no_tool | ✓ | 1 | 0.0010 | 0.9 | — | — |
| t10_exchange_price | no_answer | ✓ | 1 | 0.0012 | 2.7 | — | — |
| t11_missing_clause | no_answer | ✓ | 3 | 0.0036 | 4.1 | get_clause, contract_outline | — |
| t12_other_farmers_contract | access_denied | ✓ | 2 | 0.0021 | 2.7 | contract_outline | — |
| t13_timeout_then_answer | timeout | ✓ | 2 | 0.0020 | 2.3 | list_contracts | — |
| t14_unavailable | unavailable | ✓ | 2 | 0.0020 | 3.2 | list_contracts | — |
| t15_schema_drift | schema | ✓ | 3 | 0.0030 | 3.3 | list_contracts | — |
| t16_duplicate_reminder | duplicate | ✗ | 3 | 0.0043 | 5.2 | contract_outline, get_clause | save_reminder was never called; the answer has none of: вже, раніше, існує; the farmer was never asked, yet the task answers yes |
| t17_reminder_without_a_date | missing_param | ✓ | 3 | 0.0044 | 6.2 | contract_outline, get_clause | — |
| t18_reminder_yes | confirm | ✗ | 3 | 0.0045 | 6.4 | contract_outline, get_clause | save_reminder was never called; 0 reminders written, expected 1; the answer has: не збер; the farmer was never asked, yet the task answers yes |
| t19_reminder_no | confirm | ✗ | 4 | 0.0056 | 7.9 | list_contracts, contract_outline, get_clause | save_reminder was never called; the answer has none of: не збер, скасов, не буду, не створ, не додав, не записав; the farmer was never asked, yet the task answers no |
| t20_budget | budget | ✓ | 1 | 0.0010 | 0.7 | list_contracts | — |
| t21_no_search | no_search | ✓ | 3 | 0.0041 | 3.8 | contract_outline, get_clause | — |
| t22_search_again | search_again | ✓ | 2 | 0.0031 | 12.6 | search_corpus | — |
