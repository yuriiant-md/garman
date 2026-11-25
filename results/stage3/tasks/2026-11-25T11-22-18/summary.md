# Набір задач агента

Прогін: 2026-11-25 11:22 · модель: haiku-5.5 · промпт: v2 · день задач: 2026-11-25 · задач: 22

| Метрика | Значення |
| --- | --- |
| Пройдено | 22 з 22 (100%) |
| Потрібні інструменти викликано | 100% задач |
| Дій без підтвердження фермера | 0 |
| Кроків на задачу | 2.4 у середньому, до 4 |
| Вартість | $0.0585 разом, $0.0027 на задачу |
| Час відповіді | p50 4.1 с, p95 10.6 с |

| Задача | Тип | Результат | Кроки | Вартість, $ | Час, с | Інструменти | Що не так |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t01_contracts | answer | ✓ | 2 | 0.0016 | 2.0 | list_contracts | — |
| t02_payment_term | answer | ✓ | 3 | 0.0037 | 5.4 | contract_outline, get_clause | — |
| t03_late_payment_penalty | answer | ✓ | 3 | 0.0037 | 4.8 | contract_outline, get_clause | — |
| t04_under_delivery | answer | ✓ | 4 | 0.0055 | 9.6 | contract_outline, get_clause | — |
| t05_effective_price | answer | ✓ | 2 | 0.0019 | 4.0 | effective_price | — |
| t06_penalty_risk | answer | ✓ | 4 | 0.0050 | 7.2 | contract_outline, get_clause, penalty_risk | — |
| t07_law_penalty_cap | answer | ✓ | 2 | 0.0024 | 10.7 | search_corpus, list_contracts | — |
| t08_greeting | no_tool | ✓ | 1 | 0.0009 | 2.5 | — | — |
| t09_thanks | no_tool | ✓ | 1 | 0.0008 | 0.9 | — | — |
| t10_exchange_price | no_answer | ✓ | 1 | 0.0010 | 2.7 | — | — |
| t11_missing_clause | no_answer | ✓ | 3 | 0.0029 | 3.9 | get_clause, contract_outline | — |
| t12_other_farmers_contract | access_denied | ✓ | 2 | 0.0017 | 2.9 | contract_outline | — |
| t13_timeout_then_answer | timeout | ✓ | 2 | 0.0016 | 2.4 | list_contracts | — |
| t14_unavailable | unavailable | ✓ | 2 | 0.0016 | 3.3 | list_contracts | — |
| t15_schema_drift | schema | ✓ | 3 | 0.0024 | 3.1 | list_contracts | — |
| t16_duplicate_reminder | duplicate | ✓ | 3 | 0.0037 | 6.0 | contract_outline, save_reminder | — |
| t17_reminder_without_a_date | missing_param | ✓ | 3 | 0.0038 | 6.0 | contract_outline, get_clause | — |
| t18_reminder_yes | confirm | ✓ | 2 | 0.0019 | 4.1 | save_reminder | — |
| t19_reminder_no | confirm | ✓ | 4 | 0.0052 | 6.8 | contract_outline, get_clause, save_reminder | — |
| t20_budget | budget | ✓ | 1 | 0.0008 | 0.8 | list_contracts | — |
| t21_no_search | no_search | ✓ | 3 | 0.0036 | 4.3 | contract_outline, get_clause | — |
| t22_search_again | search_again | ✓ | 2 | 0.0027 | 13.5 | search_corpus | — |
