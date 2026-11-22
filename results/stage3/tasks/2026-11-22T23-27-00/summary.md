# Набір задач агента

Прогін: 2026-11-22 23:27 · модель: haiku-5.5 · промпт: v1 · день задач: 2026-11-25 · задач: 20

| Метрика | Значення |
| --- | --- |
| Пройдено | 20 з 20 (100%) |
| Потрібні інструменти викликано | 100% задач |
| Дій без підтвердження фермера | 0 |
| Кроків на задачу | 2.6 у середньому, до 5 |
| Вартість | $0.0592 разом, $0.0030 на задачу |
| Час відповіді | p50 3.9 с, p95 9.7 с |

| Задача | Тип | Результат | Кроки | Вартість, $ | Час, с | Інструменти | Що не так |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t01_contracts | answer | ✓ | 2 | 0.0015 | 2.2 | list_contracts | — |
| t02_payment_term | answer | ✓ | 3 | 0.0036 | 5.5 | contract_outline, get_clause | — |
| t03_late_payment_penalty | answer | ✓ | 3 | 0.0036 | 5.5 | contract_outline, get_clause | — |
| t04_under_delivery | answer | ✓ | 3 | 0.0037 | 7.4 | contract_outline, get_clause | — |
| t05_effective_price | answer | ✓ | 2 | 0.0018 | 4.0 | effective_price | — |
| t06_penalty_risk | answer | ✓ | 4 | 0.0049 | 8.0 | contract_outline, get_clause, penalty_risk | — |
| t07_law_penalty_cap | answer | ✓ | 4 | 0.0078 | 17.9 | list_contracts, search_corpus, contract_outline, get_clause | — |
| t08_greeting | no_tool | ✓ | 1 | 0.0009 | 2.6 | — | — |
| t09_thanks | no_tool | ✓ | 1 | 0.0007 | 0.9 | — | — |
| t10_exchange_price | no_answer | ✓ | 1 | 0.0009 | 2.6 | — | — |
| t11_missing_clause | no_answer | ✓ | 3 | 0.0028 | 3.7 | get_clause, contract_outline | — |
| t12_other_farmers_contract | access_denied | ✓ | 2 | 0.0016 | 3.0 | contract_outline | — |
| t13_timeout_then_answer | timeout | ✓ | 2 | 0.0015 | 2.3 | list_contracts | — |
| t14_unavailable | unavailable | ✓ | 2 | 0.0015 | 3.5 | list_contracts | — |
| t15_schema_drift | schema | ✓ | 3 | 0.0023 | 3.4 | list_contracts | — |
| t16_duplicate_reminder | duplicate | ✓ | 4 | 0.0050 | 7.1 | contract_outline, get_clause, save_reminder | — |
| t17_reminder_without_a_date | missing_param | ✓ | 3 | 0.0035 | 5.7 | contract_outline, get_clause | — |
| t18_reminder_yes | confirm | ✓ | 4 | 0.0050 | 6.6 | contract_outline, get_clause, save_reminder | — |
| t19_reminder_no | confirm | ✓ | 5 | 0.0061 | 9.3 | list_contracts, contract_outline, get_clause, save_reminder | — |
| t20_budget | budget | ✓ | 1 | 0.0007 | 0.8 | list_contracts | — |
