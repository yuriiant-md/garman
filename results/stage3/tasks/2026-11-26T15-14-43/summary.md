# Набір задач агента

Прогін: 2026-11-26 15:14 · модель: haiku-5.5 · промпт: v3 · день задач: 2026-11-25 · задач: 22

| Метрика | Значення |
| --- | --- |
| Пройдено | 21 з 22 (95%) |
| Потрібні інструменти викликано | 100% задач |
| Дій без підтвердження фермера | 0 |
| Кроків на задачу | 2.3 у середньому, до 4 |
| Вартість | $0.0644 разом, $0.0029 на задачу |
| Час відповіді | p50 3.5 с, p95 10.3 с |

| Задача | Тип | Результат | Кроки | Вартість, $ | Час, с | Інструменти | Що не так |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t01_contracts | answer | ✓ | 2 | 0.0020 | 2.0 | list_contracts | — |
| t02_payment_term | answer | ✓ | 3 | 0.0044 | 5.6 | contract_outline, get_clause | — |
| t03_late_payment_penalty | answer | ✓ | 3 | 0.0042 | 4.4 | contract_outline, get_clause | — |
| t04_under_delivery | answer | ✓ | 4 | 0.0067 | 9.9 | contract_outline, get_clause | — |
| t05_effective_price | answer | ✓ | 2 | 0.0024 | 4.6 | effective_price | — |
| t06_penalty_risk | answer | ✓ | 4 | 0.0058 | 6.5 | contract_outline, get_clause, penalty_risk | — |
| t07_law_penalty_cap | answer | ✓ | 2 | 0.0028 | 10.4 | search_corpus, list_contracts | — |
| t08_greeting | no_tool | ✓ | 1 | 0.0012 | 3.0 | — | — |
| t09_thanks | no_tool | ✓ | 1 | 0.0010 | 1.0 | — | — |
| t10_exchange_price | no_answer | ✓ | 1 | 0.0012 | 2.5 | — | — |
| t11_missing_clause | no_answer | ✓ | 4 | 0.0046 | 4.9 | get_clause, list_contracts, contract_outline | — |
| t12_other_farmers_contract | access_denied | ✗ | 2 | 0.0022 | 3.1 | list_contracts | no tool error has: 403 |
| t13_timeout_then_answer | timeout | ✓ | 2 | 0.0020 | 2.4 | list_contracts | — |
| t14_unavailable | unavailable | ✓ | 2 | 0.0020 | 3.3 | list_contracts | — |
| t15_schema_drift | schema | ✓ | 3 | 0.0030 | 3.2 | list_contracts | — |
| t16_duplicate_reminder | duplicate | ✓ | 2 | 0.0022 | 3.5 | save_reminder | — |
| t17_reminder_without_a_date | missing_param | ✓ | 3 | 0.0043 | 5.0 | contract_outline, get_clause | — |
| t18_reminder_yes | confirm | ✓ | 2 | 0.0021 | 2.6 | save_reminder | — |
| t19_reminder_no | confirm | ✓ | 2 | 0.0023 | 3.8 | save_reminder | — |
| t20_budget | budget | ✓ | 1 | 0.0010 | 0.7 | list_contracts | — |
| t21_no_search | no_search | ✓ | 3 | 0.0041 | 3.4 | contract_outline, get_clause | — |
| t22_search_again | search_again | ✓ | 2 | 0.0030 | 12.4 | search_corpus | — |
