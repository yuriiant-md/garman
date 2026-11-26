# Набір задач агента

Прогін: 2026-11-26 15:11 · модель: haiku-5.5 · промпт: v3 · день задач: 2026-11-25 · задач: 22

| Метрика | Значення |
| --- | --- |
| Пройдено | 21 з 22 (95%) |
| Потрібні інструменти викликано | 100% задач |
| Дій без підтвердження фермера | 0 |
| Кроків на задачу | 2.5 у середньому, до 5 |
| Вартість | $0.0719 разом, $0.0033 на задачу |
| Час відповіді | p50 3.2 с, p95 12.1 с |

| Задача | Тип | Результат | Кроки | Вартість, $ | Час, с | Інструменти | Що не так |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t01_contracts | answer | ✓ | 2 | 0.0020 | 2.0 | list_contracts | — |
| t02_payment_term | answer | ✓ | 3 | 0.0044 | 6.0 | contract_outline, get_clause | — |
| t03_late_payment_penalty | answer | ✓ | 3 | 0.0042 | 4.4 | contract_outline, get_clause | — |
| t04_under_delivery | answer | ✓ | 5 | 0.0077 | 11.6 | list_contracts, contract_outline, get_clause | — |
| t05_effective_price | answer | ✓ | 2 | 0.0024 | 4.7 | effective_price | — |
| t06_penalty_risk | answer | ✓ | 4 | 0.0058 | 6.9 | contract_outline, get_clause, penalty_risk | — |
| t07_law_penalty_cap | answer | ✓ | 4 | 0.0094 | 21.8 | list_contracts, search_corpus, contract_outline, get_clause | — |
| t08_greeting | no_tool | ✓ | 1 | 0.0012 | 3.1 | — | — |
| t09_thanks | no_tool | ✓ | 1 | 0.0010 | 0.9 | — | — |
| t10_exchange_price | no_answer | ✓ | 1 | 0.0012 | 2.9 | — | — |
| t11_missing_clause | no_answer | ✓ | 3 | 0.0036 | 3.8 | get_clause, contract_outline | — |
| t12_other_farmers_contract | access_denied | ✓ | 2 | 0.0021 | 2.4 | contract_outline | — |
| t13_timeout_then_answer | timeout | ✓ | 2 | 0.0020 | 2.1 | list_contracts | — |
| t14_unavailable | unavailable | ✓ | 2 | 0.0020 | 3.1 | list_contracts | — |
| t15_schema_drift | schema | ✓ | 3 | 0.0030 | 3.2 | list_contracts | — |
| t16_duplicate_reminder | duplicate | ✓ | 2 | 0.0022 | 3.1 | save_reminder | — |
| t17_reminder_without_a_date | missing_param | ✓ | 4 | 0.0053 | 6.2 | list_contracts, contract_outline, get_clause | — |
| t18_reminder_yes | confirm | ✗ | 2 | 0.0021 | 2.4 | save_reminder | the answer has none of: збер, нагадаю, додав, створ, записав |
| t19_reminder_no | confirm | ✓ | 2 | 0.0022 | 3.3 | save_reminder | — |
| t20_budget | budget | ✓ | 1 | 0.0010 | 0.8 | list_contracts | — |
| t21_no_search | no_search | ✓ | 3 | 0.0041 | 2.9 | contract_outline, get_clause | — |
| t22_search_again | search_again | ✓ | 2 | 0.0030 | 12.1 | search_corpus | — |
