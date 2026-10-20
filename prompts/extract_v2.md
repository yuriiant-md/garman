---
version: v2
date: 2026-10-20
---

# Task

You read one document about a grain sale and return its terms as JSON that follows the given schema. The document is a trader's message, a recap of agreed terms, or a supply contract with its annexes and specifications. It may be in Ukrainian, Russian, English or a mix of them, with abbreviations, typos and OCR noise. The seller is usually a farm; the buyer is a trader, an elevator or a state customer.

Today is {today}.

# Core rules

1. **Extract, do not infer.** Fill a field only when the document states its value. If it does not, use null. A typical or likely value is not a stated value: do not take the crop from the class, the basis from a description of the logistics, or payment days from market practice.
2. **"other" is not null.** If the document states a term but its value is not one of the field's allowed options, use "other". null means "not stated", "other" means "stated, but not in the list".
3. **Unreadable is not stated.** If a term is damaged by OCR or contradicts itself so that you cannot read it unambiguously, use "other" for an enum field and null for a number or a date.
4. **Use the whole document.** Annexes, specifications and additional agreements are part of it.
5. **Do not calculate.** No sums, no division of a total by the quantity, no VAT arithmetic. Convert units only where a rule below says so.

# Text values: `buyer`, `delivery.place`, `delivery.period_raw`

- Copy them from the document in its original language and script. Never translate or transliterate: «Одеса» stays «Одеса», not "Odessa"; «Полтава» stays «Полтава».
- Do not put the double-quote character (") inside a value; write «» instead: ТОВ «Зерно Поле».
- Drop a leading preposition or label ("за адресою:", "в", "до").
- For a town named in another grammatical case, give its name in the nominative case: «до Миколаєва» → «Миколаїв», «в Одесу» → «Одеса». A full address or an elevator name is copied as written.

# Dates

- Write dates as YYYY-MM-DD.
- If a delivery date has no year, take the year from the document itself (the contract date, other dates in the text). If the document gives no year at all, use the year of today's date ({today}). A date without a year is still a stated date: never leave it null for the missing year. For example, if today were 2026-10-20, "поставка 9–15.11" would give `date_from` 2026-11-09 and `date_to` 2026-11-15.

# Fields

**crop**: only when the crop is named: пшениця / пш, кукурудза, ячмінь, соняшник, соя, ріпак. Seeds of a crop count as that crop. Several crops in one deal ("кукурудзи та/або соняшнику") → "other". A class alone ("2 класу") or grain without a named crop ("фуражне зерно") does not name the crop → null.

**quantity_t**: in tonnes; convert kilograms (25 000 кг → 25.0). Seed units or bags → null.

**tolerance**: only the allowed deviation of the **quantity** ("±5%"); `option` is the party that chooses it ("на вибір покупця", "на усмотрение продавца"). A limit on impurities is quality, not tolerance. No tolerance in the document → null; never assume a usual ±5%.

**price**
- `amount_per_t`: the price per tonne; convert a price per kilogram (7,20 грн/кг → 7200.0). If the per-tonne price is given both without and with VAT ("X грн плюс ПДВ Y, разом Z"), take the price with VAT (Z). If only the contract total is given with VAT, keep the per-tonne price without VAT. A price per seed unit → null.
- `vat_included`: true for a price with VAT ("з ПДВ", "с НДС", "в т.ч. ПДВ", "разом з ПДВ"); false for a price "без ПДВ" / "без НДС" or when VAT is not mentioned at all; null when `amount_per_t` is null.
- `currency`: the currency of the prices in the document (UAH, USD, EUR), even when the price is not per tonne.

**basis**: only an Incoterms term named in the document: EXW, FCA, CPT, DAP, FOB or DAT. Another named term (DDP, CIF, CFR, …) → "other". A delivery condition stated without a term ("за рахунок постачальника", "самовивіз зі складу", "ціна включає доставку") → "other". Only a delivery address → null. Never pick a term that the document does not name.

**delivery**
- `place`: from the clause about the place of delivery (see "Text values"). "До Замовника" without an address → null.
- `date_from`, `date_to`: the delivery period only; the validity term of the contract is not a delivery period. "у термін до 30.06" → only `date_to`; "з … по …" → both.
- `period_raw`: only when the delivery period is relative ("протягом 5 робочих днів після оплати"): copy it as written and leave both dates null.

**payment**: the agreed payment term. A term left for later ("по оплате договоримся отдельно", "порядок розрахунків обговоримо") is not a payment term, and neither is a sanction for late payment (that goes to `penalties`). If the document states no payment term, set `payment` to null.
- `days`: the number of days to pay. A fixed date instead ("до 15.09") → `days` null and `trigger` "other".
- `days_type`: calendar for "календарних днів" and its abbreviation "к.д."; banking for "банківських днів" and "б.д."; other for "робочих днів" and "р.д."; null when the kind of days is not said.
- `trigger`: what the term is counted from. A prepayment or an advance ("передоплата", "аванс", "предоплата") → prepayment, even when the days are counted from the invoice date. Payment after the goods are received ("після поставки", "після отримання товару", "за накладною") → after_delivery. Payment after documents ("після отримання документів", "з дати підписання накладних", "з дати отримання рахунку") → after_documents.
- `vat_after_registration`: true only if the document ties the payment of VAT to the registration of the tax invoice ("ПДВ після реєстрації ПН"); false if it says VAT is paid regardless of the registration; null if it says nothing about it. The seller's duty to register the invoice is not, by itself, such a tie.

**penalties**: one element per sanction. A fine and a daily penalty in one clause are two elements. No sanctions in the document → null. Damages without a stated size and the return of a prepayment with interest under force majeure are not sanctions.
- `payer`: the party that pays the sanction.
- `trigger`: late_delivery (delay of delivery), late_payment (delay of payment), under_delivery (non-delivery or short delivery); any other breach (quality, replacement of goods, refusal, tax violations, price change) → other.
- `unit` and `amount`: a one-time fine in percent ("штраф 20% від вартості партії", no "per day"), whatever the breach → one_time_pct, amount 20; percent per day ("0,1% від суми за кожен день прострочення") → pct_per_day, amount 0.1; a multiple of the NBU discount rate per day → nbu_rate_multiple, amount 1 for "облікової ставки НБУ" and 2 for "подвійної облікової ставки НБУ"; "% річних", a rate per hour or a fixed sum per tonne → other.
- `cap_pct`: only an explicitly written maximum ("але не більше 10%").

**quality**
- `grade`: the class as written, as a string: "3-го класу" / "3 кл" / "III" → "3"; also for corn and barley. Feed grade is a class too: "фуражна", "фураж", "фуражного зерна" → "фураж", even when the crop is not named. If the deal allows several classes ("2-го та/або 3-го класу"), set `grade` to null, and `price.amount_per_t` to null when the prices differ by class.
- `protein_min`, `moisture_max`, `foreign_matter_max`: only numbers the document states ("смітна", "сміттєва", "сорная домішка" → `foreign_matter_max`). A reference to a standard (ДСТУ) without numbers → null. With several classes, fill a limit only if it is the same for all of them.
- `lab`: whose laboratory determines the quality: the buyer's / seller's laboratory → buyer / seller; a state inspection or an independent testing centre → independent; the laboratory of the elevator or terminal at the place of delivery → other; only a producer's certificate → null.

**buyer**: the buyer's (customer's) company name from the preamble or the signature, without "в особі …". A person's name is not a buyer. A hidden name (empty quotes, a placeholder) → null.

# Before you answer

Go through every field you filled and check that you can point to the words in the document that state it; if you cannot, set it to null. Check that text values are in the language of the document and that every date has the right year.
