"""Text normalization shared by the content checks and the scoring: who is named, not how."""

import re

QUOTES = re.compile(r"[\"«»“”„]")
APOSTROPHES = re.compile(r"['’ʼ`]")  # inside a word: "П'ятихатки" = "Пʼятихатки"
LEGAL_FORMS = re.compile(
    r"\b(?:товариство з обмеженою відповідальністю|приватне акціонерне товариство"
    r"|публічне акціонерне товариство|акціонерне товариство|приватне підприємство"
    r"|державне підприємство|фермерське господарство|тзов|тов|прат|пат|ат|пп|дп|фг)\b"
)


def normalize_text(value: str) -> str:
    """Case, quotes, the legal form ("ТОВ", "ФГ"…) and spacing do not change who is named."""
    text = QUOTES.sub(" ", APOSTROPHES.sub("", value.casefold()))
    return " ".join(LEGAL_FORMS.sub(" ", text).split())
