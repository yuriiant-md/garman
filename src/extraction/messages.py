"""The user turn every extractor sends, so all models see the document wrapped the same way."""


def user_message(text: str) -> str:
    return f"Текст угоди:\n<text>\n{text}\n</text>"
