"""The Deal schema as the models see it, and the check of their raw JSON answers: the schema
first, then the content checks of checks.py."""

from pydantic import ValidationError

from src.extraction.checks import clean_buyer, content_errors
from src.schemas.deal import Deal


def require_every_key(node: object) -> object:
    """Copy of a JSON schema in which every object lists all its properties as required."""
    if isinstance(node, list):
        return [require_every_key(item) for item in node]
    if not isinstance(node, dict):
        return node
    copy = {key: require_every_key(value) for key, value in node.items()}
    properties = copy.get("properties")
    if isinstance(properties, dict):
        copy["required"] = list(properties)
    return copy


def repair_request(error: str) -> str:
    """The message that asks a model to fix its one invalid answer."""
    return (
        "Your previous answer did not pass validation:\n"
        f"{error}\n"
        "Return the complete corrected JSON for the same document. "
        "Fix only what the error names; keep every other field as it was."
    )


def validate_deal(raw: str) -> tuple[Deal | None, str | None]:
    try:
        deal = clean_buyer(Deal.model_validate_json(raw))
    except ValidationError as error:
        return None, str(error)
    errors = content_errors(deal)
    if errors:
        return None, "\n".join(errors)
    return deal, None
