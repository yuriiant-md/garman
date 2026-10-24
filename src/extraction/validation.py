"""The Deal schema as the models see it, and the check of their raw JSON answers against it and
its content checks."""

from pydantic import ValidationError

from src.schemas.deal import Deal


def require_every_key(node: object) -> object:
    """Copy of a JSON schema in which every object lists all its properties as required."""
    if isinstance(node, list):
        return [require_every_key(item) for item in node]
    if not isinstance(node, dict):
        return node
    copy = {key: require_every_key(value) for key, value in node.items()}
    if "properties" in copy:
        copy["required"] = list(copy["properties"])
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
        return Deal.model_validate_json(raw), None
    except ValidationError as error:
        return None, str(error)
