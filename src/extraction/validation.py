"""Validate a model's raw JSON answer against the Deal schema and its content checks."""

from pydantic import ValidationError

from src.schemas.deal import Deal


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
