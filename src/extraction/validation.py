"""Validate a model's raw JSON answer against the Deal schema and its content checks."""

from pydantic import ValidationError

from src.schemas.deal import Deal


def validate_deal(raw: str) -> tuple[Deal | None, str | None]:
    try:
        return Deal.model_validate_json(raw), None
    except ValidationError as error:
        return None, str(error)
