import json

from src.extraction.validation import repair_request, validate_deal


def test_valid_json_gives_a_deal_and_no_error():
    deal, error = validate_deal(json.dumps({"quantity_t": 500}))

    assert deal.quantity_t == 500
    assert error is None


def test_schema_violation_gives_no_deal_and_the_reason():
    deal, error = validate_deal(json.dumps({"quantity_t": "a lot"}))

    assert deal is None
    assert "quantity_t" in error


LATE_PAYMENT_BY_SELLER = {
    "payer": "seller", "trigger": "late_payment", "amount": 2, "unit": "nbu_rate_multiple",
}


def test_content_error_gives_no_deal_and_every_message_for_the_model():
    raw = json.dumps({"quantity_t": 400000, "penalties": [LATE_PAYMENT_BY_SELLER]})

    deal, error = validate_deal(raw)

    assert deal is None
    assert "penalties[0].payer:" in error
    assert "quantity_t:" in error


def test_placeholder_buyer_becomes_null_without_an_error():
    deal, error = validate_deal(json.dumps({"crop": "wheat", "buyer": "ТОВ “”"}))

    assert (deal.crop.value, deal.buyer) == ("wheat", None)
    assert error is None


def test_repair_request_quotes_the_error_and_asks_for_the_whole_json_again():
    request = repair_request("delivery.date_from: must not be after date_to")

    assert "delivery.date_from: must not be after date_to" in request
    assert "JSON" in request
