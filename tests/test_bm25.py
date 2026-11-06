import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.rag.bm25 import (
    TermWeights,
    average_length,
    document_weights,
    query_weights,
    token_id,
    tokenize,
)


def weight_of(weights: TermWeights, token: str) -> float:
    return dict(zip(weights.indices, weights.values, strict=True))[token_id(token)]


def test_tokenize_lowercases_and_drops_punctuation() -> None:
    assert tokenize("Покупець СПЛАЧУЄ пеню!") == ("покупець", "сплачує", "пеню")


def test_tokenize_gives_one_token_for_every_kind_of_apostrophe() -> None:
    spellings = ("зобов'язання", "зобов’язання", "зобов`язання", "зобовʼязання")

    tokens = {tokenize(spelling) for spelling in spellings}

    assert tokens == {("зобов'язання",)}


def test_tokenize_keeps_decimal_and_clause_numbers_whole() -> None:
    assert tokenize("Пеня 0,1% за п. 7.3. і п.10.6.1") == (
        "пеня", "0,1", "за", "п", "7.3", "і", "п", "10.6.1",
    )  # fmt: skip


def test_tokenize_splits_hyphenated_words() -> None:
    assert tokenize("форс-мажор") == ("форс", "мажор")


def test_token_id_is_the_same_under_any_hash_seed() -> None:
    # Python's own hash() of a str changes between runs; an index built once must still match
    script = "from src.rag.bm25 import token_id\nprint(token_id('неустойка'))\n"
    root = Path(__file__).resolve().parents[1]

    ids = {
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=root,
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in range(4)
    }

    assert len(ids) == 1


def test_a_repeated_word_weighs_more_but_saturates() -> None:
    once = document_weights(("пеня", "ціна"), average=2.0)
    twice = document_weights(("пеня", "пеня"), average=2.0)
    often = document_weights(("пеня",) * 10 + ("ціна",) * 10, average=20.0)

    assert weight_of(once, "пеня") < weight_of(twice, "пеня") < weight_of(often, "пеня")
    assert weight_of(often, "пеня") < 10 * weight_of(once, "пеня")
    assert weight_of(often, "пеня") < 1.2 + 1  # k1 + 1 is the ceiling of the term part


def test_the_same_word_weighs_less_in_a_longer_text() -> None:
    short = document_weights(("пеня", "ціна"), average=4.0)
    long = document_weights(("пеня", "ціна", "строк", "якість", "вага", "оплата"), average=4.0)

    assert weight_of(long, "пеня") < weight_of(short, "пеня")


def test_document_weights_hold_each_token_once() -> None:
    weights = document_weights(("пеня", "ціна", "пеня"), average=3.0)

    assert sorted(weights.indices) == sorted({token_id("пеня"), token_id("ціна")})
    assert len(weights.indices) == len(weights.values)


def test_an_empty_text_has_no_weights() -> None:
    assert document_weights((), average=3.0) == TermWeights(indices=(), values=())


def test_query_weights_count_each_token_once_with_weight_one() -> None:
    weights = query_weights(("пеня", "за", "пеня"))

    assert sorted(weights.indices) == sorted({token_id("пеня"), token_id("за")})
    assert weights.values == (1.0, 1.0)


def test_average_length_is_the_mean_number_of_tokens() -> None:
    assert average_length((("а", "б"), ("в", "г", "ґ", "д"))) == 3.0


def test_average_length_of_an_empty_corpus_is_an_error() -> None:
    with pytest.raises(ValueError, match="no tokens"):
        average_length(((), ()))
