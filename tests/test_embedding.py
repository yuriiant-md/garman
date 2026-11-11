from collections.abc import Iterable

import numpy as np
import pytest
from numpy.typing import NDArray

from src.rag.embedding import LengthSortedEmbedder


class FakeModel:
    """Stands in for fastembed: a text's vector is (its length, 1.0); it records its calls."""

    def __init__(self, dimension: int = 2) -> None:
        self.dimension = dimension
        self.calls: list[tuple[list[str], int]] = []

    def embed(self, documents: Iterable[str], batch_size: int) -> Iterable[NDArray[np.float32]]:
        texts = list(documents)
        self.calls.append((texts, batch_size))
        return (np.array([len(text), 1.0][: self.dimension], dtype=np.float32) for text in texts)


def embedder(model: FakeModel, dimension: int = 2) -> LengthSortedEmbedder:
    return LengthSortedEmbedder(model=model, model_name="fake", dimension=dimension, batch_size=8)


def test_texts_reach_the_model_shortest_first() -> None:
    model = FakeModel()

    embedder(model).embed(["ccc", "a", "bb"])

    assert model.calls == [(["a", "bb", "ccc"], 8)]


def test_vectors_come_back_in_the_order_of_the_texts() -> None:
    vectors = embedder(FakeModel()).embed(["ccc", "a", "bb"])

    assert vectors == ((3.0, 1.0), (1.0, 1.0), (2.0, 1.0))


def test_no_texts_give_no_vectors() -> None:
    assert embedder(FakeModel()).embed([]) == ()


def test_a_vector_of_the_wrong_size_is_an_error() -> None:
    # an index built for 1024 numbers cannot take vectors of another size
    with pytest.raises(ValueError, match="1 numbers, expected 2"):
        embedder(FakeModel(dimension=1)).embed(["пеня"])


def test_progress_is_reported_after_every_batch() -> None:
    reported: list[tuple[int, int]] = []
    model = FakeModel()
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]

    LengthSortedEmbedder(model=model, model_name="fake", dimension=2, batch_size=2).embed(
        texts, on_progress=lambda done, total: reported.append((done, total))
    )

    assert reported == [(2, 5), (4, 5), (5, 5)]
