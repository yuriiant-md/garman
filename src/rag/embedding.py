"""Dense vectors for chunks and queries: bge-m3, run on this machine by fastembed (ONNX, no torch).

fastembed does not list bge-m3, so bge_m3() registers the ONNX export from BAAI's own Hugging Face
repo. It gives the same vectors as Ollama's bge-m3 (cosine 0.9999 on corpus chunks). bge-m3 takes
up to 8192 tokens and needs no "query:"/"passage:" prefix, so chunks and queries go in as they are.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from numpy.typing import NDArray

MODEL_NAME = "BAAI/bge-m3"
DIMENSION = 1024
BATCH_SIZE = 16
CACHE_DIR = Path.home() / ".cache" / "fastembed"  # the 2.3 GB model is downloaded here once

type Vector = tuple[float, ...]
type Progress = Callable[[int, int], None]  # (texts embedded so far, texts in all)


class Embedder(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed(
        self, texts: Sequence[str], on_progress: Progress | None = None
    ) -> tuple[Vector, ...]: ...


class TextModel(Protocol):
    """The part of fastembed's TextEmbedding this module uses."""

    def embed(self, documents: Iterable[str], batch_size: int) -> Iterable[NDArray[Any]]: ...


@dataclass(frozen=True)
class LengthSortedEmbedder:
    """Feeds the model texts of similar length together, then returns vectors in the input order.

    A batch is padded to its longest text, so a batch of one long and fifteen short chunks costs
    as much as sixteen long ones. Sorted by length, the corpus embeds three times faster on CPU.
    """

    model: TextModel
    model_name: str
    dimension: int
    batch_size: int = BATCH_SIZE

    def embed(
        self, texts: Sequence[str], on_progress: Progress | None = None
    ) -> tuple[Vector, ...]:
        """Vectors in the order of `texts`; on_progress hears after each batch how far it got."""
        order = sorted(range(len(texts)), key=lambda position: len(texts[position]))
        vectors = self.model.embed((texts[position] for position in order), self.batch_size)
        by_position: dict[int, Vector] = {}
        for done, (position, vector) in enumerate(zip(order, vectors, strict=True), start=1):
            by_position[position] = self._checked(vector)
            if on_progress and (done % self.batch_size == 0 or done == len(texts)):
                on_progress(done, len(texts))
        return tuple(by_position[position] for position in range(len(texts)))

    def _checked(self, vector: NDArray[Any]) -> Vector:
        if len(vector) != self.dimension:
            raise ValueError(
                f"{self.model_name} gave a vector of {len(vector)} numbers, "
                f"expected {self.dimension}"
            )
        return tuple(float(number) for number in vector)


def bge_m3(cache_dir: Path = CACHE_DIR, batch_size: int = BATCH_SIZE) -> LengthSortedEmbedder:
    """bge-m3 on CPU; the first call downloads the model into cache_dir."""
    # imported here: fastembed loads onnxruntime, which only a real model needs
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    known = {model["model"].lower() for model in TextEmbedding.list_supported_models()}
    if MODEL_NAME.lower() not in known:  # registering the same name twice is an error
        TextEmbedding.add_custom_model(
            model=MODEL_NAME,
            pooling=PoolingType.CLS,  # bge models take the first token's state, then normalise
            normalization=True,
            sources=ModelSource(hf=MODEL_NAME),
            dim=DIMENSION,
            model_file="onnx/model.onnx",
            additional_files=["onnx/model.onnx_data", "onnx/Constant_7_attr__value"],
            license="mit",
            size_in_gb=2.27,
        )
    model = TextEmbedding(MODEL_NAME, cache_dir=str(cache_dir))
    return LengthSortedEmbedder(
        model=model, model_name=MODEL_NAME, dimension=DIMENSION, batch_size=batch_size
    )
