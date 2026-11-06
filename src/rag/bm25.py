"""BM25 word weights for Ukrainian text, kept in Qdrant as a sparse vector.

The BM25 score of a chunk for a query is the sum, over the query's words, of IDF(word) times the
term part TF(word, chunk). The term part depends only on the chunk, so it is computed here once,
when the chunk is indexed. IDF depends on the whole corpus: Qdrant computes it itself from the
indexed vectors (a sparse vector with `Modifier.IDF`), so a new document never means re-weighting
the others. A query vector therefore carries weight 1 per word, and Qdrant multiplies it by IDF.
"""

import re
import zlib
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

K1 = 1.2  # how fast a repeated word stops adding weight (the usual BM25 value)
B = 0.75  # how much a longer chunk lowers the weight of each of its words (the usual value)

APOSTROPHE = "'"
_APOSTROPHES = re.compile(r"['’`ʼ]")  # the corpus spells the apostrophe in all these ways
# a number with inner dots or commas ("0,1", "7.3", "10.6.1") or a word with inner apostrophes
_TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+(?:'[^\W\d_]+)*")


@dataclass(frozen=True)
class TermWeights:
    """A sparse vector: token ids and their weights, in the same order."""

    indices: tuple[int, ...]
    values: tuple[float, ...]


def unify_apostrophes(text: str) -> str:
    return _APOSTROPHES.sub(APOSTROPHE, text)


def tokenize(text: str) -> tuple[str, ...]:
    """Lowercase words and numbers; every apostrophe becomes one, a hyphen splits a word."""
    return tuple(_TOKEN.findall(unify_apostrophes(text.lower())))


def token_id(token: str) -> int:
    """crc32 of the token's UTF-8 bytes: the same in every run, unlike Python's hash().

    Two different words can share an id: for the corpus's 12 770 words the chance that any pair
    does is about 2%. Their counts then add up, which BM25 survives.
    """
    return zlib.crc32(token.encode("utf-8"))


def average_length(token_lists: Iterable[Sequence[str]]) -> float:
    lengths = [len(tokens) for tokens in token_lists]
    if sum(lengths) == 0:
        raise ValueError("the corpus has no tokens, so BM25 has no average length")
    return sum(lengths) / len(lengths)


def document_weights(
    tokens: Sequence[str], *, average: float, k1: float = K1, b: float = B
) -> TermWeights:
    """The BM25 term part of each word: tf × (k1 + 1) / (tf + k1 × (1 − b + b × len / average))."""
    length_factor = k1 * (1 - b + b * len(tokens) / average)
    counts = Counter(token_id(token) for token in tokens)
    return TermWeights(
        indices=tuple(counts),
        values=tuple(tf * (k1 + 1) / (tf + length_factor) for tf in counts.values()),
    )


def query_weights(tokens: Sequence[str]) -> TermWeights:
    """Each query word once, with weight 1: Qdrant multiplies it by the word's IDF."""
    indices = tuple(dict.fromkeys(token_id(token) for token in tokens))
    return TermWeights(indices=indices, values=(1.0,) * len(indices))
