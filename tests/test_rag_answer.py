import pytest
from pydantic import ValidationError

from src.rag.answer import (
    NOT_IN_DOCUMENTS_TEXT,
    Answer,
    AnswerStatus,
    Claim,
    answer_problem,
    citation,
    render,
    warning,
)
from src.rag.chunking import Chunk
from src.rag.context import document_block
from src.rag.sources import DocType


def chunk(chunk_id: str, short_title: str, **fields: object) -> Chunk:
    values: dict[str, object] = {
        "chunk_id": chunk_id,
        "doc_id": chunk_id.split("#")[0],
        "doc_type": DocType.CONTRACT,
        "title": short_title,
        "short_title": short_title,
        "text_sha256": "0" * 64,
        "text": "Текст.",
    }
    return Chunk.model_validate(values | fields)


CLAUSE = chunk("own_003#024", "own_003", ref="п. 7.2")
ARTICLE = chunk("law_civil_code#101", "ЦК", doc_type=DocType.LAW, in_force=True, ref="ст. 625")
OLD_ARTICLE = chunk(
    "law_commercial_code#005", "ГК", doc_type=DocType.LAW, in_force=False, ref="ст. 232"
)
OLD_PRACTICE = chunk(
    "court_910_14180_18#022",
    "ВС, справа № 910/14180/18",
    doc_type=DocType.COURT_DECISION,
    cites_commercial_code=True,
    section="ВСТАНОВИВ:",
)
CONTEXT = {"S1": CLAUSE, "S2": ARTICLE, "S3": OLD_ARTICLE, "S4": OLD_PRACTICE}


def answered(*claims: Claim) -> Answer:
    return Answer(status=AnswerStatus.ANSWERED, claims=claims)


def test_an_answer_needs_a_claim_and_every_claim_a_source() -> None:
    with pytest.raises(ValidationError):
        Answer(status=AnswerStatus.ANSWERED, claims=())
    with pytest.raises(ValidationError):
        Claim(text="Покупець сплачує пеню.", sources=())


def test_not_in_documents_carries_no_claims() -> None:
    with pytest.raises(ValidationError, match="no claims"):
        Answer(
            status=AnswerStatus.NOT_IN_DOCUMENTS,
            claims=(Claim(text="Пеня 0,1%.", sources=("S1",)),),
        )


def test_a_source_outside_the_context_is_a_problem() -> None:
    answer = answered(
        Claim(text="Пеня 0,1% за день.", sources=("S1",)),
        Claim(text="Три проценти річних.", sources=("S2", "S9")),
    )

    problem = answer_problem(answer, CONTEXT)

    assert problem is not None
    assert "S9" in problem
    assert answer_problem(answered(Claim(text="Пеня.", sources=("S1",))), CONTEXT) is None


def test_a_citation_is_the_short_title_and_the_ref_or_else_the_section() -> None:
    assert citation(CLAUSE) == "[own_003, п. 7.2]"
    assert citation(ARTICLE) == "[ЦК, ст. 625]"
    assert citation(OLD_PRACTICE) == "[ВС, справа № 910/14180/18, розділ «ВСТАНОВИВ»]"
    assert citation(chunk("own_005#001", "own_005")) == "[own_005]"


def test_a_repealed_law_and_practice_on_it_get_a_warning_and_the_rest_none() -> None:
    assert warning(CLAUSE) is None
    assert warning(ARTICLE) is None
    old_article = warning(OLD_ARTICLE)
    assert old_article is not None and "втратив чинність" in old_article
    old_practice = warning(OLD_PRACTICE)
    assert old_practice is not None and "практика за ГК" in old_practice


@pytest.mark.parametrize("cited", [CLAUSE, ARTICLE, OLD_ARTICLE, OLD_PRACTICE])
def test_the_warning_the_judge_reads_is_the_one_the_farmer_sees(cited: Chunk) -> None:
    # judge_faithful_v2 tells the judge: every «Увага» line of a cited document is shown to the
    # farmer below the answer. If the two ever part, that prompt tells the judge a lie.
    # Arrange
    caution = warning(cited)
    answer = answered(Claim(text="Твердження.", sources=("S1",)))

    # Act
    block = document_block("S1", cited)
    text = render(answer, {"S1": cited})

    # Assert
    if caution is None:
        assert "Увага:" not in block and "⚠" not in text
    else:
        assert f"\nУвага: {caution}\n" in block
        assert text.endswith(f"\n\n⚠ {citation(cited)}: {caution}")


def test_render_puts_citations_after_each_claim_and_each_warning_once() -> None:
    answer = answered(
        Claim(text="Покупець сплачує пеню 0,1% за день.", sources=("S1",)),
        Claim(text="Раніше штраф рахували за ГК.", sources=("S3", "S4")),
        Claim(text="Суд застосував ГК.", sources=("S4",)),
    )

    text = render(answer, CONTEXT)

    assert text.startswith("Покупець сплачує пеню 0,1% за день. [own_003, п. 7.2]\n")
    assert "Раніше штраф рахували за ГК. [ГК, ст. 232] [ВС, справа № 910/14180/18, " in text
    assert text.count("втратив чинність") == 2  # the article's warning and the practice's
    assert text.count("практика за ГК") == 1


def test_two_chunks_that_cite_alike_give_one_citation_and_one_warning() -> None:
    # two chunks of one decision without a ref both cite as [ВС, справа …, розділ «ВСТАНОВИВ»]
    other_part = OLD_PRACTICE.model_copy(update={"chunk_id": "court_910_14180_18#025"})
    context = {**CONTEXT, "S5": other_part}
    answer = answered(
        Claim(text="Суд застосував ГК.", sources=("S4", "S5")),
        Claim(text="І ще раз.", sources=("S5",)),
    )

    text = render(answer, context)
    cited = "[ВС, справа № 910/14180/18, розділ «ВСТАНОВИВ»]"

    assert text.splitlines()[0] == f"Суд застосував ГК. {cited}"
    assert text.count(cited) == 3  # once per claim, once in the warning
    assert text.count("практика за ГК") == 1


def test_render_of_not_in_documents_says_so() -> None:
    answer = Answer(status=AnswerStatus.NOT_IN_DOCUMENTS)

    assert render(answer, CONTEXT) == NOT_IN_DOCUMENTS_TEXT
