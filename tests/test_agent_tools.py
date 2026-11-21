from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from src.agents.tools import (
    InvalidArguments,
    SchemaMismatch,
    Tool,
    ToolArgs,
    ToolDescription,
    load_tool,
    load_tool_description,
)

DESCRIPTION = """\
name: search_corpus
version: v1
side_effect: false
description: |
  Коли викликати: питання про умови договору.
  Коли не викликати: розрахунок ціни.
  Аргументи: query, текст запиту; limit, скільки фрагментів повернути.
  Повертає: фрагменти з джерелом і пунктом.
"""


class SearchArgs(ToolArgs):
    query: str = Field(min_length=1, description="текст запиту")
    limit: int = Field(default=3, ge=1, le=10)


class Fragments(BaseModel):
    """The output schema of the test tool."""

    model_config = ConfigDict(extra="forbid")

    fragments: list[str]
    total: int


def write(tmp_path: Path, text: str = DESCRIPTION, name: str = "search_corpus") -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def make_tool(tmp_path: Path, calls: list[SearchArgs] | None = None) -> Tool[SearchArgs]:
    def search(args: SearchArgs) -> str:
        if calls is not None:
            calls.append(args)
        return f"знайдено {args.limit}: {args.query}"

    return load_tool(write(tmp_path), SearchArgs, search, output=Fragments)


# --- descriptions


def test_loads_description_from_yaml(tmp_path: Path) -> None:
    description = load_tool_description(write(tmp_path))

    assert description.name == "search_corpus"
    assert description.version == "v1"
    assert description.side_effect is False
    assert description.description.startswith("Коли викликати:")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (DESCRIPTION.replace("version: v1\n", ""), "version"),
        (DESCRIPTION.replace("version: v1", "version: v 1"), "version"),
        (DESCRIPTION.replace("side_effect: false\n", ""), "side_effect"),
        (DESCRIPTION.replace("side_effect: false", 'side_effect: "false"'), "side_effect"),
        (DESCRIPTION.split("description:")[0] + "description: ''\n", "description"),
        (DESCRIPTION + "retries: 3\n", "retries"),
        ("- just\n- a list\n", "mapping"),
    ],
    ids=[
        "no-version",
        "bad-version",
        "no-side-effect",
        "side-effect-as-text",
        "empty-description",
        "unknown-key",
        "list",
    ],
)
def test_rejects_broken_description(tmp_path: Path, text: str, message: str) -> None:
    path = write(tmp_path, text)

    with pytest.raises(ValueError, match=message) as error:
        load_tool_description(path)

    assert str(path) in str(error.value)


def test_rejects_name_the_api_does_not_allow(tmp_path: Path) -> None:
    text = DESCRIPTION.replace("name: search_corpus", "name: search corpus")
    path = write(tmp_path, text, name="search corpus")  # file and name agree: only the API refuses

    with pytest.raises(ValueError, match="name: String should match pattern"):
        load_tool_description(path)


def test_rejects_name_that_differs_from_file_name(tmp_path: Path) -> None:
    path = write(tmp_path, name="search")

    with pytest.raises(ValueError, match="search_corpus"):
        load_tool_description(path)


# --- spec for the model


def test_spec_takes_name_and_description_from_yaml(tmp_path: Path) -> None:
    tool = make_tool(tmp_path)

    assert tool.name == "search_corpus"
    assert tool.version == "v1"
    assert tool.spec["name"] == "search_corpus"
    assert tool.spec["description"] == load_tool_description(write(tmp_path)).description


def test_spec_input_schema_comes_from_args_model(tmp_path: Path) -> None:
    schema = SearchArgs.model_json_schema()

    assert make_tool(tmp_path).spec["input_schema"] == schema
    assert set(schema["properties"]) == {"query", "limit"}
    assert schema["required"] == ["query"]
    assert schema["additionalProperties"] is False  # extra="forbid" tells the model too


# --- running


def test_valid_arguments_reach_the_function_as_a_model(tmp_path: Path) -> None:
    calls: list[SearchArgs] = []
    tool = make_tool(tmp_path, calls)

    result = tool.run({"query": "строк оплати", "limit": 2})

    assert result == "знайдено 2: строк оплати"
    assert calls == [SearchArgs(query="строк оплати", limit=2)]


def test_defaults_fill_missing_optional_arguments(tmp_path: Path) -> None:
    result = make_tool(tmp_path).run({"query": "базис"})

    assert result == "знайдено 3: базис"


@pytest.mark.parametrize(
    ("arguments", "field"),
    [
        ({}, "query"),
        ({"query": ""}, "query"),
        ({"query": "базис", "limit": "п'ять"}, "limit"),
        ({"query": "базис", "limit": 50}, "limit"),
        ({"query": "базис", "contract": "own_003"}, "contract"),
    ],
    ids=["missing", "empty", "wrong-type", "out-of-range", "unknown"],
)
def test_invalid_arguments_never_reach_the_function(
    tmp_path: Path, arguments: dict[str, object], field: str
) -> None:
    calls: list[SearchArgs] = []
    tool = make_tool(tmp_path, calls)

    with pytest.raises(InvalidArguments) as error:
        tool.run(arguments)  # type: ignore[arg-type]

    assert calls == []
    assert f"{field}:" in str(error.value)


def test_invalid_arguments_message_names_the_tool_and_every_bad_field(tmp_path: Path) -> None:
    tool = make_tool(tmp_path)

    with pytest.raises(InvalidArguments) as error:
        tool.run({"limit": 0, "extra": True})

    message = str(error.value)
    assert message.startswith("invalid arguments for search_corpus")
    assert all(f"{field}:" in message for field in ("query", "limit", "extra"))


class Item(ToolArgs):
    tonnes: float = Field(gt=0)


class ItemsArgs(ToolArgs):
    items: list[Item]


def items_tool() -> Tool[ItemsArgs]:
    description = ToolDescription(
        name="sum_tonnes", version="v1", side_effect=False, description="Сума тонн."
    )
    return Tool(description=description, args=ItemsArgs, func=lambda args: "ok", output=Fragments)


def test_a_bad_field_inside_a_list_is_named_by_its_path() -> None:
    with pytest.raises(InvalidArguments) as error:
        items_tool().run({"items": [{"tonnes": 5}, {"tonnes": "багато"}]})

    assert "items.1.tonnes:" in str(error.value)


def test_the_message_names_at_most_ten_problems_and_counts_the_rest() -> None:
    bad_items: list[JsonValue] = [{"tonnes": -1}] * 50

    with pytest.raises(InvalidArguments) as error:
        items_tool().run({"items": bad_items})

    message = str(error.value)
    assert message.count("items.") == 10
    assert message.endswith("... and 40 more")


# --- the answer's schema and the side effect


def test_an_answer_of_the_output_schema_passes(tmp_path: Path) -> None:
    make_tool(tmp_path).check_output('{"fragments": ["п. 7.3"], "total": 1}')


@pytest.mark.parametrize(
    ("answer", "named"),
    [
        ('{"fragments": ["п. 7.3"]}', "total:"),
        ('{"fragments_v2": [], "total": 0}', "fragments:"),
        ('{"fragments": [], "total": "one"}', "total:"),
        ("знайдено", "Invalid JSON"),
    ],
    ids=["missing", "renamed", "wrong-type", "not-json"],
)
def test_an_answer_off_the_output_schema_is_refused_with_the_field_named(
    tmp_path: Path, answer: str, named: str
) -> None:
    with pytest.raises(SchemaMismatch) as error:
        make_tool(tmp_path).check_output(answer)

    message = str(error.value)
    assert message.startswith("search_corpus answered off its output schema:")
    assert named in message


@pytest.mark.parametrize("side_effect", [True, False])
def test_the_side_effect_comes_from_the_description(tmp_path: Path, side_effect: bool) -> None:
    text = DESCRIPTION.replace("side_effect: false", f"side_effect: {str(side_effect).lower()}")
    tool = load_tool(write(tmp_path, text), SearchArgs, lambda args: "", output=Fragments)

    assert tool.side_effect is side_effect
