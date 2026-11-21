"""Agent tools: a versioned description in tools/<name>.yaml, a Pydantic model of the arguments
and a plain Python function.

The description is a prompt: it decides when the model calls the tool, so it lives outside the
code with a version, like prompts/ (a change → a new version and a line in tools/CHANGELOG.md).
The argument schema the model sees is generated from the Pydantic model, so the two cannot drift.

The answer is checked too, by a Pydantic model of the output (check_output, called for every
attempt in faults.run_guarded): an answer off its schema is an error, never a silent gap.

The model's arguments are validated before the function runs. Invalid ones raise
InvalidArguments with every bad field named; the agent sends that text back to the model as the
tool's error (graph.py), and the model can fix the call. The function itself never sees them.

The graph needs only what AgentTool names, so a tool may also live on an MCP server
(mcp_tools.py): there the server holds the description and checks the arguments.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml
from anthropic.types import ToolParam
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictBool, ValidationError

from src.extraction.prompts import VERSION_PATTERN

TOOL_NAME_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"  # what the Anthropic API allows as a tool name
MAX_PROBLEMS = 10  # named in one error; a list of a thousand bad items must not fill the context


class ToolArgs(BaseModel):
    """Base of every tool's arguments: an argument the tool does not know is an error too."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ToolDescription(BaseModel):
    """tools/<name>.yaml: what the model reads about a tool, and how the agent may run it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=TOOL_NAME_PATTERN)
    version: str = Field(pattern=f"^{VERSION_PATTERN.pattern}$")
    # no default and no "false" as text: neither may make a tool look safe to retry (#19)
    side_effect: StrictBool
    description: str = Field(min_length=1)  # when to call, when not, arguments, what it returns


class InvalidArguments(ValueError):
    """The model called a tool with arguments its schema does not allow."""


class ToolRefused(Exception):
    """The tool said no (403, not found, bad arguments): its text is for the model, as is."""


class SchemaMismatch(Exception):
    """The tool's answer does not match its output schema: the tool changed, not the question."""


class AgentTool(Protocol):
    """What the agent needs of a tool, wherever the tool runs."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def spec(self) -> ToolParam: ...

    @property
    def side_effect(self) -> bool: ...  # True: never retried automatically (faults.py)

    def run(self, arguments: Mapping[str, JsonValue]) -> str: ...

    def check_output(self, text: str) -> None: ...  # raises SchemaMismatch


@dataclass(frozen=True)
class Tool[A: ToolArgs]:
    description: ToolDescription
    args: type[A]
    func: Callable[[A], str]
    output: type[BaseModel]  # the shape of the answer, checked for every answer

    @property
    def name(self) -> str:
        return self.description.name

    @property
    def version(self) -> str:
        return self.description.version

    @property
    def side_effect(self) -> bool:
        return self.description.side_effect

    @property
    def spec(self) -> ToolParam:
        """The tool as the model sees it."""
        return {
            "name": self.name,
            "description": self.description.description,
            "input_schema": self.args.model_json_schema(),
        }

    def run(self, arguments: Mapping[str, JsonValue]) -> str:
        try:
            parsed = self.args.model_validate(arguments)
        except ValidationError as error:
            raise InvalidArguments(
                f"invalid arguments for {self.name}: {_problems(error)}"
            ) from None
        return self.func(parsed)

    def check_output(self, text: str) -> None:
        try:
            self.output.model_validate_json(text)
        except ValidationError as error:
            raise SchemaMismatch(
                f"{self.name} answered off its output schema: {_problems(error)}"
            ) from None


def load_tool_description(path: Path) -> ToolDescription:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: a tool description must be a YAML mapping")
    try:
        description = ToolDescription.model_validate(data)
    except ValidationError as error:
        raise ValueError(f"{path}: {_problems(error)}") from None
    if description.name != path.stem:  # one file per tool, found by its name
        raise ValueError(f"{path}: name {description.name!r} differs from the file name")
    return description


def load_tool[A: ToolArgs](
    path: Path, args: type[A], func: Callable[[A], str], *, output: type[BaseModel]
) -> Tool[A]:
    return Tool(description=load_tool_description(path), args=args, func=func, output=output)


def _problems(error: ValidationError) -> str:
    """The invalid fields with their reasons, short enough for the model to act on."""
    problems = error.errors()
    named = "; ".join(
        f"{'.'.join(str(part) for part in problem['loc']) or 'arguments'}: {problem['msg']}"
        for problem in problems[:MAX_PROBLEMS]
    )
    rest = len(problems) - MAX_PROBLEMS
    return f"{named}; ... and {rest} more" if rest > 0 else named
