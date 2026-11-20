import asyncio
import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import anyio
import pymupdf
import pytest
from mcp.client.stdio import StdioServerParameters, get_default_environment
from mcp.server.mcpserver import MCPServer
from mcp.shared.exceptions import MCPError
from mcp.types import ListToolsResult, Tool

from src.agents.graph import Agent
from src.agents.limits import Limits, StopReason
from src.agents.mcp_tools import (
    LOOP_THREAD_NAME,
    McpConnection,
    McpTool,
    _all_tools,
    garman_servers,
    system_prompt,
)
from src.agents.tools import ToolRefused, load_tool_description
from src.documents import contract
from src.extraction.prompts import load_prompt
from src.servers import calculators_server, contract_access, contracts_server
from src.servers.calculators_server import FarmerSetup
from src.servers.contract_access import ContractShelf
from tests.agent_fixtures import FakeModel, ask, tool_results, turn
from tests.tools_fixtures import TODAY, make_playbook, make_rates

REPO = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO / "tools"
AGENT_PROMPT = REPO / "prompts" / "agent_v1.md"
CALL_TIMEOUT_S = 5
STDIO_TIMEOUT_S = 30  # two cold subprocess starts take ~1.5 s
CONTRACT_TOOLS = ("list_contracts", "contract_outline", "get_clause")
CALCULATOR_TOOLS = ("effective_price", "penalty_risk")
PAGES = (
    "ДОГОВІР ПОСТАВКИ № 1\n",
    "1. Предмет\n1.1. Продавець передає пшеницю 3 класу.\n1.2. Покупець приймає товар.\n",
    "2. Ціна\n2.1. Ціна 9 000 грн за тонну.\n2.2. Пеня 0,1% за кожен день прострочення.\n",
)
LIMITS = Limits(
    max_steps=5, max_tokens=10_000, max_cost_usd=1.0, tool_timeout_s=5, request_timeout_s=60
)


@pytest.fixture(autouse=True)
def fresh_cache() -> None:
    contract_access.clear_cache()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """farmer_a owns k_001, farmer_b owns k_900."""
    for farmer, contract_id in (("farmer_a", "k_001"), ("farmer_b", "k_900")):
        (tmp_path / farmer).mkdir()
        (tmp_path / farmer / f"{contract_id}.pdf").write_bytes(b"%PDF-1.7")
    return tmp_path


@pytest.fixture
def servers(root: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, MCPServer]:
    """Both servers of farmer_a, in memory: no subprocess, the PDF text is given."""
    monkeypatch.setattr(contract, "pdf_pages", lambda path: PAGES)
    setup = FarmerSetup(playbook=make_playbook(), rates=make_rates())
    return {
        "contracts": contracts_server.build_server(ContractShelf(root=root, farmer_id="farmer_a")),
        "calculators": calculators_server.build_server(setup, clock=lambda: TODAY),
    }


@pytest.fixture
def connection(servers: dict[str, MCPServer]) -> Iterator[McpConnection]:
    with McpConnection(servers, call_timeout_s=CALL_TIMEOUT_S) as opened:
        yield opened


def by_name(connection: McpConnection, name: str) -> McpTool:
    return next(tool for tool in connection.tools if tool.name == name)


def loop_threads() -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name == LOOP_THREAD_NAME]


def versioned_server(*, version: str | None) -> MCPServer:
    """A one-tool server; its tool sleeps as long as asked, to test the call timeout."""
    server = MCPServer("test-server", instructions="Тестовий сервер.")

    async def nap(seconds: float) -> str:
        await anyio.sleep(seconds)
        return "прокинувся"

    server.add_tool(nap, meta=None if version is None else {"version": version})
    return server


# --- what the agent gets from the servers


def test_every_tool_of_both_servers_comes_with_its_yaml_version_and_description(
    connection: McpConnection,
) -> None:
    assert sorted(tool.name for tool in connection.tools) == sorted(
        CONTRACT_TOOLS + CALCULATOR_TOOLS
    )
    for tool in connection.tools:
        yaml_description = load_tool_description(TOOLS_DIR / f"{tool.name}.yaml")
        assert tool.version == yaml_description.version
        assert tool.spec["name"] == tool.name
        assert tool.spec.get("description") == yaml_description.description


def test_the_argument_schema_is_the_servers_own(connection: McpConnection) -> None:
    schema = by_name(connection, "effective_price").spec["input_schema"]

    assert schema["required"] == ["terms"]  # type: ignore[index]
    assert "$defs" in schema  # PriceTerms with its nested models, as the server builds it


def test_the_instructions_of_each_server_are_kept_under_its_name(
    connection: McpConnection,
) -> None:
    assert connection.instructions == {
        "contracts": contracts_server.INSTRUCTIONS,
        "calculators": calculators_server.INSTRUCTIONS,
    }


# --- calls


def test_a_result_is_the_servers_structured_answer_as_json(connection: McpConnection) -> None:
    result = by_name(connection, "get_clause").run({"contract_id": "k_001", "clause": "2.2"})

    clause = json.loads(result)
    assert clause["number"] == "2.2"
    assert "Пеня 0,1%" in clause["text"]
    assert "\n" not in result  # compact: the model pays for every space


def test_a_calculator_answer_keeps_ukrainian_letters_readable(connection: McpConnection) -> None:
    result = by_name(connection, "effective_price").run(
        {
            "terms": {
                "crop": "wheat",
                "amount_per_t": 9000,
                "currency": "UAH",
                "place": "Невідоме село",
            }
        }
    )

    assert "\\u04" not in result  # ensure_ascii off: the model reads letters, not escapes
    assert json.loads(result)["missing"]  # a place without freight in the playbook


def test_another_farmers_contract_is_refused_with_the_servers_403_text(
    connection: McpConnection,
) -> None:
    with pytest.raises(ToolRefused, match="403 access denied: contract k_900 is not yours"):
        by_name(connection, "get_clause").run({"contract_id": "k_900", "clause": "2.2"})


def test_invalid_arguments_are_refused_by_the_server_with_the_field_named(
    connection: McpConnection,
) -> None:
    with pytest.raises(ToolRefused, match="contract_id"):
        by_name(connection, "get_clause").run({"contract_id": "../farmer_b/k_900", "clause": "1"})


def test_a_call_over_the_timeout_fails_instead_of_hanging() -> None:
    with McpConnection({"test": versioned_server(version="v1")}, call_timeout_s=0.2) as opened:
        started = time.perf_counter()
        with pytest.raises(MCPError):
            opened.tools[0].run({"seconds": 10})

    assert time.perf_counter() - started < 3


# --- the agent with MCP tools


def make_agent(model: FakeModel, connection: McpConnection) -> Agent:
    return Agent(
        model=model,
        tools=connection.tools,
        system=system_prompt("Ти помічник фермера.", connection.instructions),
        limits=LIMITS,
        prices=(1.0, 5.0),
        model_name="haiku-5.5",
    )


def test_the_agent_gets_a_403_as_an_error_result_and_the_run_goes_on(
    connection: McpConnection,
) -> None:
    model = FakeModel(
        turn("", ask("get_clause", contract_id="k_900", clause="2.2")),
        turn("Цей договір не ваш, я не можу його відкрити."),
    )

    trace = make_agent(model, connection).run("Яка пеня в договорі k_900?")

    call = trace.steps[0].tool_calls[0]
    assert call.tool_version == "v1"
    assert call.error is not None and "403" in call.error
    assert "403 access denied: contract k_900 is not yours" in call.error
    assert not call.error.startswith("ToolRefused")  # the server's text as is, not our wrapper
    result = tool_results(model, 1)[0]
    assert result["is_error"] is True
    assert trace.stop_reason == StopReason.ANSWERED


def test_the_calls_of_one_turn_run_in_parallel_and_all_come_back(
    connection: McpConnection,
) -> None:
    clauses = ("1.1", "1.2", "2.1", "2.2")
    model = FakeModel(
        turn(
            "",
            *(
                ask("get_clause", f"t{i}", contract_id="k_001", clause=clause)
                for i, clause in enumerate(clauses)
            ),
        ),
        turn("Готово."),
    )

    trace = make_agent(model, connection).run("Що в договорі k_001?")

    calls = trace.steps[0].tool_calls
    assert [json.loads(call.result or "{}")["number"] for call in calls] == list(clauses)


def test_the_model_sees_the_mcp_tools_and_the_server_instructions(
    connection: McpConnection,
) -> None:
    model = FakeModel(turn("ok"))

    make_agent(model, connection).run("Привіт")

    request = model.requests[0]
    assert {tool["name"] for tool in request["tools"]} == set(  # type: ignore[attr-defined]
        CONTRACT_TOOLS + CALCULATOR_TOOLS
    )
    assert "На 403 не шукай обхідних шляхів" in str(request["system"])


# --- start and stop


@pytest.mark.parametrize("version", [None, "v1 beta"])
def test_a_tool_without_a_valid_version_stops_the_start_and_leaves_no_thread(
    version: str | None,
) -> None:
    with pytest.raises(ValueError, match="nap.*version"):
        with McpConnection({"test": versioned_server(version=version)}, call_timeout_s=1):
            pass

    assert loop_threads() == []


def test_after_the_connection_closes_its_thread_is_gone_and_tools_refuse_to_run(
    servers: dict[str, MCPServer],
) -> None:
    with McpConnection(servers, call_timeout_s=CALL_TIMEOUT_S) as opened:
        tool = by_name(opened, "list_contracts")
        assert len(loop_threads()) == 1

    assert loop_threads() == []
    with pytest.raises(RuntimeError, match="closed"):
        tool.run({})


def test_a_connection_is_opened_once_even_after_it_closed() -> None:
    connection = McpConnection({"test": versioned_server(version="v1")}, call_timeout_s=1)
    with connection, pytest.raises(RuntimeError, match="opened once"):
        connection.__enter__()

    with pytest.raises(RuntimeError, match="opened once"):
        connection.__enter__()


def test_an_error_in_the_with_block_is_the_one_raised_and_the_servers_are_closed(
    servers: dict[str, MCPServer],
) -> None:
    with pytest.raises(KeyError, match="the agent failed"):
        with McpConnection(servers, call_timeout_s=CALL_TIMEOUT_S):
            raise KeyError("the agent failed")

    assert loop_threads() == []


def test_calls_of_one_turn_wait_for_each_other_only_as_long_as_the_slowest() -> None:
    naps = 4
    model = FakeModel(
        turn("", *(ask("nap", f"t{i}", seconds=0.5) for i in range(naps))), turn("Готово.")
    )
    with McpConnection({"test": versioned_server(version="v1")}, call_timeout_s=5) as opened:
        started = time.perf_counter()
        trace = make_agent(model, opened).run("Поспи")
        elapsed = time.perf_counter() - started

    results = [call.result for call in trace.steps[0].tool_calls]
    assert results == ['{"result":"прокинувся"}'] * naps
    assert elapsed < naps * 0.5 / 2  # one after another would take 2 s


# --- our servers as subprocesses


def test_garman_servers_name_the_farmer_and_never_pass_the_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secret")

    params = garman_servers("farmer_a", extra_env={"GARMAN_CONTRACTS_DIR": "/tmp/contracts"})

    assert "ANTHROPIC_API_KEY" not in get_default_environment()  # what the SDK adds to env
    assert set(params) == {"contracts", "calculators"}
    for server in params.values():
        assert isinstance(server, StdioServerParameters)
        assert server.env == {
            "GARMAN_FARMER_ID": "farmer_a",
            "GARMAN_CONTRACTS_DIR": "/tmp/contracts",
        }
        assert server.cwd == REPO


def test_both_servers_start_as_subprocesses_and_answer(root: Path) -> None:
    pdf = root / "farmer_a" / "k_001.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), "1.1. Price is fixed.\n1.2. Delivery by truck.")
        doc.save(pdf)
    params = garman_servers("farmer_a", extra_env={"GARMAN_CONTRACTS_DIR": str(root)})

    with McpConnection(
        params, call_timeout_s=STDIO_TIMEOUT_S, start_timeout_s=STDIO_TIMEOUT_S
    ) as opened:
        clause = json.loads(
            by_name(opened, "get_clause").run({"contract_id": "k_001", "clause": "1.2"})
        )
        with pytest.raises(ToolRefused, match="403"):
            by_name(opened, "get_clause").run({"contract_id": "k_900", "clause": "1.1"})

    assert "Delivery by truck" in clause["text"]


def test_a_server_that_cannot_start_stops_the_connection_with_an_error() -> None:
    broken = StdioServerParameters(
        command="python3", args=["-c", "import sys; sys.exit('no farmer')"], cwd=REPO
    )

    with pytest.raises(MCPError):
        with McpConnection({"broken": broken}, call_timeout_s=1, start_timeout_s=STDIO_TIMEOUT_S):
            pass

    assert loop_threads() == []


def test_a_server_that_never_answers_stops_the_start_with_its_name_and_no_process_left() -> None:
    silent = StdioServerParameters(command="sleep", args=["300"], cwd=REPO)

    started = time.perf_counter()
    with pytest.raises(TimeoutError, match="silent did not start in 1 s"):
        with McpConnection({"silent": silent}, call_timeout_s=1, start_timeout_s=1):
            pass

    assert time.perf_counter() - started < 10
    assert loop_threads() == []


# --- the system prompt


def test_the_system_prompt_adds_each_servers_instructions_under_its_name() -> None:
    system = system_prompt("Базові правила.", {"contracts": "Лише читання.", "calculators": ""})

    assert system.startswith("Базові правила.")
    assert "## MCP server contracts\nЛише читання." in system
    assert "calculators" not in system  # a server without instructions adds nothing


def test_the_agent_prompt_has_a_version_and_the_rule_for_a_refused_contract() -> None:
    prompt = load_prompt(AGENT_PROMPT)

    assert prompt.version == "v1"
    assert "403" in prompt.text


# --- a server that lists its tools in pages


class PagedClient:
    """Lists two tools on two pages, as a server with many tools may."""

    async def list_tools(self, *, cursor: str | None = None) -> ListToolsResult:
        if cursor is None:
            return ListToolsResult(tools=[server_tool("first")], next_cursor="page-2")
        assert cursor == "page-2"
        return ListToolsResult(tools=[server_tool("second")])


def server_tool(name: str) -> Tool:
    return Tool(name=name, input_schema={"type": "object"}, meta={"version": "v1"})


def test_every_page_of_a_servers_tool_list_is_read() -> None:
    tools = asyncio.run(_all_tools(PagedClient()))  # type: ignore[arg-type]

    assert [tool.name for tool in tools] == ["first", "second"]
