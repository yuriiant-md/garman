import json
import shutil
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent

from src.servers.calculators_server import (
    FARMER_ENV,
    PLAYBOOKS_DIR_ENV,
    RATES_PATH_ENV,
    TOOLS_DIR,
    FarmerSetup,
    build_server,
    main,
    setup_from_env,
)
from src.tools.playbook import PLAYBOOKS_DIR, RATES_PATH
from src.tools.price import PriceTerms, effective_price
from tests.tools_fixtures import TODAY, make_playbook, make_rates

pytestmark = pytest.mark.anyio

REPO = Path(__file__).resolve().parents[1]
STDIO_TIMEOUT_S = 30  # a cold start of the subprocess takes ~2 s
TOOL_NAMES = ["effective_price", "penalty_risk"]
WAREHOUSE = "Склад, с. Зелене"
SETUP = FarmerSetup(playbook=make_playbook(), rates=make_rates())


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def server() -> MCPServer:
    return build_server(SETUP, clock=lambda: TODAY)


def terms(**changes: Any) -> dict[str, Any]:
    """10 000 UAH/t without VAT at the farmer's warehouse, prepaid, as JSON from the model."""
    values: dict[str, Any] = {
        "crop": "wheat",
        "amount_per_t": 10000,
        "currency": "UAH",
        "vat_included": False,
        "basis": "FCA",
        "place": WAREHOUSE,
        "trigger": "prepayment",
    }
    return values | changes


def case(**changes: Any) -> dict[str, Any]:
    """0.5 % a day on 1 000 000 UAH for 10 days."""
    values: dict[str, Any] = {"sum_uah": 1000000, "days": 10, "amount": 0.5, "unit": "pct_per_day"}
    return values | changes


async def call(server: MCPServer, tool: str, arguments: dict[str, Any]) -> CallToolResult:
    async with Client(server) as client:
        return await client.call_tool(tool, arguments)


def answer(result: CallToolResult) -> dict[str, Any]:
    assert not result.is_error, result.content
    assert result.structured_content is not None
    return result.structured_content


def error_text(result: CallToolResult) -> str:
    assert result.is_error
    content = result.content[0]
    assert isinstance(content, TextContent)
    return content.text


# --- what a client sees


async def test_the_tools_come_with_their_yaml_description_version_and_read_only_hint(
    server: MCPServer,
) -> None:
    async with Client(server) as client:
        tools = (await client.list_tools()).tools

    assert [tool.name for tool in tools] == TOOL_NAMES
    for tool in tools:
        yaml_text = (TOOLS_DIR / f"{tool.name}.yaml").read_text(encoding="utf-8")
        assert tool.description is not None and tool.description.splitlines()[0] in yaml_text
        assert tool.meta == {"version": "v1"}
        assert tool.annotations is not None
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.open_world_hint is False


async def test_the_model_can_name_neither_the_farmer_nor_the_day(server: MCPServer) -> None:
    async with Client(server) as client:
        tools = (await client.list_tools()).tools

    for tool in tools:
        schema = json.dumps(tool.input_schema)
        assert "farmer" not in schema and "today" not in schema


async def test_the_argument_schema_refuses_extra_fields(server: MCPServer) -> None:
    async with Client(server) as client:
        tools = (await client.list_tools()).tools

    defs = {name: d for tool in tools for name, d in tool.input_schema["$defs"].items()}
    assert defs["PriceTerms"]["additionalProperties"] is False
    assert defs["PenaltyCase"]["additionalProperties"] is False



async def test_the_output_schema_announces_every_field_of_the_answer(server: MCPServer) -> None:
    async with Client(server) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    schema = tools["effective_price"].output_schema
    assert schema is not None  # a client checks the answer against it: a field it lacks fails
    assert {"uah_per_t", "steps", "missing", "stale"} <= set(schema["properties"])


# --- effective price


async def test_cpt_odesa_minus_freight_comes_back_with_each_step_and_its_source(
    server: MCPServer,
) -> None:
    arguments = {"terms": terms(amount_per_t=9200, basis="CPT", place="Одеса")}

    result = answer(await call(server, "effective_price", arguments))

    assert Decimal(result["uah_per_t"]) == Decimal("8450.00")  # 9 200 − 750 freight
    assert [Decimal(step["uah_per_t"]) for step in result["steps"]] == [
        Decimal("9200.00"),
        Decimal("-750.00"),
    ]
    assert all(step["source"] for step in result["steps"])
    assert result["missing"] == []


async def test_waiting_for_the_money_costs_the_farmers_rate(server: MCPServer) -> None:
    arguments = {"terms": terms(trigger="after_delivery", payment_days=30)}

    result = answer(await call(server, "effective_price", arguments))

    waiting = Decimal(result["steps"][-1]["uah_per_t"])
    assert waiting == Decimal("-147.95")  # 10 000 × 18 % × 30 / 365
    assert Decimal(result["uah_per_t"]) == Decimal("9852.05")


async def test_what_the_text_does_not_say_comes_back_as_missing_not_as_an_error(
    server: MCPServer,
) -> None:
    arguments = {"terms": terms(vat_included=None, place="Херсон")}

    result = answer(await call(server, "effective_price", arguments))

    assert result["uah_per_t"] is None
    assert result["steps"] == []
    assert len(result["missing"]) == 2
    assert any("ПДВ" in gap for gap in result["missing"])
    assert any("Херсон" in gap and "Одеса" in gap for gap in result["missing"])


async def test_the_day_comes_from_the_servers_clock(server: MCPServer) -> None:
    arguments = {"terms": terms(amount_per_t=200, currency="USD")}  # rate of 15.11.2026
    later = build_server(SETUP, clock=lambda: date(2027, 1, 1))

    fresh = answer(await call(server, "effective_price", arguments))
    old = answer(await call(later, "effective_price", arguments))

    assert Decimal(fresh["uah_per_t"]) == Decimal(old["uah_per_t"]) == Decimal("8300.00")
    assert (fresh["stale"], old["stale"]) == (False, True)


# --- penalty risk


async def test_a_penalty_by_the_day_comes_back_with_its_formula(server: MCPServer) -> None:
    result = answer(await call(server, "penalty_risk", {"case": case()}))

    assert Decimal(result["uah"]) == Decimal("50000.00")  # 1 000 000 × 0.5 % × 10
    assert "50000.00" in result["formula"]
    assert result["source"] == "умови договору"
    assert result["above_threshold"] is True  # 0.5 % a day > the farmer's 0.1 %


async def test_a_capped_penalty_stops_at_the_cap(server: MCPServer) -> None:
    result = answer(await call(server, "penalty_risk", {"case": case(cap_pct=2)}))

    assert Decimal(result["uah"]) == Decimal("20000.00")  # 2 % of 1 000 000
    assert result["capped"] is True


async def test_a_penalty_by_the_nbu_rate_names_the_rate_and_its_day(server: MCPServer) -> None:
    arguments = {"case": case(amount=2, unit="nbu_rate_multiple")}

    result = answer(await call(server, "penalty_risk", arguments))

    assert "облікова ставка НБУ" in result["source"] and "01.11.2026" in result["source"]



async def test_a_one_time_penalty_has_no_daily_rate_to_compare(server: MCPServer) -> None:
    arguments = {"case": case(amount=5, unit="one_time_pct")}

    result = answer(await call(server, "penalty_risk", arguments))

    assert Decimal(result["uah"]) == Decimal("50000.00")  # 5 % of 1 000 000, once
    assert result["pct_per_day"] is None and result["above_threshold"] is None


async def test_what_the_text_does_not_say_about_the_term_is_counted_and_said(
    server: MCPServer,
) -> None:
    arguments = {"terms": terms(trigger=None, days_type="other", payment_days=30)}

    result = answer(await call(server, "effective_price", arguments))

    assert Decimal(result["steps"][-1]["uah_per_t"]) == Decimal("-147.95")
    assert "рахую календарні" in result["steps"][-1]["label"]
    assert "рахую від поставки" in result["steps"][-1]["label"]


async def test_an_other_payment_event_comes_back_as_missing(server: MCPServer) -> None:
    arguments = {"terms": terms(trigger="other", payment_days=30)}

    result = answer(await call(server, "effective_price", arguments))

    assert result["uah_per_t"] is None and len(result["missing"]) == 1


# --- refusals

TOO_MANY_DISCOUNTS = [{"reason": "вологість", "pct": 60}, {"reason": "смітна домішка", "pct": 50}]


@pytest.mark.parametrize(
    ("tool", "arguments", "field"),
    [
        ("effective_price", {"terms": terms(currency="other")}, "currency"),
        ("effective_price", {"terms": terms(currency="GBP")}, "currency"),
        ("effective_price", {"terms": terms(amount_per_t=0)}, "amount_per_t"),
        ("effective_price", {"terms": terms(amount_per_t="дев'ять тисяч")}, "amount_per_t"),
        ("effective_price", {"terms": terms(payment_days="30")}, "payment_days"),
        ("effective_price", {"terms": terms(payment_days=True)}, "payment_days"),
        ("effective_price", {"terms": terms(payment_days=-1)}, "payment_days"),
        ("effective_price", {"terms": terms(vat_included="так")}, "vat_included"),
        ("effective_price", {"terms": terms(place="   ")}, "place"),
        ("effective_price", {"terms": terms(farmer_id="farmer_b")}, "farmer_id"),
        ("effective_price", {"terms": terms(today="2026-11-01")}, "today"),
        ("effective_price", {"terms": terms(discounts=TOO_MANY_DISCOUNTS)}, "100 %"),
        ("effective_price", {}, "terms"),
        ("penalty_risk", {"case": case(unit="other")}, "unit"),
        ("penalty_risk", {"case": case(days="10")}, "days"),
        ("penalty_risk", {"case": case(days=731)}, "days"),
        ("penalty_risk", {"case": case(cap_pct=0)}, "cap_pct"),
        ("penalty_risk", {"case": case(sum_uah=-1)}, "sum_uah"),
        ("penalty_risk", {"sum_uah": 1000000, "days": 10}, "case"),
    ],
)
async def test_invalid_input_is_refused_with_the_field_named(
    server: MCPServer, tool: str, arguments: dict[str, Any], field: str
) -> None:
    text = error_text(await call(server, tool, arguments))

    assert "validation error" in text
    assert field in text


# --- start


def test_a_server_with_a_side_effect_tool_is_refused(tmp_path: Path) -> None:
    tools_dir = tmp_path / "tools"
    shutil.copytree(TOOLS_DIR, tools_dir)
    path = tools_dir / "penalty_risk.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace("side_effect: false", "side_effect: true"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="penalty_risk.*read-only"):
        build_server(SETUP, tools_dir=tools_dir)


def test_the_farmers_playbook_and_the_rates_come_from_the_repo_configs() -> None:
    setup = setup_from_env({FARMER_ENV: "farmer_a"})

    assert setup.playbook.farmer_id == "farmer_a"
    assert setup.rates.fx_uah


def test_the_files_can_be_moved_by_the_environment(tmp_path: Path) -> None:
    (tmp_path / "farmer_z.yaml").write_text(
        (REPO / PLAYBOOKS_DIR / "farmer_a.yaml")
        .read_text(encoding="utf-8")
        .replace("farmer_id: farmer_a", "farmer_id: farmer_z"),
        encoding="utf-8",
    )
    rates = tmp_path / "rates.yaml"
    shutil.copy(REPO / RATES_PATH, rates)

    setup = setup_from_env(
        {FARMER_ENV: "farmer_z", PLAYBOOKS_DIR_ENV: str(tmp_path), RATES_PATH_ENV: str(rates)}
    )

    assert setup.playbook.farmer_id == "farmer_z"


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({}, FARMER_ENV),
        ({FARMER_ENV: "farmer_nobody"}, "farmer_nobody.yaml: no such file"),
        ({FARMER_ENV: "../configs/rates"}, "invalid farmer id"),
        ({FARMER_ENV: "farmer_a", RATES_PATH_ENV: "/no/rates.yaml"}, "rates.yaml: no such file"),
    ],
)
def test_a_server_without_its_numbers_does_not_start(env: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        setup_from_env(env)



def test_main_without_a_farmer_exits_with_the_variable_named(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(FARMER_ENV, raising=False)

    with pytest.raises(SystemExit) as stopped:
        main()

    assert stopped.value.code == f"garman-calculators: {FARMER_ENV} is not set: " + (
        "the server serves one farmer, named at start"
    )
    assert capsys.readouterr().out == ""  # stdout carries the protocol: nothing else goes there


async def test_the_server_answers_over_stdio_as_a_subprocess() -> None:
    arguments = terms(amount_per_t=210, currency="USD", basis="CPT", place="Одеса")
    real = setup_from_env({FARMER_ENV: "farmer_a"})
    expected = effective_price(
        PriceTerms.model_validate(arguments), real.playbook, real.rates, date.today()
    )
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.servers.calculators_server"],
        env={FARMER_ENV: "farmer_a"},
        cwd=REPO,
    )

    with anyio.fail_after(STDIO_TIMEOUT_S):  # a server that never starts must not hang CI
        async with Client(params) as client:
            result = await client.call_tool("effective_price", {"terms": arguments})

    assert expected.uah_per_t is not None
    assert Decimal(answer(result)["uah_per_t"]) == expected.uah_per_t
