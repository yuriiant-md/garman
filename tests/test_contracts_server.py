import logging
import shutil
import sys
from pathlib import Path

import anyio
import pymupdf
import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent

from src.documents import contract
from src.servers import contract_access
from src.servers.contract_access import ContractShelf
from src.servers.contracts_server import (
    CONTRACTS_DIR_ENV,
    FARMER_ENV,
    TOOLS_DIR,
    build_server,
    shelf_from_env,
)

pytestmark = pytest.mark.anyio

REPO = Path(__file__).resolve().parents[1]
STDIO_TIMEOUT_S = 30  # a cold start of the subprocess takes ~2 s
TOOL_NAMES = ["list_contracts", "contract_outline", "get_clause"]
PAGES = (
    "ДОГОВІР ПОСТАВКИ № 1\n",
    "1. Предмет\n1.1. Продавець передає пшеницю 3 класу.\n1.2. Покупець приймає товар.\n",
    "2. Ціна\n2.1. Ціна 9 000 грн за тонну.\n2.2. Пеня 0,1% за кожен день прострочення.\n",
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


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
def server(root: Path, monkeypatch: pytest.MonkeyPatch) -> MCPServer:
    monkeypatch.setattr(contract, "pdf_pages", lambda path: PAGES)
    return build_server(ContractShelf(root=root, farmer_id="farmer_a"))


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


async def test_no_tool_lets_the_model_name_the_farmer(server: MCPServer) -> None:
    async with Client(server) as client:
        tools = (await client.list_tools()).tools

    for tool in tools:
        assert not any("farmer" in name for name in tool.input_schema.get("properties", {}))


# --- answers


async def test_list_contracts_returns_only_the_farmers_own(server: MCPServer) -> None:
    async with Client(server) as client:
        result = await client.call_tool("list_contracts", {})

    assert not result.is_error
    assert result.structured_content == {
        "result": [{"contract_id": "k_001", "clauses": 4, "note": None}]
    }


async def test_contract_outline_lists_the_clause_numbers(server: MCPServer) -> None:
    async with Client(server) as client:
        result = await client.call_tool("contract_outline", {"contract_id": "k_001"})

    assert result.structured_content is not None
    numbers = [item["number"] for item in result.structured_content["result"]]
    assert numbers == ["1.1", "1.2", "2.1", "2.2"]


async def test_get_clause_returns_the_clause_with_its_section_and_pages(
    server: MCPServer,
) -> None:
    async with Client(server) as client:
        result = await client.call_tool("get_clause", {"contract_id": "k_001", "clause": "2.2"})

    assert result.structured_content == {
        "contract_id": "k_001",
        "number": "2.2",
        "section": "2. Ціна",
        "pages": [3],
        "text": "2.2. Пеня 0,1% за кожен день прострочення.",
    }


# --- refusals


async def test_another_farmers_contract_is_a_403_error_the_model_reads(
    server: MCPServer, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        async with Client(server) as client:
            result = await client.call_tool("get_clause", {"contract_id": "k_900", "clause": "1.1"})

    assert "403 access denied" in error_text(result)
    assert "farmer_a" in caplog.text and "k_900" in caplog.text  # who asked for what, on the server


async def test_an_extra_farmer_argument_does_not_open_another_farmers_contract(
    server: MCPServer,
) -> None:
    async with Client(server) as client:
        result = await client.call_tool(
            "contract_outline", {"contract_id": "k_900", "farmer_id": "farmer_b"}
        )

    assert "403 access denied" in error_text(result)  # the SDK drops the extra argument


async def test_an_unknown_contract_is_a_404_error(server: MCPServer) -> None:
    async with Client(server) as client:
        result = await client.call_tool("contract_outline", {"contract_id": "k_404"})

    assert "404 not found" in error_text(result)


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("contract_outline", {"contract_id": "../farmer_b/k_900"}),
        ("contract_outline", {"contract_id": "/etc/passwd"}),
        ("contract_outline", {"contract_id": "k_001\n"}),
        ("contract_outline", {}),
        ("contract_outline", {"contract_id": 1}),
        ("get_clause", {"contract_id": "k_001", "clause": "п. 7.3"}),
        ("get_clause", {"contract_id": "k_001"}),
    ],
)
async def test_invalid_input_is_refused_by_the_server(
    server: MCPServer, tool: str, arguments: dict[str, object]
) -> None:
    async with Client(server) as client:
        result = await client.call_tool(tool, arguments)

    assert "validation error" in error_text(result)


# --- start


def test_a_server_with_a_side_effect_tool_is_refused(root: Path, tmp_path: Path) -> None:
    tools_dir = tmp_path / "tools"
    shutil.copytree(TOOLS_DIR, tools_dir)
    path = tools_dir / "get_clause.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace("side_effect: false", "side_effect: true"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="get_clause.*read-only"):
        build_server(ContractShelf(root=root, farmer_id="farmer_a"), tools_dir=tools_dir)


def test_the_farmer_and_the_folder_come_from_the_environment(root: Path) -> None:
    shelf = shelf_from_env({FARMER_ENV: "farmer_a", CONTRACTS_DIR_ENV: str(root)})

    assert (shelf.root, shelf.farmer_id) == (root, "farmer_a")


def test_a_server_without_a_farmer_does_not_start(root: Path) -> None:
    with pytest.raises(ValueError, match=FARMER_ENV):
        shelf_from_env({CONTRACTS_DIR_ENV: str(root)})


async def test_the_server_answers_over_stdio_as_a_subprocess(root: Path) -> None:
    pdf = root / "farmer_a" / "k_001.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), "1.1. Price is fixed.\n1.2. Delivery by truck.")
        doc.save(pdf)
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.servers.contracts_server"],
        env={FARMER_ENV: "farmer_a", CONTRACTS_DIR_ENV: str(root)},
        cwd=REPO,
    )

    with anyio.fail_after(STDIO_TIMEOUT_S):  # a server that never starts must not hang CI
        async with Client(params) as client:
            result = await client.call_tool(
                "get_clause", {"contract_id": "k_001", "clause": "1.2"}
            )

    assert result.structured_content is not None
    assert result.structured_content["text"] == "1.2. Delivery by truck."
