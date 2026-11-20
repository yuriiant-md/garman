"""MCP server `contracts`: one farmer's contracts, read-only, over stdio.

    GARMAN_FARMER_ID=farmer_a uv run python -m src.servers.contracts_server

The farmer is named once, at start, by whoever launches the server (the bot, the agent, Claude
Desktop), never by the model; GARMAN_CONTRACTS_DIR overrides the folder (data/raw/farmers/).
The rules of access live in contract_access.py; this file only puts them behind MCP.

A tool's description is the one in tools/<name>.yaml, its version goes to the client in the
tool's meta (registry.py). The SDK checks the arguments against the schema built from the type
hints, so an id that could be a path is refused before contract_access sees it (and there again).

A refusal of ours (403, 404, an unreadable file, a bad argument) becomes a ToolError: the
model reads its text and can explain it to the farmer. Any other exception the SDK reports
without its text, so a crash shows nothing of the server to the client.
"""

import logging
import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from src.servers.contract_access import (
    CLAUSE_PATTERN,
    ID_PATTERN,
    AccessDenied,
    Clause,
    ContractShelf,
    ContractSummary,
    InvalidRequest,
    NotFound,
    OutlineItem,
    UnreadableContract,
)
from src.servers.registry import register_read_only

REPO = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO / "tools"
DEFAULT_CONTRACTS_DIR = REPO / "data" / "raw" / "farmers"  # real PDFs: never in git
FARMER_ENV = "GARMAN_FARMER_ID"
CONTRACTS_DIR_ENV = "GARMAN_CONTRACTS_DIR"
SERVER_NAME = "garman-contracts"
INSTRUCTIONS = (
    "Договори одного фермера, лише читання. Фермера задано при запуску сервера. "
    "Порядок: list_contracts → contract_outline → get_clause. "
    "На 403 не шукай обхідних шляхів: поясни фермеру, що це не його договір."
)
REFUSALS = (InvalidRequest, AccessDenied, NotFound, UnreadableContract)

ContractId = Annotated[
    str, Field(pattern=f"^{ID_PATTERN}$", description='contract_id із list_contracts, "k_001"')
]
ClauseNumber = Annotated[
    str, Field(pattern=f"^{CLAUSE_PATTERN}$", description='номер пункту без "п.", "7.3"')
]

logger = logging.getLogger(__name__)


def build_server(shelf: ContractShelf, *, tools_dir: Path = TOOLS_DIR) -> MCPServer:
    """The server for one farmer; refuses a tool whose description says it has a side effect."""

    def list_contracts() -> tuple[ContractSummary, ...]:
        with _refusals_for_the_model(shelf):
            return shelf.contracts()

    def contract_outline(contract_id: ContractId) -> tuple[OutlineItem, ...]:
        with _refusals_for_the_model(shelf):
            return shelf.outline(contract_id)

    def get_clause(contract_id: ContractId, clause: ClauseNumber) -> Clause:
        with _refusals_for_the_model(shelf):
            return shelf.clause(contract_id, clause)

    server = MCPServer(SERVER_NAME, instructions=INSTRUCTIONS)
    for func in (list_contracts, contract_outline, get_clause):
        register_read_only(server, func, tools_dir)
    return server


@contextmanager
def _refusals_for_the_model(shelf: ContractShelf) -> Iterator[None]:
    """Our refusals reach the model with their text; a 403 is also logged with who asked."""
    try:
        yield
    except REFUSALS as error:
        level = logging.WARNING if isinstance(error, AccessDenied) else logging.INFO
        logger.log(level, "farmer %s: %s", shelf.farmer_id, error)
        raise ToolError(str(error)) from error


def shelf_from_env(env: Mapping[str, str]) -> ContractShelf:
    farmer_id = env.get(FARMER_ENV, "")
    if not farmer_id:
        raise ValueError(f"{FARMER_ENV} is not set: the server serves one farmer, named at start")
    root = Path(env[CONTRACTS_DIR_ENV]) if env.get(CONTRACTS_DIR_ENV) else DEFAULT_CONTRACTS_DIR
    return ContractShelf(root=root, farmer_id=farmer_id)


def main() -> None:
    logging.basicConfig(level=logging.INFO)  # to stderr: stdout carries the protocol
    try:
        shelf = shelf_from_env(os.environ)
    except ValueError as error:
        sys.exit(f"{SERVER_NAME}: {error}")
    build_server(shelf).run()  # stdio


if __name__ == "__main__":
    main()
