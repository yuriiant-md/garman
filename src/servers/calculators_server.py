"""MCP server `calculators`: effective price and penalty risk in UAH for one farmer, over stdio.

    GARMAN_FARMER_ID=farmer_a uv run python -m src.servers.calculators_server

The farmer is named once, at start, never by the model: the server reads that farmer's playbook
(configs/playbooks/<farmer_id>.yaml, freight and cost of money) and the shared rates
(configs/rates.yaml) when it starts. GARMAN_PLAYBOOKS_DIR and GARMAN_RATES_PATH override them.
A missing or broken file stops the start with a message that names the file.

The numbers come from src/tools/ (Decimal, rounded to the kopeck), never from the model. The
SDK checks the arguments against PriceTerms / PenaltyCase (strict ints and bools, ranges, known
units and currencies, no extra fields) and sends a refusal's text back to the model.

The model does not pass today's date: the server takes it from its clock, so the model cannot
make an old exchange rate look fresh. A number the calculation lacks is not an error: the
answer lists it in `missing`, and the model asks the farmer.
"""

import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from src.servers.registry import register_read_only
from src.tools import price, risk
from src.tools.playbook import PLAYBOOKS_DIR, RATES_PATH, Playbook, Rates, load_playbook, load_rates
from src.tools.price import EffectivePrice, PriceTerms
from src.tools.risk import PenaltyCase, PenaltyRisk

REPO = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO / "tools"
FARMER_ENV = "GARMAN_FARMER_ID"
PLAYBOOKS_DIR_ENV = "GARMAN_PLAYBOOKS_DIR"
RATES_PATH_ENV = "GARMAN_RATES_PATH"
SERVER_NAME = "garman-calculators"
INSTRUCTIONS = (
    "Калькулятори фермера в гривнях. Фермера задано при запуску сервера. "
    "Гроші рахує сервер: не перераховуй числа сам і не вигадуй тих, яких бракує. "
    "Якщо відповідь має missing, спитай фермера саме про це. "
    "Якщо stale, скажи фермерові, що курс чи ставка застарілі."
)


@dataclass(frozen=True)
class FarmerSetup:
    """What the server needs for one farmer, read once at start."""

    playbook: Playbook
    rates: Rates


def build_server(
    setup: FarmerSetup,
    *,
    clock: Callable[[], date] = date.today,
    tools_dir: Path = TOOLS_DIR,
) -> MCPServer:
    """The server for one farmer; refuses a tool whose description says it has a side effect."""

    def effective_price(terms: PriceTerms) -> EffectivePrice:
        return price.effective_price(terms, setup.playbook, setup.rates, clock())

    def penalty_risk(case: PenaltyCase) -> PenaltyRisk:
        return risk.penalty_risk(case, setup.playbook, setup.rates, clock())

    server = MCPServer(SERVER_NAME, instructions=INSTRUCTIONS)
    for func in (effective_price, penalty_risk):
        register_read_only(server, func, tools_dir)
    return server


def setup_from_env(env: Mapping[str, str]) -> FarmerSetup:
    """The farmer's playbook and the rates; a ValueError names what is missing or wrong."""
    farmer_id = env.get(FARMER_ENV, "")
    if not farmer_id:
        raise ValueError(f"{FARMER_ENV} is not set: the server serves one farmer, named at start")
    playbooks = Path(env[PLAYBOOKS_DIR_ENV]) if env.get(PLAYBOOKS_DIR_ENV) else REPO / PLAYBOOKS_DIR
    rates = Path(env[RATES_PATH_ENV]) if env.get(RATES_PATH_ENV) else REPO / RATES_PATH
    return FarmerSetup(playbook=load_playbook(farmer_id, playbooks), rates=load_rates(rates))


def main() -> None:
    try:
        setup = setup_from_env(os.environ)
    except ValueError as error:
        sys.exit(f"{SERVER_NAME}: {error}")
    build_server(setup).run()  # stdio


if __name__ == "__main__":
    main()
