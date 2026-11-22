"""Stage 3, paid: the agent's task set on the real model, every tool real.

    make agent                      # the whole set: Qdrant, the farmers' contracts, then this
    make agent IDS=t01,t11          # a few tasks

The agent gets what the bot will give it: the MCP servers `contracts` and `calculators` started
for the task's farmer (data/raw/farmers/, laid out by scripts/setup_farmers.py), search_corpus
over the stage 2 index, and save_reminder (runner.py). The model, the limits and the retries are
those of configs/agent.yaml; the prompt is prompts/agent_v1.md with the servers' instructions.
The run goes to results/stage3/tasks/<start>/ (runner.py); about $0.003 a task on Haiku 5.5.
"""

import argparse
import os
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import yaml

from scripts.chunk_rag import CHUNKS_PATH
from scripts.index_rag import QDRANT_URL, STRUCTURAL_COLLECTION, reachable_client
from src.agents.corpus_tool import search_corpus_tool
from src.agents.limits import AGENT_CONFIG_PATH, AgentConfig, load_agent_config
from src.agents.mcp_tools import McpConnection, garman_servers, system_prompt
from src.agents.model import AnthropicChatModel
from src.agents.runner import RESULTS_DIR, SUMMARY_NAME, RunnerSetup, run_tasks
from src.agents.tasks import TASKS_PATH, Task, coverage_problems, load_tasks
from src.agents.tools import AgentTool
from src.extraction.anthropic_extractor import AnthropicConfig
from src.extraction.prompts import Prompt, load_prompt
from src.extraction.run import ANTHROPIC, load_anthropic_config, load_prices, model_provider
from src.rag.answer_config import load_answer_config
from src.rag.chunking import Chunk
from src.rag.context import ContextSettings, Search
from src.rag.documents import read_jsonl
from src.rag.embedding import bge_m3
from src.rag.index import StaleIndexError, index_stamp
from src.rag.search import Searcher, bge_reranker
from src.servers.contracts_server import CONTRACTS_DIR_ENV, DEFAULT_CONTRACTS_DIR

API_KEY = "ANTHROPIC_API_KEY"
AGENT_PROMPT_PATH = Path("prompts/agent_v1.md")
SETUP_ERRORS = (OSError, ValueError, KeyError, TypeError, yaml.YAMLError)


@dataclass(frozen=True)
class AgentRun:
    """What a run needs before any server starts or any call is paid for."""

    config: AgentConfig
    model: AnthropicConfig
    prices: tuple[float, float]
    prompt: Prompt
    tasks: tuple[Task, ...]
    today: date
    context: ContextSettings  # of configs/answer.yaml: what search_corpus packs
    chunks: tuple[Chunk, ...]
    warnings: tuple[str, ...]  # what the whole set lacks for the course (tasks.coverage_problems)


def select_tasks(tasks: Sequence[Task], ids: str | None) -> tuple[Task, ...]:
    """The tasks named in a comma-separated list, in the order of the set; all without one."""
    if ids is None:
        return tuple(tasks)
    wanted = {task_id.strip() for task_id in ids.split(",") if task_id.strip()}
    if not wanted:
        raise ValueError(f"--ids {ids!r} names no tasks")
    if unknown := sorted(wanted - {task.id for task in tasks}):
        raise ValueError(f"no such tasks: {', '.join(unknown)}")
    return tuple(task for task in tasks if task.id in wanted)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ids", help="comma-separated task ids, e.g. t01,t11; all by default")
    parser.add_argument("--model", help="a section of configs/models.yaml; the config's by default")
    parser.add_argument("--tasks", type=Path, default=TASKS_PATH)
    parser.add_argument("--config", type=Path, default=AGENT_CONFIG_PATH)
    parser.add_argument("--prompt", type=Path, default=AGENT_PROMPT_PATH)
    parser.add_argument("--contracts-dir", type=Path, default=DEFAULT_CONTRACTS_DIR)
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", QDRANT_URL))
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args(argv)


def load_run(args: argparse.Namespace) -> AgentRun:
    config = load_agent_config(args.config)
    if args.model:
        config = config.model_copy(update={"model": args.model})
    if model_provider(config.model) != ANTHROPIC:
        raise ValueError(f"{config.model} is not an anthropic model: the agent calls Claude")
    task_set = load_tasks(args.tasks)
    tasks = select_tasks(task_set.tasks, args.ids)
    for task in tasks:  # here, not when the task's turn comes after the others were paid for
        try:
            task.limits_from(config)
        except ValueError as error:
            raise ValueError(f"task {task.id}: {error}") from None
    return AgentRun(
        config=config,
        model=load_anthropic_config(section=config.model),
        prices=load_prices(section=config.model),
        prompt=load_prompt(args.prompt),
        tasks=tasks,
        today=task_set.today,
        context=load_answer_config().context,
        chunks=read_jsonl(args.chunks, Chunk),
        warnings=() if args.ids else tuple(coverage_problems(task_set)),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not os.environ.get(API_KEY):
        print(f"{API_KEY} is not set: put it in .env and run `make agent`.", file=sys.stderr)
        return 2
    try:
        run = load_run(args)
    except SETUP_ERRORS as error:
        print(f"cannot start: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    for warning in run.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    search = corpus_search(args.qdrant_url, run.chunks)
    if search is None:
        return 1
    try:
        run_dir = run_with_servers(run, search, args)
    except FileExistsError as error:
        print(error, file=sys.stderr)
        return 1
    print(f"summary: {run_dir / SUMMARY_NAME}")
    return 0


def corpus_search(url: str, chunks: Sequence[Chunk]) -> Search | None:
    """The stage 2 search as ask_rag builds it, or None (and why) when it cannot run."""
    client = reachable_client(url)
    if client is None:
        return None
    print("loading bge-m3 and the reranker …", flush=True)
    embedder = bge_m3()
    try:
        return Searcher(
            client, STRUCTURAL_COLLECTION, embedder, index_stamp(chunks, embedder), bge_reranker()
        )
    except StaleIndexError as error:
        print(error, file=sys.stderr)
        return None


def run_with_servers(run: AgentRun, search: Search, args: argparse.Namespace) -> Path:
    """Both MCP servers for each farmer of the tasks, open for the whole run, then closed."""
    corpus = search_corpus_tool(search, run.context)
    farmers = sorted({task.farmer for task in run.tasks})
    extra_env = {CONTRACTS_DIR_ENV: str(args.contracts_dir.resolve())}
    with ExitStack() as stack:
        connections = {
            farmer: stack.enter_context(
                McpConnection(
                    garman_servers(farmer, extra_env=extra_env),
                    call_timeout_s=run.config.limits.tool_timeout_s,
                )
            )
            for farmer in farmers
        }
        instructions = connections[farmers[0]].instructions  # the servers' own: one per server

        def tools_for(farmer: str) -> tuple[AgentTool, ...]:
            return (*connections[farmer].tools, corpus)

        setup = RunnerSetup(
            model=AnthropicChatModel(run.model),
            model_name=run.config.model,
            prices=run.prices,
            system=system_prompt(run.prompt.text, instructions),
            prompt_version=run.prompt.version,
            config=run.config,
            tools_for=tools_for,
            contracts_root=args.contracts_dir,
            today=run.today,
        )
        return run_tasks(run.tasks, setup, args.results_dir, datetime.now())


if __name__ == "__main__":
    sys.exit(main())
