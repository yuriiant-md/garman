"""The agent's tools that live on MCP servers: each server is started once for a whole session.

    with McpConnection(garman_servers("farmer_a"), call_timeout_s=limits.tool_timeout_s) as mcp:
        agent = Agent(tools=mcp.tools, system=system_prompt(prompt.text, mcp.instructions), ...)

A server over stdio takes about a second to start, so it is started once, not per call. Every
tool keeps what its server says of it: the description and the argument schema the model reads,
and the version (meta.version, from tools/<name>.yaml) that goes to the trace. The server checks
the arguments itself; its refusal (403, a bad field) comes back as ToolRefused with the server's
own text, which the model reads and can explain to the farmer. Each answer is checked by the
tool's outputSchema from the server (JSON Schema) like a local tool's by its Pydantic model, and
a tool is safe to retry only when the server marks it read-only (annotations.read_only_hint);
a tool with no outputSchema stops the start, as one with no version does.

Sync and async. The agent runs tool calls in threads (graph.py), the MCP SDK is async. So the
connection runs one event loop in a thread of its own, and one task in that loop holds every
server connection from start to close: anyio lets a connection be closed only by the task that
opened it. A tool call from any thread is sent to that loop and waits there at most
call_timeout_s, so a stuck server cannot keep the agent's thread forever.

Closing waits for that task to close every server, so no server process is left behind; a server
that does not close in time is cancelled, and a failure while closing is logged, not raised, so
it never hides the error that ended the `with` block.
"""

import asyncio
import json
import logging
import re
import sys
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from contextlib import AsyncExitStack
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import Self

from anthropic.types import ToolParam
from jsonschema import SchemaError
from jsonschema.protocols import Validator
from jsonschema.validators import validator_for
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent
from mcp.types import Tool as ServerTool
from pydantic import JsonValue

from src.agents.tools import MAX_PROBLEMS, TOOL_NAME_PATTERN, SchemaMismatch, ToolRefused
from src.extraction.prompts import VERSION_PATTERN
from src.servers.contracts_server import FARMER_ENV

REPO = Path(__file__).resolve().parents[2]
LOOP_THREAD_NAME = "mcp-loop"
START_TIMEOUT_S = 30  # both servers start in ~1.5 s; a server that never answers must not hang
CLOSE_TIMEOUT_S = 10  # to close every server; then the holding task is cancelled
CALL_MARGIN_S = 1  # over call_timeout_s: the SDK's own timeout fires first and names the cause
GARMAN_SERVERS = {
    "contracts": "src.servers.contracts_server",
    "calculators": "src.servers.calculators_server",
}

# what mcp.Client connects to: a server in this process (tests) or a subprocess over stdio
type ServerTarget = MCPServer | StdioServerParameters
type CallTool = Callable[[dict[str, JsonValue]], CallToolResult]
type Described = tuple[tuple[McpTool, ...], dict[str, str]]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class McpTool:
    """A tool of an MCP server, as the agent sees it (tools.AgentTool)."""

    server: str
    name: str
    version: str
    spec: ToolParam
    call: CallTool
    output: Validator  # of the server's outputSchema
    side_effect: bool  # False only for a tool the server marks read-only

    def run(self, arguments: Mapping[str, JsonValue]) -> str:
        return _result_text(self.call(dict(arguments)))

    def check_arguments(self, arguments: Mapping[str, JsonValue]) -> None:
        """Nothing to check here: the server checks the arguments of each call it gets.

        Only a tool with a side effect is checked before a confirmation, and both garman servers
        are read-only; a server tool with a side effect would be put to the farmer unchecked.
        """

    def check_output(self, text: str) -> None:
        try:
            answer = json.loads(text)
        except json.JSONDecodeError:
            raise SchemaMismatch(f"{self.name} answered off its output schema: not JSON") from None
        # a path may mix names and indexes: sorted as text, never compared str with int
        errors = sorted(
            self.output.iter_errors(answer), key=lambda error: [str(part) for part in error.path]
        )
        if errors:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in error.path) or 'answer'}: {error.message}"
                for error in errors[:MAX_PROBLEMS]
            )
            raise SchemaMismatch(f"{self.name} answered off its output schema: {problems}")


class McpConnection:
    """Opens every server on enter and closes them all on exit; used once, tools work in between."""

    def __init__(
        self,
        servers: Mapping[str, ServerTarget],
        *,
        call_timeout_s: float,
        start_timeout_s: float = START_TIMEOUT_S,
    ) -> None:
        self._servers = dict(servers)
        self._call_timeout_s = call_timeout_s
        self._start_timeout_s = start_timeout_s
        self._used = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        # set in the loop's thread by _start_holding; read by close
        self._task: asyncio.Task[None] | None = None
        self._stop: asyncio.Event | None = None
        self._done = threading.Event()  # the holding task has ended, however it ended
        self._hold_error: BaseException | None = None
        # a call is sent to the loop only while the connection is not closing
        self._lock = threading.Lock()
        self._closing = False
        self.tools: tuple[McpTool, ...] = ()
        self.instructions: dict[str, str] = {}  # server name → what the server asks of a client

    def __enter__(self) -> Self:
        if self._used:
            raise RuntimeError("an MCP connection is opened once: make a new one")
        self._used = True
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name=LOOP_THREAD_NAME, daemon=True
        )
        self._thread.start()
        opened: Future[Described] = Future()
        self._loop.call_soon_threadsafe(self._start_holding, opened)
        try:
            self.tools, self.instructions = opened.result(timeout=self._start_timeout_s)
        except BaseException as error:
            # before close: cancelling the task completes `opened` with CancelledError
            timed_out = isinstance(error, TimeoutError) and not opened.done()
            self._close(cancel=True)
            if timed_out:
                names = ", ".join(self._servers)
                raise TimeoutError(
                    f"MCP servers {names} did not start in {self._start_timeout_s:g} s"
                ) from None
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._close(cancel=False)

    # --- in the loop's thread

    def _start_holding(self, opened: Future[Described]) -> None:
        self._stop = asyncio.Event()
        self._task = asyncio.get_running_loop().create_task(self._hold(opened))
        self._task.add_done_callback(self._held)  # runs even when cancelled before it started

    async def _hold(self, opened: Future[Described]) -> None:
        """The one task that opens every server, keeps them open until stop, then closes them."""
        assert self._stop is not None  # set by _start_holding before this task starts
        try:
            async with AsyncExitStack() as stack:
                clients = {
                    name: await stack.enter_async_context(Client(target))
                    for name, target in self._servers.items()
                }
                opened.set_result(await self._describe(clients))
                await self._stop.wait()
        except BaseException as error:
            if not opened.done():
                opened.set_exception(_cause(error))
            raise

    def _held(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            self._hold_error = task.exception()  # retrieved here, so asyncio does not warn
        self._done.set()

    async def _describe(self, clients: Mapping[str, Client]) -> Described:
        tools = [
            _agent_tool(server, tool, partial(self._call, client, tool.name))
            for server, client in clients.items()
            for tool in await _all_tools(client)
        ]
        instructions = {server: client.instructions or "" for server, client in clients.items()}
        return tuple(tools), instructions

    # --- in the agent's threads

    def _call(self, client: Client, name: str, arguments: dict[str, JsonValue]) -> CallToolResult:
        """Called from the agent's tool thread; the call itself runs in the connection's loop."""
        with self._lock:  # a call sent after close began would never run: refuse it now
            if self._closing or self._loop is None:
                raise RuntimeError(f"{name}: the MCP connection is closed")
            future = asyncio.run_coroutine_threadsafe(
                client.call_tool(name, arguments, read_timeout_seconds=self._call_timeout_s),
                self._loop,
            )
        try:
            return future.result(timeout=self._call_timeout_s + CALL_MARGIN_S)
        except TimeoutError:
            future.cancel()
            raise

    def _close(self, *, cancel: bool) -> None:
        """Close every server, then stop the loop; cancel: the start failed, do not wait for it."""
        with self._lock:
            if self._closing or self._loop is None:
                return
            self._closing = True
        loop, task, stop = self._loop, self._task, self._stop
        if task is not None and stop is not None:
            loop.call_soon_threadsafe(task.cancel if cancel else stop.set)
            if not self._done.wait(CLOSE_TIMEOUT_S) and not cancel:
                logger.warning("MCP servers did not close in %g s: cancelled", CLOSE_TIMEOUT_S)
                loop.call_soon_threadsafe(task.cancel)
                self._done.wait(CLOSE_TIMEOUT_S)
        if self._hold_error is not None and not cancel:  # on a failed start it is already raised
            logger.warning("closing the MCP servers failed: %r", _cause(self._hold_error))
        self._stop_loop()

    def _stop_loop(self) -> None:
        loop, thread = self._loop, self._thread
        if loop is None or thread is None:
            return
        loop.call_soon_threadsafe(loop.stop)
        thread.join(CLOSE_TIMEOUT_S)
        if not thread.is_alive():  # a loop that still runs cannot be closed; the thread is a daemon
            loop.close()
        self._loop, self._thread = None, None


async def _all_tools(client: Client) -> list[ServerTool]:
    """Every tool of a server: the list may come in pages."""
    page = await client.list_tools()
    tools = list(page.tools)
    while page.next_cursor:
        page = await client.list_tools(cursor=page.next_cursor)
        tools.extend(page.tools)
    return tools


def _agent_tool(server: str, tool: ServerTool, call: CallTool) -> McpTool:
    """The server's tool for the agent; no version or a name the API refuses stops the start."""
    version = (tool.meta or {}).get("version")
    if not isinstance(version, str) or not VERSION_PATTERN.fullmatch(version):
        raise ValueError(f"server {server}: tool {tool.name} has no valid version in its meta")
    if not re.fullmatch(TOOL_NAME_PATTERN, tool.name):
        raise ValueError(f"server {server}: tool name {tool.name!r} is not allowed by the API")
    spec: ToolParam = {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": tool.input_schema,
    }
    read_only = tool.annotations is not None and tool.annotations.read_only_hint is True
    return McpTool(
        server=server,
        name=tool.name,
        version=version,
        spec=spec,
        call=call,
        output=_output_validator(server, tool),
        side_effect=not read_only,
    )


def _output_validator(server: str, tool: ServerTool) -> Validator:
    """A validator of the tool's outputSchema; none, or a broken one, stops the start."""
    if tool.output_schema is None:
        raise ValueError(f"server {server}: tool {tool.name} has no output schema")
    schema = tool.output_schema
    cls = validator_for(schema)
    try:
        cls.check_schema(schema)
    except SchemaError as error:
        raise ValueError(
            f"server {server}: tool {tool.name}: bad output schema: {error.message}"
        ) from None
    return cls(schema)


def _cause(error: BaseException) -> BaseException:
    """The one error inside the SDK's task groups, so the start fails with the real reason."""
    while isinstance(error, BaseExceptionGroup) and len(error.exceptions) == 1:
        error = error.exceptions[0]
    return error


def _result_text(result: CallToolResult) -> str:
    """What the model reads: the structured answer as compact JSON, or the server's own text."""
    text = "\n".join(block.text for block in result.content if isinstance(block, TextContent))
    if result.is_error:
        raise ToolRefused(text or "the tool refused without a reason")
    if result.structured_content is not None:
        return json.dumps(result.structured_content, ensure_ascii=False, separators=(",", ":"))
    if not text:
        raise ValueError("the tool returned no text")
    return text


def garman_servers(
    farmer_id: str, *, extra_env: Mapping[str, str] | None = None
) -> dict[str, StdioServerParameters]:
    """Our two servers as subprocesses of this Python, for one farmer.

    The SDK gives a subprocess only a short safe list of the parent's variables (PATH, HOME, ...)
    plus these, so the API key in .env never reaches a server.
    """
    env = {FARMER_ENV: farmer_id, **(extra_env or {})}
    return {
        name: StdioServerParameters(
            command=sys.executable, args=["-m", module], env=dict(env), cwd=REPO
        )
        for name, module in GARMAN_SERVERS.items()
    }


def system_prompt(base: str, instructions: Mapping[str, str]) -> str:
    """The agent's prompt, then what each server asks of its clients, under the server's name."""
    sections = [
        f"## MCP server {server}\n{text.strip()}"
        for server, text in instructions.items()
        if text.strip()
    ]
    return "\n\n".join([base.strip(), *sections])
