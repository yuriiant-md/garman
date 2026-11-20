"""How our MCP servers put a function behind a tool: the description from tools/<name>.yaml.

Both servers (contracts, calculators) only read: a tool whose description says it has a side
effect is refused at start, so a server that could change something never starts by mistake.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from src.agents.tools import load_tool_description

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


def register_read_only(server: MCPServer, func: Callable[..., Any], tools_dir: Path) -> None:
    """Add func as the tool tools_dir/<func name>.yaml describes; its version goes to meta."""
    description = load_tool_description(tools_dir / f"{func.__name__}.yaml")
    if description.side_effect:
        raise ValueError(f"{description.name}: the server {server.name} is read-only")
    server.add_tool(
        func,
        name=description.name,
        description=description.description,
        annotations=READ_ONLY,
        meta={"version": description.version},
    )
