"""Tool execution helpers."""

from __future__ import annotations

import json
from typing import Any

from . import ToolContext, ToolExecutionError, ToolResult, get_tool_entry


async def execute_tool(name: str, arguments: str | dict[str, Any], *, context: ToolContext) -> ToolResult:
    """
    Execute the registered tool ``name`` with ``arguments``.

    ``arguments`` may be a JSON string (as provided by OpenAI) or a parsed
    mapping.  The helper normalises the payload, instantiates the tool class,
    and returns its :class:`ToolResult`.
    """

    entry = get_tool_entry(name)
    if entry is None:
        raise ToolExecutionError(f"Unknown tool '{name}'")

    if isinstance(arguments, str):
        try:
            parsed_args = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(f"Invalid JSON arguments for tool '{name}': {exc}") from exc
    else:
        parsed_args = arguments

    tool = entry()
    try:
        return await tool.run(context=context, **parsed_args)
    except ToolExecutionError:
        raise
    except TypeError as exc:
        raise ToolExecutionError(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ToolExecutionError(f"Tool '{name}' execution failed: {exc}") from exc
