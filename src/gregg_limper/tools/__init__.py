"""
Auto-discovery & registry for tool handlers.

Any module inside ``tools/handlers`` that defines::

    from gregg_limper.tools import register_tool, Tool, ToolSpec

    @register_tool(ToolSpec(...))
    class MyTool(Tool):
        async def run(self, *, context, **kwargs): ...

is picked up automatically at import-time. The response pipeline uses this
registry to advertise tools to OpenAI and execute tool calls on demand.

Adding a new tool requires changes across the app:

1. Create a handler module under ``tools/handlers`` and register it with
   :func:`register_tool`.
2. Ensure the handler returns meaningful text for the LLM (update tests in
   ``tests/tools``).
3. Document any new configuration in ``config/CONFIG.md`` and update ``config/.env.example``.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from pkgutil import iter_modules
from typing import Any, Dict, Iterable, List, Type

__all__ = [
    "Tool",
    "ToolContext",
    "ToolResult",
    "ToolSpec",
    "ToolExecutionError",
    "build_tool_prompt",
    "get_registered_tool_specs",
    "get_tool_entry",
    "register_tool",
]


@dataclass(slots=True)
class ToolSpec:
    """Static description of a tool exposed to the LLM."""

    name: str
    description: str
    parameters: Dict[str, Any]

    def to_openai(self) -> Dict[str, Any]:
        """Return this spec formatted for OpenAI function calling."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(slots=True)
class ToolContext:
    """Runtime metadata describing the Discord message being serviced."""

    guild_id: int | None
    channel_id: int | None
    message_id: int | None


@dataclass(slots=True)
class ToolResult:
    """Normalized tool output returned to the LLM."""

    content: str


class ToolExecutionError(RuntimeError):
    """Raised when a tool fails to execute successfully."""

    pass


class Tool:
    """Base class for concrete tool handlers."""

    spec: ToolSpec

    async def run(self, *, context: ToolContext, **kwargs: Any) -> ToolResult:
        """Execute the tool and return a :class:`ToolResult`."""

        raise NotImplementedError


_registry: dict[str, Type[Tool]] = {}
_HANDLERS_IMPORTED = False


def register_tool(spec: ToolSpec):
    """Class decorator that binds ``spec`` to the decorated :class:`Tool`."""

    def decorator(cls: Type[Tool]) -> Type[Tool]:
        if not issubclass(cls, Tool):
            raise TypeError("register_tool expects a Tool subclass")
        if spec.name in _registry:
            raise ValueError(f"Tool with name '{spec.name}' already registered")
        cls.spec = spec
        _registry[spec.name] = cls
        return cls

    return decorator


def get_registered_tool_specs() -> List[ToolSpec]:
    """Return all registered specs."""

    return [entry.spec for entry in _registry.values()]


def get_tool_entry(name: str) -> Type[Tool] | None:
    """Return the registered :class:`Tool` subclass for ``name``."""

    return _registry.get(name)


def build_tool_prompt(specs: Iterable[ToolSpec]) -> str:
    """Render a succinct Markdown description of available tools."""

    lines = ["### Tools", "The assistant can call these tools when helpful:"]
    for spec in specs:
        lines.append(f"- **{spec.name}**: {spec.description}")
    return "\n".join(lines)


def _import_handlers() -> None:
    """Import every handler module exactly once."""

    global _HANDLERS_IMPORTED
    if _HANDLERS_IMPORTED:
        return

    pkg_path = Path(__file__).resolve().parent / "handlers"
    for _, modname, _ in iter_modules([str(pkg_path)]):
        if modname.startswith("_"):
            continue
        import_module(f"{__name__}.handlers.{modname}")

    _HANDLERS_IMPORTED = True


_import_handlers()
