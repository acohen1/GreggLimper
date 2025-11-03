"""Tool registry and base classes for model assistance.

Handlers under :mod:`gregg_limper.tools.handlers` register themselves here via
decorators. The response pipeline in turn exposes the registry to OpenAI tools
and executes tool calls at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
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
_handlers_loaded = False


def register_tool(spec: ToolSpec):
    """Class decorator that binds ``spec`` to the decorated :class:`Tool`."""
    def decorator(cls: Type[Tool]) -> Type[Tool]:
        if not issubclass(cls, Tool):
            raise TypeError("Tool registrations must derive from Tool")
        if spec.name in _registry:
            raise ValueError(f"Tool with name '{spec.name}' already registered")
        cls.spec = spec
        _registry[spec.name] = cls
        return cls

    return decorator


def get_registered_tool_specs() -> List[ToolSpec]:
    """Return all registered specs, ensuring handlers are imported."""

    _ensure_handlers_loaded()
    return [entry.spec for entry in _registry.values()]


def get_tool_entry(name: str) -> Type[Tool] | None:
    """Return the registered :class:`Tool` subclass for ``name``."""

    _ensure_handlers_loaded()
    return _registry.get(name)


def build_tool_prompt(specs: Iterable[ToolSpec]) -> str:
    """Render a succinct Markdown description of available tools."""

    lines = ["### Tools", "The assistant can call these tools when helpful:"]
    for spec in specs:
        lines.append(f"- **{spec.name}**: {spec.description}")
    return "\n".join(lines)


def _ensure_handlers_loaded() -> None:
    """Import handler modules so decorator side-effects fire once."""

    global _handlers_loaded
    if _handlers_loaded:
        return
    # Import side-effects register built-in tools
    from . import handlers  # noqa: F401

    _handlers_loaded = True
