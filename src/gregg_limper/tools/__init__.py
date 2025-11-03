"""Tool registry and base classes for model assistance."""

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
    name: str
    description: str
    parameters: Dict[str, Any]

    def to_openai(self) -> Dict[str, Any]:
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
    guild_id: int | None
    channel_id: int | None
    message_id: int | None


@dataclass(slots=True)
class ToolResult:
    content: str


class ToolExecutionError(RuntimeError):
    pass


class Tool:
    spec: ToolSpec

    async def run(self, *, context: ToolContext, **kwargs: Any) -> ToolResult:
        raise NotImplementedError


_registry: dict[str, Type[Tool]] = {}
_handlers_loaded = False


def register_tool(spec: ToolSpec):
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
    _ensure_handlers_loaded()
    return [entry.spec for entry in _registry.values()]


def get_tool_entry(name: str) -> Type[Tool] | None:
    _ensure_handlers_loaded()
    return _registry.get(name)


def build_tool_prompt(specs: Iterable[ToolSpec]) -> str:
    lines = ["### Tools", "The assistant can call these tools when helpful:"]
    for spec in specs:
        lines.append(f"- **{spec.name}**: {spec.description}")
    return "\n".join(lines)


def _ensure_handlers_loaded() -> None:
    global _handlers_loaded
    if _handlers_loaded:
        return
    # Import side-effects register built-in tools
    from . import handlers  # noqa: F401

    _handlers_loaded = True
