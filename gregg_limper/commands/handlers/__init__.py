"""
Auto-discovery & registry for command handlers.

Any file inside commands/handlers/ that defines:
    from . import register

    @register
    class MyCommandHandler:
        command_str = "mycommand"
        @staticmethod
        async def handle(client: discord.Client, message: discord.Message, args: str): ...

is picked up automatically at import-time.

NOTE: If adding a new handler, ensure:
1. It has a unique 'command_str' string
2. It implements the CommandHandler protocol (see below).
3. It is placed in this directory (commands/handlers/).
"""
from __future__ import annotations
from typing import Protocol, Dict
from importlib import import_module
from pkgutil import iter_modules
from pathlib import Path
import discord

class CommandHandler(Protocol):
    """Protocol for command handler classes."""

    command_str: str

    @staticmethod
    async def handle(client: discord.Client, message: discord.Message, args: str) -> None:
        """Coroutine invoked when the command is dispatched.

        :param client: Discord client instance.
        :param message: Incoming command message.
        :param args: Raw argument string.
        """


_REGISTRY: Dict[str, CommandHandler] = {}


def register(cls: CommandHandler):
    """Decorator that registers a ``CommandHandler`` implementation.

    :param cls: Class implementing the handler protocol.
    :returns: The class unchanged.
    """
    _REGISTRY[cls.command_str] = cls
    return cls


def get(command: str) -> CommandHandler | None:
    """Return handler class for ``command`` or ``None``."""
    return _REGISTRY.get(command)


def all_commands() -> Dict[str, CommandHandler]:
    """Return copy of the command registry."""
    return dict(_REGISTRY)


# ------------------------------------------------------------------ #
# Auto-import sibling modules to populate registry
# ------------------------------------------------------------------ #
_pkg_path = Path(__file__).resolve().parent
for _, modname, _ in iter_modules([str(_pkg_path)]):
    if modname != "__init__":
        import_module(f"{__name__}.{modname}")
