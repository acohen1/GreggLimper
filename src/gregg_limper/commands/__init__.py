"""Slash command registration helpers."""

from __future__ import annotations

import logging
from importlib import import_module
from pathlib import Path
from pkgutil import iter_modules
from typing import List, Optional, Type

from discord.ext import commands as commands_ext

logger = logging.getLogger(__name__)

_COG_CLASSES: List[Type[commands_ext.Cog]] = []
_DISCOVERED = False


def register_cog(cls: Optional[Type[commands_ext.Cog]] = None):
    """
    Register a Cog class for later attachment to the bot.

    Modules should decorate their Cog classes with ``@register_cog`` so they are
    attached automatically during :func:`setup`.
    """

    def _register(cog_cls: Type[commands_ext.Cog]):
        if not issubclass(cog_cls, commands_ext.Cog):
            raise TypeError("register_cog expects a discord.ext.commands.Cog subclass")

        _COG_CLASSES.append(cog_cls)
        return cog_cls

    if cls is None:
        return _register
    return _register(cls)


def _discover_handlers() -> None:
    """Import handler modules once so they can register their cogs."""

    global _DISCOVERED
    if _DISCOVERED:
        return

    pkg_path = Path(__file__).resolve().parent / "handlers"
    for _, modname, _ in iter_modules([str(pkg_path)]):
        if modname.startswith("_"):
            continue
        import_module(f"{__name__}.handlers.{modname}")

    _DISCOVERED = True


async def setup(bot: commands_ext.Bot) -> None:
    """
    Attach registered cogs to ``bot``.

    This must be invoked during the bot setup phase (typically inside
    ``commands.Bot.setup_hook``).
    """

    _discover_handlers()

    for cog_cls in _COG_CLASSES:
        if bot.get_cog(cog_cls.__name__):
            continue
        await bot.add_cog(cog_cls(bot))

    if _COG_CLASSES:
        logger.info("Registered %d command cog(s)", len(_COG_CLASSES))
    else:
        logger.warning("No command cogs discovered; command tree is empty")


__all__ = [
    "register_cog",
    "setup",
]
