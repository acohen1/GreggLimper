"""
commands/
=========
This package handles explicit bot commands that trigger immediate side effects
(e.g., '/lobotomy'). Commands are typically identified by special prefixes and
parsed before any message formatting or caching occurs.

Main entrypoints:
- parse_command(message: discord.Message) -> Optional[str]
    Determines if the message contains a recognized command.
- dispatch_command(command: str, message: discord.Message) -> Awaitable[bool]
    Executes the command and returns whether the message should continue through
    the standard processing pipeline.

Commands are isolated from core formatting logic and do not mutate shared state
outside of their intended effect (e.g., clearing cache).
"""
