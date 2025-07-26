"""
responses/
==========
This package defines post-caching behaviors that trigger when the bot is mentioned
or otherwise flagged to respond. These actions do not bypass the formatting pipeline.

Use case:
- When the bot is mentioned in a message, we fetch recent context from GLCache,
  build a prompt, and generate a response via OpenAI.

Main entrypoints:
- respond_to_mention(message: discord.Message) -> Awaitable[None]
    Called after the message has been cached and formatted. Fetches context and
    calls the OpenAI client to generate a response.

This package cleanly separates response logic from event_hooks and command parsing.
"""
