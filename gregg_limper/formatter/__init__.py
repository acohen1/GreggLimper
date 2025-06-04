from discord import Message
from .classifier import classify
from .composer import compose

async def format_message(msg: Message) -> str:
    """
    Async entry-point used by GLCache / bot.
    """
    classified_msg = classify(msg)
    return await compose(msg.author.display_name, classified_msg)
