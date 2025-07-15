from discord import Message
from gregg_limper.formatter.classifier import classify
from gregg_limper.formatter.composer import compose

async def format_message(msg: Message) -> str:
    """
    Async entry-point used by GLCache / bot.
    """
    classified_msg = classify(msg)
    return await compose(msg.author, classified_msg)
