from discord import Message
from gregg_limper.formatter.classifier import classify
from gregg_limper.formatter.composer import compose

async def format_message(msg: Message) -> str:
    """
    Takes a Discord Message object, classifies it, and composes the media fragments. Returns a serialized JSON string
    formatted for the Gregg Limper cache:
    {
        "author": {"id": <user_id>, "name": <display_name>},
        "channel_id": <channel_id>,
        "timestamp": <isoformat_timestamp>,
        "fragments": [
            {"type": "<media-type>", "title": "<cleaned-title>", "vision": "<frame description>", ...},
            ...
        ]
    }
    """
    classified_msg = classify(msg)
    return await compose(msg, classified_msg)
