"""Helpers for interacting with a local LLaMA-based server"""

from gregg_limper.config import Config
from ollama import AsyncClient

client = AsyncClient(host=Config.LOCAL_SERVER_URL)

async def chat(
        messages: list[dict],
        model = Config.LOCAL_MODEL_ID
    ) -> str:
    """
    Send a prompt to the local Ollama server and return its reply.

    Example input messages list[dict]:

    .. code-block:: python
        [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ]
    """
    resp = await client.chat(
        model=model,
        messages=messages
    )

    return resp.message.content.strip()



