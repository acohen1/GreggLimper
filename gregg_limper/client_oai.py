# client_oai.py
import base64
from openai import AsyncOpenAI
from config import Config

# One global async-capable client
aoai = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)

async def describe_image_bytes(
    blob: bytes,
    mime: str = "image/png",
    prompt: str = (
        "Describe the image clearly and objectively, focusing on visible text, "
        "layout, and key visual elements. Avoid interpretation or embellishment—"
        "act as if explaining the image to someone who cannot see it."
    ),
    model: str = Config.IMG_MODEL_ID,
) -> str:
    """
    Returns plain-text description from the vision model.
    """
    b64 = base64.b64encode(blob).decode()

    resp = await aoai.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    { "type": "input_text",  "text": prompt },
                    { "type": "input_image",
                      "image_url": f"data:{mime};base64,{b64}" },
                ],
            }
        ],
    )
    return resp.output_text.strip()