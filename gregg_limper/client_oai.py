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


async def summarize_url(
    url: str,
    context_size: str = "low",                # "low" · "medium" · "high"
    prompt: str = (
        "Read the webpage at the given URL and produce a concise but information-dense abstract (≈150-200 words)."
        " - Capture the page's main thesis or purpose, key supporting points, and any notable statistics or figures."
        " - Include names, dates, and proper nouns that are central to understanding the content."
        " - Omit navigation menus, ads, and unrelated sidebars."
        " - Write in complete sentences—no bullet points."
        " - End with a single-sentence takeaway that someone could quote to convey the essence of the page."
    ),
    model: str = Config.WEB_MODEL_ID,
) -> str:
    """
    Async wrapper around the Web-Search tool.

    - Sends the URL as input plus the built-in `web_search_preview` tool.
    - Returns plain-text `output_text`.
    """
    resp = await aoai.responses.create(
        model=model,
        tools=[{
            "type": "web_search_preview",
            "search_context_size": context_size,
        }],
        input=f"{prompt}\nURL: {url}",
    )
    return resp.output_text.strip()