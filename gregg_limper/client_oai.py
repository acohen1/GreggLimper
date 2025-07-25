# client_oai.py
import base64
from openai import AsyncOpenAI
from gregg_limper.config import Config
import re

import logging
logger = logging.getLogger(__name__)

# One global async-capable client
aoai = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)

async def describe_image_bytes(
    blob: bytes,
    mime: str = "image/png",        # image/jpeg, image/png, etc.
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
    "Open and read the exact page at the URL—not previews or related content. "
    "Write a concise, information-dense summary (~150-200 words) that captures the main purpose, key arguments, and critical details. "
    "Include names, dates, figures, and statistics. Use full sentences, no bullet points. "
    "End with one sentence that distills the page's core message. Focus only on high-signal content."
    ),
    model: str = Config.WEB_MODEL_ID,
    enable_citations: bool = True,
) -> str:
    """
    Async wrapper around the Web-Search tool.

    - Sends the URL as input plus the built-in `web_search_preview` tool.
    - Returns plain-text `output_text`.
    - Context size controls how much detail the model should include (low, medium, high).
    """
    try:
        resp = await aoai.chat.completions.create(
            model=model,
            web_search_options={
                "search_context_size": context_size,
            },
            messages=[{
                "role": "user",
                "content": f"{prompt}\nURL: {url}",
            }],
        )

        summary = resp.choices[0].message.content.strip()
        # Remove in-text citations if requested
        if not enable_citations:
            summary = re.sub(r'\[.*?\]\(https?://[^\)]+\)', '', summary)
        return summary
    
        # NOTE: (LEGACY) The following code is an alternative way to use the web search tool for models
        # that aren't specifically dedicated to web search. If using -search-preview models, ignore this,
        # you can use the above method directly.
        # This is left here for reference but not used in the current implementation.
        
        # resp = await aoai.responses.create(
        #     model=model,
        #     tools=[{
        #         "type": "web_search_preview",
        #         "search_context_size": context_size,
        #     }],
        #
        #     input=f"{prompt}\nURL: {url}",
        #     # max_output_tokens=max_output_tokens
        # )
        # summary = resp.output_text.strip()
        # # Remove in-text citations if requested
        # if not enable_citations:
        #     summary = re.sub(r'\[.*?\]\(https?://[^\)]+\)', '', summary)
        # return summary

    except Exception as e:
        logger.error(f"Error summarizing URL {url}: {e}")
        return "Error summarizing URL"