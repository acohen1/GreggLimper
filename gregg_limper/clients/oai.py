"""Helpers for interacting with OpenAI API"""
import base64
from openai import AsyncOpenAI
from gregg_limper.config import core, rag
import re
import numpy as np

import logging
logger = logging.getLogger(__name__)

# One global async-capable client
aoai = AsyncOpenAI(api_key=core.OPENAI_API_KEY)

# ==============================================
# Embedding utilities
# ==============================================
async def embed_text(text: str, model: str | None = None) -> np.ndarray:
    """
    Return a float32 numpy vector for the given text using OpenAI embeddings.
    Model defaults to rag.EMB_MODEL_ID.
    """
    if not text:
        # Use rag.EMB_DIM if provided; else assume common dims for the model
        dim = rag.EMB_DIM
        return np.zeros(dim, dtype=np.float32)

    use_model = model or rag.EMB_MODEL_ID
    resp = await aoai.embeddings.create(model=use_model, input=text)
    vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
    # Optional safety check if you set EMB_DIM in rag
    dim = rag.EMB_DIM
    if vec.size != dim:
        raise ValueError(f"Unexpected embedding size {vec.size} != {dim} for model {use_model}")
    
    return vec

# ==============================================
# Image utilities
# ==============================================

async def describe_image_bytes(
    blob: bytes,
    mime: str = "image/png",  # image/jpeg, image/png, etc.
    prompt: str = (
        "Describe the image clearly and objectively, focusing on visible text, "
        "layout, and key visual elements. Avoid interpretation or embellishment—"
        "act as if explaining the image to someone who cannot see it."
    ),
    model: str = core.IMG_MODEL_ID,
) -> str:
    """Returns plain-text description from the vision model."""
    b64 = base64.b64encode(blob).decode()

    resp = await aoai.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"},
                ],
            }
        ],
    )
    return resp.output_text.strip()


async def summarize_url(
    url: str,
    context_size: str = "low",  # "low" · "medium" · "high"
    prompt: str = (
        "Open and read the exact page at the URL—not previews or related content. "
        "Write a concise, information-dense summary (~150-200 words) that captures the main purpose, key arguments, and critical details. "
        "Include names, dates, figures, and statistics. Use full sentences, no bullet points. "
        "End with one sentence that distills the page's core message. Focus only on high-signal content."
    ),
    model: str = core.WEB_MODEL_ID,
    enable_citations: bool = True,
) -> str:
    """
    Async wrapper around the Web-Search tool.

    - Sends the URL as input plus the built-in `web_search_preview` tool.
    - Returns plain-text `output_text`.
    - Context size controls how much detail the model should include (low, medium, high).
    """
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

# ==============================================
# Text utilities
# ==============================================

async def chat(
    messages: list[dict],
    model=core.MSG_MODEL_ID,
) -> str:
    """
    Send a basic chat completion request to OpenAI and return the response text.

    Example message format:
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
    resp = await aoai.chat.completions.create(
       model=model,
       messages=messages,
    )

    return resp.choices[0].message.content.strip()
