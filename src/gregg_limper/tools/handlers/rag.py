"""Tool for retrieving contextual memories via RAG."""

from __future__ import annotations

from typing import Any

from gregg_limper.config import rag as rag_cfg
from gregg_limper.memory import rag

from .. import Tool, ToolContext, ToolResult, ToolSpec, ToolExecutionError, register_tool

_MAX_RESULTS = max(1, rag_cfg.VECTOR_SEARCH_K)
_DEFAULT_RESULTS = min(3, _MAX_RESULTS)


@register_tool(
    ToolSpec(
        name="retrieve_context",
        description="Search the server's history for past events or context. Use this tool when the user talks about something that happened in the past, references previous conversations, or when you need to recall specific details from the server's history to answer a question. This is useful for providing context-aware responses based on prior interactions.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query.",
                },
                "k": {
                    "type": "integer",
                    "description": f"Desired number of results (capped at server limit {_MAX_RESULTS}).",
                    "minimum": 1,
                    "maximum": _MAX_RESULTS,
                    "default": _DEFAULT_RESULTS,
                },
            },
            "required": ["query"],
        },
    )
)
class RetrieveContextTool(Tool):
    """Fetch cached context using the existing RAG vector search."""

    async def run(self, *, context: ToolContext, **kwargs: Any) -> ToolResult:
        """Return a ranked list of matching fragments or a fallback message."""

        query = kwargs.get("query")
        if not query or not isinstance(query, str):
            raise ToolExecutionError("'query' must be supplied as a string")

        requested_k = kwargs.get("k", _DEFAULT_RESULTS)
        if not isinstance(requested_k, int):
            raise ToolExecutionError("'k' must be an integer")
        k = max(1, min(_MAX_RESULTS, requested_k))

        guild_id = context.guild_id
        channel_id = context.channel_id
        if guild_id is None or channel_id is None:
            raise ToolExecutionError("Missing guild or channel context for retrieval")

        results = await rag.vector_search(guild_id, channel_id, query, k=k)
        if not results:
            return ToolResult(context_content="No related context was found.")

        lines: list[str] = []
        for idx, row in enumerate(results, start=1):
            snippet = row.get("content") or row.get("title") or "(no content)"
            author = row.get("author_id") or "unknown"
            lines.append(f"{idx}. {snippet} (author: {author}, message: {row.get('message_id')})")

        return ToolResult(context_content="\n".join(lines))
