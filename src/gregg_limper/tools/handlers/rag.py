"""Tool for retrieving contextual memories via RAG."""

from __future__ import annotations

from typing import Any

from gregg_limper.memory import rag

from .. import Tool, ToolContext, ToolResult, ToolSpec, ToolExecutionError, register_tool


@register_tool(
    ToolSpec(
        name="retrieve_context",
        description="Search cached memories relevant to the given query.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query.",
                },
                "k": {
                    "type": "integer",
                    "description": "Maximum number of results to return (1-6).",
                    "minimum": 1,
                    "maximum": 6,
                    "default": 3,
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

        k = kwargs.get("k", 3)
        if not isinstance(k, int):
            raise ToolExecutionError("'k' must be an integer")
        k = max(1, min(6, k))

        guild_id = context.guild_id
        channel_id = context.channel_id
        if guild_id is None or channel_id is None:
            raise ToolExecutionError("Missing guild or channel context for retrieval")

        results = await rag.vector_search(guild_id, channel_id, query, k=k)
        if not results:
            return ToolResult(content="No related context was found.")

        lines: list[str] = []
        for idx, row in enumerate(results, start=1):
            snippet = row.get("content") or row.get("title") or "(no content)"
            author = row.get("author_id") or "unknown"
            lines.append(f"{idx}. {snippet} (author: {author}, message: {row.get('message_id')})")

        return ToolResult(content="\n".join(lines))
