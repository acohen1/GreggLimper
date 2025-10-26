from __future__ import annotations

"""Dataclass models for formatter fragments.

Fragment schema (output of :meth:`Fragment.to_dict`):

```
{"type": "text", "description": "Hello world!"}
{"type": "image", "title": "sunset.jpg", "caption": "red-orange sky"}
{"type": "gif", "title": "cat", "caption": "looping cat"}
{"type": "youtube", "title": "<title>", "description": "<summary>",
 "thumbnail_url": "...", "thumbnail_caption": "..."}
{"type": "link", "title": "https://example.com", "description": "<og:description>"}
```

All fragments share a consistent set of fields and may define type-specific
extras. Objects keep only non-``None`` attributes when serialized. ``id`` is a
stable identifier assigned during composition.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any


def _drop_nones(d: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of ``d`` without keys mapped to ``None``."""
    return {k: v for k, v in d.items() if v is not None}


FragmentType = Literal["text", "image", "gif", "youtube", "link"]


@dataclass(slots=True)
class Fragment:
    """Base fragment type."""

    type: FragmentType
    id: str = field(init=False, default="")
    title: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None

    def to_markdown(self) -> str:
        """Render the fragment as a concise Markdown bullet."""

        parts = [f"- **{self.type}**"]
        if self.title:
            parts.append(f": {self.title}")
        if self.description:
            parts.append(f" – {self.description}")
        if self.url:
            parts.append(f" ([link]({self.url}))")
        return "".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        base = {
            "type": self.type,
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "url": self.url,
        }
        return _drop_nones(base)
    
    def to_llm(self) -> Dict[str, Any]:
        """Return a dict with only fields useful for LLM consumption."""
        base = {
            "type": self.type,
            "title": self.title,
            "description": self.description,
        }
        return _drop_nones(base)

    def content_text(self) -> str:
        """Return text that should be embedded for RAG."""
        return (self.description or "").strip()

    def __str__(self) -> str:  # pragma: no cover - convenience only
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass(slots=True)
class TextFragment(Fragment):
    """Plain text fragment."""

    type: Literal["text"] = "text"
    # description holds the body text

    def to_markdown(self) -> str:
        body = (self.description or "").strip()
        if len(body) > 160:
            body = f"{body[:157]}…"
        parts = [f"- **{self.type}**"]
        if body:
            parts.append(f": {body}")
        if self.url:
            parts.append(f" ([link]({self.url}))")
        return "".join(parts)


@dataclass(slots=True)
class ImageFragment(Fragment):
    """Still image fragment."""

    type: Literal["image"] = "image"
    caption: str = ""
    thumbnail_url: Optional[str] = None

    def to_markdown(self) -> str:
        parts = [f"- **{self.type}**"]
        if self.title:
            parts.append(f": {self.title}")
        caption = (self.caption or "").strip()
        if caption:
            parts.append(f" – {caption}")
        elif self.description:
            parts.append(f" – {self.description}")
        if self.url:
            parts.append(f" ([image]({self.url}))")
        elif self.thumbnail_url:
            parts.append(f" ([preview]({self.thumbnail_url}))")
        return "".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        base = Fragment.to_dict(self)
        base.update(_drop_nones({"caption": self.caption, "thumbnail_url": self.thumbnail_url}))
        return base
    
    def to_llm(self) -> Dict[str, Any]:
        base = Fragment.to_llm(self)
        base.update(_drop_nones({"caption": self.caption}))
        return base

    def content_text(self) -> str:
        return (self.caption or "").strip()


@dataclass(slots=True)
class GIFFragment(Fragment):
    """Animated image fragment."""

    type: Literal["gif"] = "gif"
    caption: str = ""
    thumbnail_url: Optional[str] = None

    def to_markdown(self) -> str:
        parts = [f"- **{self.type}**"]
        if self.title:
            parts.append(f": {self.title}")
        caption = (self.caption or "").strip()
        if caption:
            parts.append(f" – {caption}")
        if self.url:
            parts.append(f" ([gif]({self.url}))")
        elif self.thumbnail_url:
            parts.append(f" ([preview]({self.thumbnail_url}))")
        return "".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        base = Fragment.to_dict(self)
        base.update(_drop_nones({"caption": self.caption, "thumbnail_url": self.thumbnail_url}))
        return base

    def to_llm(self) -> Dict[str, Any]:
        base = Fragment.to_llm(self)
        base.update(_drop_nones({"caption": self.caption}))
        return base

    def content_text(self) -> str:
        return (self.caption or "").strip()
    

@dataclass(slots=True)
class YouTubeFragment(Fragment):
    """YouTube video fragment."""

    type: Literal["youtube"] = "youtube"
    thumbnail_url: Optional[str] = None
    thumbnail_caption: Optional[str] = None
    channel: Optional[str] = None
    duration: Optional[int] = None

    def _format_duration(self) -> str | None:
        if not self.duration:
            return None
        seconds = max(self.duration, 0)
        minutes, sec = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{sec:02d}"
        return f"{minutes:d}:{sec:02d}"

    def to_markdown(self) -> str:
        parts = [f"- **{self.type}**"]
        if self.title:
            parts.append(f": {self.title}")
        elif self.description:
            parts.append(f": {self.description}")

        details: list[str] = []
        if self.channel:
            details.append(f"channel {self.channel}")
        duration = self._format_duration()
        if duration:
            details.append(f"{duration}")
        if details:
            parts.append(f" ({', '.join(details)})")

        if self.thumbnail_caption:
            parts.append(f" – {self.thumbnail_caption}")
        if self.url:
            parts.append(f" ([watch]({self.url}))")
        elif self.thumbnail_url:
            parts.append(f" ([preview]({self.thumbnail_url}))")
        return "".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        base = Fragment.to_dict(self)
        base.update(
            _drop_nones(
                {
                    "thumbnail_url": self.thumbnail_url,
                    "thumbnail_caption": self.thumbnail_caption,
                    "channel": self.channel,
                    "duration": self.duration,
                }
            )
        )
        return base
    
    def to_llm(self) -> Dict[str, Any]:
        base = Fragment.to_llm(self)
        base.update(
            _drop_nones(
                {
                    "thumbnail_caption": self.thumbnail_caption,
                    "channel": self.channel,
                    "duration": self.duration,
                }
            )
        )
        return base
    
    def content_text(self) -> str:
        return (self.title or self.description or "").strip()

@dataclass(slots=True)
class LinkFragment(Fragment):
    """Generic hyperlink fragment."""

    type: Literal["link"] = "link"
    site_name: Optional[str] = None
    thumbnail_url: Optional[str] = None
    thumbnail_caption: Optional[str] = None

    def to_markdown(self) -> str:
        parts = [f"- **{self.type}**"]
        headline = self.title or self.url
        if headline:
            parts.append(f": {headline}")
        if self.description:
            parts.append(f" – {self.description}")
        if self.site_name:
            parts.append(f" ({self.site_name})")
        if self.url:
            parts.append(f" ([link]({self.url}))")
        elif self.thumbnail_url:
            parts.append(f" ([preview]({self.thumbnail_url}))")
        return "".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        base = Fragment.to_dict(self)
        base.update(
            _drop_nones(
                {
                    "site_name": self.site_name,
                    "thumbnail_url": self.thumbnail_url,
                    "thumbnail_caption": self.thumbnail_caption,
                }
            )
        )
        return base

    def to_llm(self) -> Dict[str, Any]:
        base = Fragment.to_llm(self)
        base.update(
            _drop_nones(
                {
                    "site_name": self.site_name,
                    "thumbnail_caption": self.thumbnail_caption,
                }
            )
        )
        return base


__all__ = [
    "Fragment",
    "TextFragment",
    "ImageFragment",
    "GIFFragment",
    "YouTubeFragment",
    "LinkFragment",
]

# Mapping of fragment type string to dataclass for deserialization
_FRAG_REGISTRY: Dict[str, type[Fragment]] = {
    cls.__dataclass_fields__["type"].default: cls  # type: ignore[index]
    for cls in Fragment.__subclasses__()
}

def fragment_from_dict(data: Dict[str, Any]) -> Fragment:
    """
    Instantiate a concrete :class:`Fragment` from a serialized dict.

    :param data: Serialized fragment dictionary.
    :returns: Constructed :class:`Fragment` subclass.
    """
    typ = data.get("type")
    frag_cls = _FRAG_REGISTRY.get(typ)
    if frag_cls is None:
        raise ValueError(f"Unknown fragment type: {typ}")
    kwargs = {k: v for k, v in data.items() if k not in {"type", "id"}}
    frag = frag_cls(**kwargs)
    frag.id = data.get("id", "")
    return frag

__all__.append("fragment_from_dict")
