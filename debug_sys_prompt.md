## Message Schema
Cached conversation history is provided in JSON format. Each message has the form:

```json
{
  "author": "display_name",
  "fragments": [
    {"type": "text", "description": "Hello world!"},
    {"type": "image", "title": "sunset.jpg", "caption": "a red-orange sky"},
    {"type": "youtube", "title": "<title>", "description": "<video summary>",
     "thumbnail_url": "...", "thumbnail_caption": "..."},
    {"type": "link", "title": "<url>", "description": "<summary>"},
    {"type": "gif", "title": "<cleaned-title>", "caption": "<frame description>"}
  ]
}
```

Do not respond in this format unless explicitly instructed. This schema is only for interpreting cached messages.


## Semantic Memory (top-k, JSON)
```json
[
  {
    "author": "Alex",
    "title": null,
    "content": "Oracle up 40% today off of earnings... raised revenue guidance for their cloud/data infra from $18bn to $144bn <a:Woow:1007797017256407162>",
    "type": "text",
    "url": null
  },
  {
    "author": "Alex",
    "title": null,
    "content": "Apple",
    "type": "text",
    "url": null
  },
  {
    "author": "Alex",
    "title": null,
    "content": "<@1299961656595710024> /rag_opt_in",
    "type": "text",
    "url": null
  }
]
```