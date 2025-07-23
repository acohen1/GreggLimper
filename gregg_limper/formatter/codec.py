import json

def dump(rec: dict, *, fmt: str = "json") -> str:
    """
    Serialize a media-record dict from handlers to a string.
    fmt = "json"  -> compact JSON line
    fmt = "kv"    -> [type] key=value | ...
    """
    if fmt == "json":
        return json.dumps(rec, ensure_ascii=False, separators=(",", ":"))
    if fmt == "kv":
        head = f"[{rec['type']}] "
        body = " | ".join(
            f"{k}={v}".replace("|", "\\|") for k, v in rec.items() if k != "type"
        )
        return head + body
    raise ValueError(f"Unsupported fmt: {fmt}")
