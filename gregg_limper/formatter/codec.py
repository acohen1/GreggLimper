import json
from typing import Dict

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

def load(s: str, *, fmt: str | None = None) -> Dict:
    """Inverse of :pyfunc:`dump` - deserialize *s* back to a dict.

    Parameters
    ----------
    s : str
        The serialized media record.
    fmt : {"json", "kv", None}
        Explicit format to parse. If *None*, the function will attempt to
        auto-detect: it treats strings starting with ``{"`` as JSON, otherwise
        assumes ``kv``.
    """
    fmt = fmt or ("json" if s.lstrip().startswith("{") else "kv")

    if fmt == "json":
        return json.loads(s)

    if fmt == "kv":
        if not s.startswith("["):
            raise ValueError("KV format expects leading [type] tag")

        # Split head and body
        head, _, body = s.partition("] ")
        record_type = head.lstrip("[")

        record: Dict[str, str] = {"type": record_type}
        for pair in body.split(" | "):
            if not pair:
                continue
            k, _, v = pair.partition("=")
            record[k] = v.replace("\\|", "|")
        return record

    raise ValueError(f"Unsupported fmt: {fmt}")