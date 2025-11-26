import os


class LocalLLM:
    def __init__(self, config: dict | None = None) -> None:
        llm_cfg = (config or {}).get("gregglimper", {}).get("local_llm", {})
        use_local_raw = llm_cfg.get("use_local", os.getenv("USE_LOCAL", "0"))
        self.USE_LOCAL: bool = str(use_local_raw).lower() in ("1", "true", "yes")
        self.LOCAL_MODEL_ID: str = str(llm_cfg.get("local_model_id", os.getenv("LOCAL_MODEL_ID", "gpt-oss-20b")))
        self.LOCAL_SERVER_URL: str = str(
            llm_cfg.get("local_server_url", os.getenv("LOCAL_SERVER_URL", "http://localhost:11434"))
        )
