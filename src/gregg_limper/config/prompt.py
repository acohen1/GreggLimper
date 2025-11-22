import os


class Prompt:
    def __init__(self, config: dict | None = None) -> None:
        prompt_cfg = (config or {}).get("gregglimper", {}).get("retrieval", {})
        self.VECTOR_SEARCH_K: int = int(
            prompt_cfg.get("rag_vector_search_k", os.getenv("RAG_VECTOR_SEARCH_K", "3"))
        )
