from dataclasses import dataclass
import os

@dataclass
class Prompt:
    VECTOR_SEARCH_K: int = int(os.getenv("RAG_VECTOR_SEARCH_K", "3"))                  # Number of nearest neighbors to return in RAG vector search