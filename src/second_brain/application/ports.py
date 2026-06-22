from __future__ import annotations

from cmd import PROMPT
from typing import Any, Protocol

from  second_brain.domain import RetrievalHit, SourceDocument, TextChunk

class  DocumentSource(Protocol):
    """
    Application port for data loading.

    Hugging Face, local JSON, S3, MongoDB, or Notion can all implement this
    interface. The rest of the app only needs SourceDocument objects
    """

    def load(self, limit: int | None = None) -> list[SourceDocument]:
        ...

class Chunker(Protocol):
    """Application port of chunking experiments"""
    def chunk(self, document: SourceDocument) -> list[TextChunk]:
        ...

class VectorIndex(Protocol):
    """Application port for vector DBs.

    chroma is the default local adapter, but FAISS, Qdrant, Milvus, MongoDB
    Atlas, or Elastic search can be added without changing the use cases.
    """

    @property
    def collection_name(self) -> str:
        ...

    def reset(self) -> None:
        ...

    def add(self, chunks: list[TextChunk]) -> None:
        ...

    def search(self, query: str, k: int) -> list[RetrievalHit]:
        ...


class ChatModel(Protocol):
    """Application port for the LLM used by the agent"""

    def generate(self, messages: list[dict[str, str]], json_mode: bool = False) -> str:
        ...

class TraceSink(Protocol):
    """Application port for observability.

    JSONL is internationally simple for local experiments. An opik adapter can
    implement the same method when you want hosted LLMops
    """

    def  record(self, event: str, payload: dict[str, Any]) -> None:
        ...





