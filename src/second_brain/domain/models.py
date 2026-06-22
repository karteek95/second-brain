from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class SourceDocument:
    """Provider neutral document used by the application layer"""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class TextChunk:
    """chunk that is  ready to be indexed

    The text may be a normal chunk or a smaller child chunk. parent-child
    retrival stores the larger parent text in metadata so the vector DB can
    search small text but return wider context.
    """

    id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class RetrievalHit:
    """Search result returned by any vector DB adapter"""

    text: str
    metadata: dict[str, Any]
    score: float | None = None

@dataclass(frozen=True)
class EvaluationResult:
    question: str
    answer: str
    collection: str
    retrieved_chunks: int
    answer_chunks: int