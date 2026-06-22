from __future__ import annotations

from second_brain.domain import SourceDocument, TextChunk

def _word_windows(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap < 0:
        raise ValueError("chunk overlap cannot be negative")
    if overlap >= size:
        raise ValueError("chunk overlap must be smaller than chunk size")

    words = text.split()
    step = size - overlap
    chunks: list[str] = []

    for start in range(0, len(words), step):
        chunk_words = words[start: start + size]
        if len(chunk_words) < 40:
            continue
        chunks.append(" ".join(chunk_words))

    return chunks


class WindowChunker:
    """Simple overlapping window  chunker.

    This uses  word counts instead of tokenizer counts on purpose: it makes the
    lab portable and dependency-light. Replace this class with a tokenizer-based
    implementation when you need exact model-token accounting.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: SourceDocument) -> list[TextChunk]:
        chunks = _word_windows(document.text, self.chunk_size, self.chunk_overlap)

        return [
            TextChunk(
                id=f"{document.id}-chunk-{idx}",
                document_id=document.id,
                text=text,
                metadata={
                    **document.metadata,
                    "document_id": document.id,
                    "chunk_index": idx,
                    "chunking_strategy": "windows",
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                },
            )
            for idx, text in enumerate(chunks)
        ]

class ParentChildChunker:
    """Small-child search with larger-prent context.

    We index  child chunks for precise search, then store parent text
    as metadata so the agent receives richer context after retrieval.
    """

    def __init__(
            self,
            parent_chunk_size: int,
            parent_chunk_overlap: int,
            child_chunk_size: int,
            child_chunk_overlap: int,
    ) -> None:
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap

    def chunk(self, document: SourceDocument) -> list[TextChunk]:
        parent_chunks = _word_windows(
            document.text, self.parent_chunk_size, self.parent_chunk_overlap
        )

        child_chunks: list[TextChunk] = []
        for parent_idx, parent_text in enumerate(parent_chunks):
            children = _word_windows(
                parent_text, self.child_chunk_size, self.child_chunk_overlap
            )

            for child_idx, child_text in enumerate(children):
                child_chunks.append(
                    TextChunk(
                        id=f"{document.id}-parent-{parent_idx}-child-{child_idx}",
                        document_id=document.id,
                        text=child_text,
                        metadata={
                            **document.metadata,
                            "document_id": document.id,
                            "parent_index": parent_idx,
                            "child_index": child_idx,
                            "chunking_strategy": "parent_child",
                            "parent_chunk_size": self.parent_chunk_size,
                            "parent_chunk_overlap": self.parent_chunk_overlap,
                            "child_chunk_size": self.child_chunk_size,
                            "child_chunk_overlap": self.child_chunk_overlap,
                            "retrieval_text": parent_text
                        },
                    )
                )

        return  child_chunks


def build_chunker(config: dict) -> WindowChunker | ParentChildChunker:
    strategy = config["strategy"]

    if strategy == "window":
        return WindowChunker(
            chunk_size=int(config["chunk_size"]),
            chunk_overlap=int(config["chunk_overlap"])
        )

    if strategy == "parent_child":
        return ParentChildChunker(
            parent_chunk_size=int(config["parent_chunk_size"]),
            parent_chunk_overlap=int(config["parent_chunk_overlap"]),
            child_chunk_size=int(config["child_chunk_size"]),
            child_chunk_overlap=int(config["child_chunk_overlap"])
        )

    raise ValueError(f"Unsupported chunking strategy: {strategy}")
