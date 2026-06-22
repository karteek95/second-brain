from __future__ import annotations

from typing import Any

from second_brain.application.ports import Chunker, DocumentSource, TraceSink, VectorIndex

class IndexDocumentUseCase:
    """Feature-pipeline use case: documents -> chunks -> vector index"""

    def __init__(
            self,
            source: DocumentSource,
            chunker: Chunker,
            vector_index: VectorIndex,
            trace_sink: TraceSink
    ) -> None:
        self.source = source
        self.chunker = chunker
        self.vector_index = vector_index
        self.trace_sink = trace_sink

    def run(self, limit: int | None = None) -> dict[str, Any]:
        documents = self.source.load(limit=limit)
        chunks = [
            chunk
            for document in documents
            for chunk in self.chunker.chunk(document)
        ]

        self.vector_index.reset()
        self.vector_index.add(chunks)

        metrics = {
            "collection": self.vector_index.collection_name,
            "documents": len(documents),
            "chunks": len(chunks)
        }

        self.trace_sink.record("index_built", metrics)

        return metrics
