from __future__ import annotations

import json
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from second_brain.domain import RetrievalHit, TextChunk

class ChromaVectorIndex:
    """Chroma adapter used for the  local experiment.

    Chroma is selected because it is persistent, local, and does not require
    Docker. The application code depends on the VectorIndex port, not Chroma
    """

    def __init__(self, path: str, collection: str, embedding_model: str) -> None:
        self.path = path
        self._collection_name = collection
        self.embedding_model = embedding_model
        self.client = chromadb.PersistentClient(path=path)
        self.embedding_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        self.collection = self._get_or_create()

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def reset(self) -> None:
        try:
            self.client.delete_collection(self._collection_name)
        except Exception:
            pass
        self.collection = self._get_or_create()

    def add(self, chunks: list[TextChunk]) -> None:
        if not chunks:
            return

        batch_size = 500
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start: start + batch_size]
            self.collection.add(
                ids=[chunk.id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                metadatas=[_sanitize_metadata(chunk.metadata) for chunk in batch],
            )

    def search(self, query: str, k: int) -> list[RetrievalHit]:
            result = self.collection.query(query_texts=[query], n_results=k)

            documents = result["documents"][0]
            metadatas = result["metadatas"][0]
            distances = result["distances"][0]

            hits: list[RetrievalHit] = []
            for text, metadata, distance in zip(documents, metadatas, distances):
                retrieval_text = str(metadata.get("retrieval_text") or text)
                hits.append(
                    RetrievalHit(
                        text=retrieval_text,
                        metadata=metadata,
                        score=float(distance),
                    )
                )

            return hits

    def _get_or_create(self):
        return self.client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space":"cosine"}
        )

def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    clean:dict[str, str | int | float | bool] = {}

    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = json.dumps(value, ensure_ascii=False)

    return clean

def build_vector_index(config: dict) -> ChromaVectorIndex:
    index_type = config["type"]

    if index_type == "chroma":
        return ChromaVectorIndex(
            path=config.get("path", ".chroma"),
            collection=config["collection"],
            embedding_model=config.get(
                "embedding_model", "sentence-transformers/all-MiniLM-L6_v2"
            ),
        )

    raise ValueError(f"Unsupported vector index: {index_type}")
