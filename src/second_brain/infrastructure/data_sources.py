from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from datasets import load_dataset

from second_brain.domain import SourceDocument

class HuggingFaceSummarizationDataSource:
    """Loads the author's ready dataset from Hugging Face.

    The dataset was created for summarization fine-tuning, not as a perfect RAG
    corpus. For chunking experiments, the instruction text is still useful raw
    material because it contains substantial source documents
    """

    def __init__(
            self,
            dataset_id: str,
            text_field: str = "instruction",
            answer_field: str = "answer",
            splits: list[str] | None = None,
    ):
        self.dataset_id = dataset_id
        self.text_field = text_field
        self.answer_field = answer_field
        self.splits = splits

    def load(self, limit: int | None = None) -> list[SourceDocument]:
        dataset = load_dataset(self.dataset_id)
        split_names = self.splits or list(dataset.keys())
        documents: list[SourceDocument] = []

        for split_name in split_names:
            split = dataset[split_name]
            for row_idx, row in enumerate(split):
                text = str(row.get(self.text_field) or "").strip()
                if len(text) < 200:
                    continue

                documents.append(
                    SourceDocument(
                        id=f"{split_name}-{row_idx}",
                        text=text,
                        metadata={
                            "source": self.dataset_id,
                            "split": split_name,
                            "row": row_idx,
                            "answer": str(row.get(self.answer_field) or ""),
                        },
                    )
                )

                if limit and  len(documents) >= limit:
                    return documents

        return documents

class LocaJsonDocumentSource:
    """Loads  custom documents from JSON when you want to leave HF data behind"""

    def __init__(self, path: Path, text_field: str = "text") -> None:
        self.path = path
        self.text_field = text_field

    def load(self, limit: int | None = None) -> list[SourceDocument]:
        rows: list[dict[str, Any]] = json.loads(self.path.read_text(encoding="utf-8"))
        documents: list[SourceDocument] = []

        for idx, row in enumerate(rows):
            text = str(row.get(self.text_field) or "").strip()
            if not text:
                continue

            metadata = {k: v for k, v in row.items() if k != self.text_field}
            documents.append(
                SourceDocument(id=str(row.get("id", idx)), text=text, metadata=metadata)
            )

            if limit and len(documents) >= limit:
                return documents

        return documents


def build_document_source(config: dict) -> HuggingFaceSummarizationDataSource | LocaJsonDocumentSource:
    source_type = config["type"]

    if source_type == "huggingface_summarization":
        return HuggingFaceSummarizationDataSource(
            dataset_id=config["dataset_id"],
            text_field=config.get("text_field", "instruction"),
            answer_field=config.get("answer_field", "answer"),
            splits=config.get("splits")

        )

    if source_type == "local_json":
        return LocaJsonDocumentSource(
            path=Path(config["path"]),
            text_field=config.get("text_field","text"),
        )

    raise ValueError(f"Unsupported document source: {source_type}")




        
