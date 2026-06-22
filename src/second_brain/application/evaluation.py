from __future__ import annotations

from dataclasses import asdict
from pathlib import  Path
import json

from second_brain.application.agent import AgenticRagUseCase
from second_brain.application.ports import TraceSink, VectorIndex
from second_brain.domain import EvaluationResult

DEFAULT_QUESTIONS = [
    "What is Feature/Training/Inference architecture?",
    "How does chunk overlap affect RAG?",
    "What are vector databases and vector indexes?",
    "What is the role of observability in LLMOps?",
    "What are the common ways to evaluate a RAG system?"
]

class EvaluationUseCase:
    """Local LLMOps use case.

    This intentionally writes a simple JSON report. The tracing port already
    captures detailed events, and the report makes chunking runs easy to compare
    """

    def __init__(
            self,
            agent: AgenticRagUseCase,
            vector_index: VectorIndex,
            trace_sink: TraceSink
    ) -> None:
        self.agent = agent
        self.vector_index = vector_index
        self.trace_sink = trace_sink

    def run(
            self,
            questions: list[str] | None = None,
            k: int = 5,
            output_path: Path | None = None,
            ) -> list[EvaluationResult]:
        questions = questions or DEFAULT_QUESTIONS
        results: list[EvaluationResult] = []

        for question in questions:
            answer = self.agent.answer(question, k=k)
            results.append(
                EvaluationResult(
                    question=question,
                    answer=answer,
                    collection=self.vector_index.collection_name,
                    retrieved_chunks=k,
                    answer_chunks=len(answer)
                )
            )
        output_path = output_path or Path(f"runs/eval_{self.vector_index.collection_name}.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps([asdict(result) for result in results], indent = 2),
            encoding="utf-8"
        )

        self.trace_sink.record(
            "evaluation_finished",
            {
                "collection": self.vector_index.collection_name,
                "questions": len(questions),
                "output_path": str(output_path)
            }
        )

        return results