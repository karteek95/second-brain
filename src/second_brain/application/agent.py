from __future__ import annotations

import json
import uuid
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from second_brain.application.ports import ChatModel, TraceSink, VectorIndex
from second_brain.domain import RetrievalHit


class AgentPlan(BaseModel):
    action: Literal["search_knowledge_base", "final_answer"]
    query: Optional[str] = Field(default=None, description="The query to search in the vector DB")
    answer: Optional[str] = Field(default=None, description="The final answer to return to the user")


class AgentSynthesis(BaseModel):
    answer: str = Field(description="The final synthesized text answer")
    is_grounded: Literal[0, 1] = Field(description="1 if output is grounded in context, 0 otherwise")
    is_relevant: Literal[0, 1] = Field(description="1 if answer is relevant to question, 0 otherwise")


class AgenticRagUseCase:
    """Small tool-using RAG agent"""

    def __init__(
            self,
            vector_index: VectorIndex,
            chat_model: ChatModel,
            trace_sink: TraceSink,
    ) -> None:
        self.vector_index = vector_index
        self.chat_model = chat_model
        self.trace_sink = trace_sink

    def answer(self, question: str, k: int = 5, max_steps: int = 3) -> str:
        run_id = str(uuid.uuid4())
        observations: list[dict[str, Any]] = []

        for step in range(max_steps):
            plan = self._plan(question, observations, run_id, step)

            if plan.get("action") == "search_knowledge_base":
                query = str(plan.get("query") or question)
                hits = self.vector_index.search(query, k=k)
                observations.append(
                    {
                        "query": query,
                        "hits": [self._hit_to_trace(hit) for hit in hits],
                    }
                )
                self.trace_sink.record(
                    "retrieval",
                    {
                        "run_id": run_id,
                        "query": query,
                        "collection": self.vector_index.collection_name,
                        "hits": [self._hit_to_trace(hit) for hit in hits]
                    }
                )
                continue

            if plan.get("action") == "final_answer":
                answer = str(plan.get("answer") or "")
                self.trace_sink.record(
                    "final_answer",
                    {"run_id": run_id, "question": question, "answer": answer}
                )
                return answer

        answer = self._synthesize(question, observations, run_id)
        self.trace_sink.record(
            "final_answer",
            {"run_id": run_id, "question": question, "answer": answer}
        )
        return answer

    def _plan(self, question: str, observations: list[dict[str, Any]], run_id: str, step: int) -> dict[str, Any]:
        # Drastically simplified prompt! We no longer need to explain the JSON format.
        prompt = f"""
        You are an agentic RAG planner.
        You have one tool: search_knowledge_base.

        Question:
        {question}

        Previous observations:
        {json.dumps(observations, ensure_ascii=False)[:8000]}
        """

        messages = [{"role": "user", "content": prompt}]

        # Pass the Pydantic schema directly to Ollama
        raw_output = self.chat_model.generate(messages, response_format=AgentPlan)

        self.trace_sink.record(
            "agent_plan",
            {
                "run_id": run_id,
                "step": step,
                "input": messages,
                "output": raw_output
            },
        )

        try:
            # We still validate it to map the string to our dictionary
            plan_obj = AgentPlan.model_validate_json(raw_output)
            plan = plan_obj.model_dump(exclude_none=True)
        except ValidationError as e:
            print(f"Pydantic Validation Error: {e}")
            return {"action": "search_knowledge_base", "query": question}

        return plan

    def _synthesize(self, question: str, observations: list[dict[str, Any]], run_id: str) -> str:
        context = self._format_context(observations)

        # Simplified prompt! Ollama knows the schema already.
        prompt = f"""
        Answer the question using only the retrieved context.

        Question:
        {question}

        Context:
        {context}
        """

        messages = [{"role": "user", "content": prompt}]

        # Pass the Pydantic schema directly to Ollama
        raw_output = self.chat_model.generate(messages, response_format=AgentSynthesis)

        try:
            synthesis_obj = AgentSynthesis.model_validate_json(raw_output)
            final_answer = synthesis_obj.answer

            eval_metrics = {
                "grounded_score": synthesis_obj.is_grounded,
                "relevance_score": synthesis_obj.is_relevant
            }
        except ValidationError as e:
            print(f"Pydantic Synthesis Validation Error: {e}")
            final_answer = raw_output
            eval_metrics = {"grounded_score": None, "relevance_score": None}

        self.trace_sink.record(
            "agent_synthesize",
            {
                "run_id": run_id,
                "input": messages,
                "output": raw_output,
                "metrics": eval_metrics
            },
        )

        return final_answer

    @staticmethod
    def _format_context(observations: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for observation in observations:
            for hit in observation["hits"]:
                meta = hit["metadata"]
                blocks.append(
                    "\n".join(
                        [
                            f"[document={meta.get('document_id')} chunk={meta.get('chunk_index', meta.get('child_index'))}]",
                            hit["text"],
                        ]
                    )
                )
        return "\n\n".join(blocks)

    @staticmethod
    def _hit_to_trace(hit: RetrievalHit) -> dict[str, Any]:
        return {
            "text": hit.text,
            "metadata": hit.metadata,
            "score": hit.score
        }