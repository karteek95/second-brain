from __future__ import annotations

import json
import uuid
from typing import Any

from second_brain.application.ports import ChatModel, TraceSink, VectorIndex
from second_brain.domain import RetrievalHit

class AgenticRagUseCase:
    """Small tool-using RAG agent

    The agent can decide to call a retrieval tool before answering. Keeping
    retrieval as a tool makes the design close to the course's agentic RAG
    module while still small enough to run locally.
    """

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
            plan = self._plan(question, observations)
            self.trace_sink.record(
                "agent_plan",
                {"run_id": run_id, "step": step, "plan": plan},
            )

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

        answer = self._synthesize(question, observations)
        self.trace_sink.record(
            "final_answer",
            {"run_id":run_id, "question": question, "answer": answer}
        )
        return answer

    def _plan(self, question: str, observations: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = f"""
        You are an agentic RAG planner.
        
        You have one tool:
        - search_knowledge_base(query): retrieve relevant chunks  from the  vector DB.
        Return JSON only.
        
        Allowed JSON:
        {{"action": "search_knowledge_base", "query": "..."}}
        {{"action": "final_answer", "query": "..."}}
        
        Question:
        {question}
        
        Previous observations:
        {json.dumps(observations, ensure_ascii=False)[:8000]}
        """

        raw = self.chat_model.generate([{"role": "user", "content": prompt}], json_mode=True)

        try:
            plan = json.loads(raw)
        except json.JSONDecodeError:
            # Local models sometimes ignore JSON mode. The fallback keeps the
            # experiment loop robust instead of failing before retrieval.
            return {"action": "search_knowledge_base", "query": question}

        if not isinstance(plan, dict) or "action" not in plan:
            return {"action": "search_knowledge_base", "query": question}

        return plan

    def _synthesize(self, question: str, observations: list[dict[str, Any]]) -> str:
        context = self._format_context(observations)
        prompt = f"""
        Answer the question using only the retrieved context.
        Also give me binary outputs for "1) whether output is grounded in the retrieved context?,
        2) whether answer relevant to the question or not?"
        Give me binary outputs at the end of the answer.
        
        Question:
        {question}
        
        Context:
        {context}
        """

        return self.chat_model.generate([{"role": "user", "content": prompt}])

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



