from __future__ import annotations

import argparse
from traceback import print_tb
from urllib.parse import uses_query

from pathlib import Path

from rich import print
from dotenv import load_dotenv

from second_brain.application import (
    AgenticRagUseCase,
    EvaluationUseCase,
    IndexDocumentUseCase,
    build_chunker
)
from second_brain.infrastructure.config import load_config
from second_brain.infrastructure.data_sources import build_document_source
from second_brain.infrastructure.llms import build_chat_model
from second_brain.infrastructure.observability import build_trace_sink
from second_brain.infrastructure.vector_stores import build_vector_index


def build_runtime(config_path: Path):
    load_dotenv()
    config = load_config(config_path)

    source = build_document_source(config["data_source"])
    chunker = build_chunker(config["chunking"])
    vector_index = build_vector_index(config["vector_index"])
    chat_model = build_chat_model(config["llm"])
    trace_sink = build_trace_sink(config["observability"])

    return source, chunker, vector_index, chat_model, trace_sink

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local agentic RAG lab for chunking and vector DB experiments"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    index_cmd = subcommands.add_parser("index", help="Build or rebuild an index.")
    index_cmd.add_argument("--config", type=Path, required=True)
    index_cmd.add_argument("--limit", type=int, default=300)

    ask_cmd = subcommands.add_parser("ask", help="Ask the agent one question.")
    ask_cmd.add_argument("--config", type=Path, required=True)
    ask_cmd.add_argument("--question", type=Path, required=True)
    ask_cmd.add_argument("--k", type=int, default=5)
    ask_cmd.add_argument("--max-steps", type=int, default=3)

    eval_cmd = subcommands.add_parser("eval",help="Run a small local evaluation")
    eval_cmd.add_argument("--config", type=Path, required=True)
    eval_cmd.add_argument("--k", type=int, default=5)
    eval_cmd.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()
    source, chunker, vector_index, chat_model, trace_sink = build_runtime(args.config)

    if args.command == "index":
        use_case = IndexDocumentUseCase(
            source=source,
            chunker=chunker,
            vector_index=vector_index,
            trace_sink=trace_sink
        )
        print(use_case.run(limit=args.limit))
        return

    agent = AgenticRagUseCase(
        vector_index=vector_index,
        chat_model=chat_model,
        trace_sink=trace_sink
    )

    if args.command == "ask":
        print(agent.answer(args.question, k= args.k, max_steps=args.max_steps))
        return

    if args.command == "eval":
        use_case = EvaluationUseCase(
            agent=agent,
            vector_index=vector_index,
            trace_sink=trace_sink
        )

        results = use_case.run(k=args.k, output_path=args.output)
        print(results)
        return

if __name__ == "__main__":
    main()