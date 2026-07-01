from __future__ import annotations

import atexit
import json
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Union

try:
    # v4 requires get_client and propagate_attributes
    from langfuse import get_client, propagate_attributes
except ImportError:
    get_client = None  # type: ignore


class JsonlTraceSink:
    """Tiny local observability adapter."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def record(self, event: str, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"ts": time.time(), "event": event, "payload": payload}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(row, default=_json_default, ensure_ascii=False)
                + "\n"
            )


class LangfuseTraceSink:
    """Langfuse v4 adapter strictly adhering to the JsonlTraceSink contract."""

    def __init__(self, trace_name: str = "agent_workflow") -> None:
        if get_client is None:
            raise ImportError(
                "Langfuse SDK is not installed. Please run: pip install langfuse"
            )

        self.langfuse = get_client()
        self.trace_name = trace_name

        # 1. Generate a single overarching Trace ID for the lifespan of this Run
        self.trace_id = self.langfuse.create_trace_id()

        # 2. OpenTelemetry requires a dummy 16-hexchar parent span to group independent spans
        self.parent_span_id = "1234567890abcdef"

        atexit.register(self.langfuse.flush)

    def record(self, event: str, payload: dict[str, Any]) -> None:
        safe_payload = json.loads(json.dumps(payload, default=_json_default))

        # 3. propagate_attributes ensures the underlying Trace inherits our specific trace_name
        with propagate_attributes(trace_name=self.trace_name):
            # 4. We append a new span explicitly mapped to our sink's Trace ID
            with self.langfuse.start_as_current_observation(
                    as_type="span",
                    name=event,
                    trace_context={
                        "trace_id": self.trace_id,
                        "parent_span_id": self.parent_span_id
                    }
            ) as span:
                span.update(metadata=safe_payload)


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return str(value)


def build_trace_sink(config: dict) -> Union[JsonlTraceSink, LangfuseTraceSink]:
    """Auto-switches to Langfuse if the environment is configured."""

    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        return LangfuseTraceSink(
            trace_name=config.get("run_name", "local_agent_run")
        )

    return JsonlTraceSink(Path(config.get("trace_path", "runs/traces.jsonl")))