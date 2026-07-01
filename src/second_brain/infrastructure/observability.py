from __future__ import annotations

import atexit
import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Union, Literal

try:
    from langfuse import get_client, propagate_attributes
except ImportError:
    get_client = None  # type: ignore

import threading  # <-- Add this import at the top


class JsonlTraceSink:
    """Tiny local observability adapter."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()  # Fix: Thread lock prevents Windows file-write crashes

    def record(self, event: str, payload: dict[str, Any]) -> None:
        # Enforce exclusive access to the file resource
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            row = {"ts": time.time(), "event": event, "payload": payload}
            with self.path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(row, default=_json_default, ensure_ascii=False)
                    + "\n"
                )

    @contextmanager
    def observe_stream(self, event: str, input_data: Any, metadata: dict):
        class DummyObs:
            def __init__(self):
                self.data = {}

            def update(self, **kwargs):
                self.data.update(kwargs)

        obs = DummyObs()
        yield obs

        payload = {"input": input_data, **metadata, **obs.data}
        self.record(event, payload)


class LangfuseTraceSink:
    """Langfuse v4 adapter strictly adhering to the JsonlTraceSink contract."""

    def __init__(self, trace_name: str = "agent_workflow") -> None:
        if get_client is None:
            raise ImportError(
                "Langfuse SDK is not installed. Please run: pip install langfuse"
            )

        self.langfuse = get_client()
        self.trace_name = trace_name
        self.trace_id = self.langfuse.create_trace_id()
        self.parent_span_id = "1234567890abcdef"

        atexit.register(self.langfuse.flush)

    def record(self, event: str, payload: dict[str, Any]) -> None:
        safe_payload = json.loads(json.dumps(payload, default=_json_default))

        span_input = safe_payload.pop("question", safe_payload.pop("input", None))
        span_output = safe_payload.pop("answer", safe_payload.pop("output", None))

        with propagate_attributes(trace_name=self.trace_name):
            observation_type: Literal["generation", "span"] = "generation" if event == "chat_completion" else "span"

            with self.langfuse.start_as_current_observation(
                    as_type=observation_type,
                    name=event,
                    trace_context={  # type: ignore
                        "trace_id": self.trace_id,
                        "parent_span_id": self.parent_span_id
                    }
            ) as obs:
                obs.update(
                    input=span_input,
                    output=span_output,
                    metadata=safe_payload
                )

        self.langfuse.flush()

    @contextmanager
    def observe_stream(self, event: str, input_data: Any, metadata: dict):
        """Dedicated context manager for wrapping live generators."""
        with propagate_attributes(trace_name=self.trace_name):
            observation_type: Literal["generation", "span"] = "generation" if event == "chat_completion" else "span"

            # The "stopwatch" starts when this block opens
            with self.langfuse.start_as_current_observation(
                    as_type=observation_type,
                    name=event,
                    trace_context={  # type: ignore
                        "trace_id": self.trace_id,
                        "parent_span_id": self.parent_span_id
                    }
            ) as obs:
                obs.update(input=input_data, metadata=metadata)

                # Yield control back to the FastAPI endpoint to run the stream
                yield obs

        # The "stopwatch" stops exactly when the block closes. Flush the time to the cloud.
        self.langfuse.flush()


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return str(value)


def build_trace_sink(config: dict) -> Union[JsonlTraceSink, LangfuseTraceSink]:
    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        return LangfuseTraceSink(
            trace_name=config.get("run_name", "local_agent_run")
        )

    return JsonlTraceSink(Path(config.get("trace_path", "runs/traces.jsonl")))