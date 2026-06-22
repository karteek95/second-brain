from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

class JsonlTraceSink:
    """Tiny local observability adapter.

    This gives you the important LLMOps habit first: every indexing, retrieval,
    planning, and answer event is recorded. Swap this for Opik later without
    changing the agent or indexing use cases.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def record(self, event: str, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": time.time(),
            "event": event,
            "payload": payload
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=_json_default, ensure_ascii=False) + "\n")

def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return  asdict(value)
    return str(value)

def build_trace_sink(config: dict) -> JsonlTraceSink:
    return JsonlTraceSink(Path(config.get("trace_path", "runs/traces.jsonl")))