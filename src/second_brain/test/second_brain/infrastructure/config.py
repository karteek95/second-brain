from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

def load_config(path: Path) -> dict[str, Any]:
    """Load YAML without turning config into global state"""

    return yaml.safe_load(path.read_text(encoding="utf-8"))