"""Messages between the IPython session and the GUI process.

One JSON object per line, over a Unix socket in a directory only this user can
open. The GUI sends requests ``{"id": n, "op": ..., ...}``; the session answers
each with ``{"reply": n, "ok": bool, "result": ...}`` or
``{"reply": n, "ok": false, "error": {...}}``, and pushes events
``{"event": ..., ...}`` as they happen.
"""

from __future__ import annotations

import datetime
import enum
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

#: Arrays with more elements than this are sent as their shape and dtype only.
MAX_ARRAY_ELEMENTS = 1000


def to_jsonable(value: Any) -> Any:
    """Return ``value`` as plain JSON types; anything else becomes its ``repr``."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, enum.Enum):
        return to_jsonable(value.value)
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    np = sys.modules.get("numpy")
    if np is not None:
        if isinstance(value, np.ndarray):
            if value.size > MAX_ARRAY_ELEMENTS:
                return {"shape": list(value.shape), "dtype": str(value.dtype)}
            return to_jsonable(value.tolist())
        if isinstance(value, np.generic):
            return to_jsonable(value.item())
    return repr(value)


def encode(message: Mapping[str, Any]) -> bytes:
    """One message as a line of JSON."""
    return (json.dumps(to_jsonable(message), separators=(",", ":")) + "\n").encode()


def decode(line: bytes) -> dict[str, Any]:
    """Return the message on one line; raise ``ValueError`` if not a JSON object."""
    message = json.loads(line)
    if not isinstance(message, dict):
        raise ValueError(f"Expected a JSON object, got {type(message).__name__}")
    return message


def new_socket_path() -> Path:
    """Return a socket path in a fresh directory that only this user can open."""
    # mkdtemp creates the directory with mode 0700.
    directory = tempfile.mkdtemp(
        prefix="bluesky-gui-", dir=os.environ.get("XDG_RUNTIME_DIR") or None
    )
    return Path(directory) / "session.sock"
