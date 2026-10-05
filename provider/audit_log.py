"""
Structured JSON audit logging for every tool call attempt, successful or
rejected. One JSON object per line (JSONL) so it's trivially greppable and
streamable.

Sensitive fields (the auth token itself, and anything else added to
REDACT_KEYS) are never written to disk in full -- we log a short fingerprint
instead so that repeated calls with the same token are still correlatable
without leaking the credential into a log file.
"""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REDACT_KEYS = {"token", "auth_token", "password", "secret", "api_key"}

DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent / "audit.log.jsonl"


def _fingerprint(value: str) -> str:
    """Short, non-reversible fingerprint so redacted values stay correlatable."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _redact(arguments: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in arguments.items():
        if key.lower() in REDACT_KEYS and isinstance(value, str):
            redacted[key] = f"<redacted:{_fingerprint(value)}>"
        else:
            redacted[key] = value
    return redacted


class AuditLogger:
    def __init__(self, path: Path | str = DEFAULT_LOG_PATH):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.path.touch(exist_ok=True)

    def log_call(
        self,
        *,
        caller: str,
        tool_name: str,
        arguments: dict[str, Any],
        outcome: str,          # "success" | "rejected" | "error"
        error_code: str | None = None,
        error_message: str | None = None,
        latency_ms: float | None = None,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "caller": caller,
            "tool_name": tool_name,
            "arguments": _redact(arguments),
            "outcome": outcome,
            "error_code": error_code,
            "error_message": error_message,
            "latency_ms": latency_ms,
        }
        line = json.dumps(entry, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        """Convenience for tests: read back every entry written so far."""
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
