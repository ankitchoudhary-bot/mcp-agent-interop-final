"""
Versioned input/output schemas shared by the provider and consumer.

Every tool has an explicit Pydantic model for its input and its output.
SCHEMA_VERSION is bumped whenever any model in this file changes shape in a
way that isn't backward compatible. The provider reports SCHEMA_VERSION in
its discovery resource (see provider/server.py); the consumer checks it on
connect (see consumer/mcp_client.py) and refuses to proceed silently if the
server is running a newer major version than the client was built against.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "v1"

# A client built against this file understands schema versions up to and
# including this one. Anything with a higher MAJOR component is a breaking
# change the client doesn't know how to speak.
SUPPORTED_MAJOR_VERSION = 1


def schema_major_version(version_string: str) -> int:
    """'v1' -> 1, 'v2' -> 2, etc. Raises ValueError on malformed input."""
    if not version_string.startswith("v"):
        raise ValueError(f"Malformed schema version: {version_string!r}")
    return int(version_string[1:])


# --------------------------------------------------------------------------
# Common envelope
# --------------------------------------------------------------------------

class ToolError(BaseModel):
    """
    Typed error shape returned (as tool content, not a raw exception) for
    every rejected or failed call. `code` is machine-readable and stable;
    `message` is human-readable and may change wording over time.
    """
    code: Literal[
        "AUTH_MISSING",
        "AUTH_INVALID",
        "AUTH_SCOPE_DENIED",
        "VALIDATION_ERROR",
        "NOT_FOUND",
        "INTERNAL_ERROR",
        "UNSAFE_QUERY",
    ]
    message: str
    schema_version: str = SCHEMA_VERSION


# --------------------------------------------------------------------------
# calendar.list_events
# --------------------------------------------------------------------------

class CalendarListEventsInput(BaseModel):
    token: str = Field(..., description="Bearer auth token, required on every call.")
    start_date: str = Field(..., description="ISO 8601 date, inclusive, e.g. 2026-09-25")
    end_date: str = Field(..., description="ISO 8601 date, inclusive, e.g. 2026-09-30")

    @field_validator("start_date", "end_date")
    @classmethod
    def _valid_date(cls, v: str) -> str:
        datetime.strptime(v, "%Y-%m-%d")  # raises ValueError if malformed
        return v


class CalendarEvent(BaseModel):
    id: int
    title: str
    start: str  # ISO 8601 datetime
    end: str    # ISO 8601 datetime


class CalendarListEventsOutput(BaseModel):
    events: list[CalendarEvent]
    schema_version: str = SCHEMA_VERSION


# --------------------------------------------------------------------------
# calendar.create_event
# --------------------------------------------------------------------------

class CalendarCreateEventInput(BaseModel):
    token: str
    title: str = Field(..., min_length=1, max_length=200)
    start: str = Field(..., description="ISO 8601 datetime, e.g. 2026-09-26T14:00:00")
    end: str = Field(..., description="ISO 8601 datetime, e.g. 2026-09-26T15:00:00")

    @field_validator("start", "end")
    @classmethod
    def _valid_datetime(cls, v: str) -> str:
        datetime.fromisoformat(v)
        return v


class CalendarCreateEventOutput(BaseModel):
    event: CalendarEvent
    schema_version: str = SCHEMA_VERSION


# --------------------------------------------------------------------------
# database.query
# --------------------------------------------------------------------------

class DatabaseQueryInput(BaseModel):
    token: str
    sql: str = Field(..., min_length=1, max_length=2000)


class DatabaseQueryOutput(BaseModel):
    columns: list[str]
    rows: list[list[str | int | float | None]]
    row_count: int
    schema_version: str = SCHEMA_VERSION


# --------------------------------------------------------------------------
# search.web
# --------------------------------------------------------------------------

class SearchWebInput(BaseModel):
    token: str
    query: str = Field(..., min_length=1, max_length=500)
    max_results: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class SearchWebOutput(BaseModel):
    results: list[SearchResult]
    schema_version: str = SCHEMA_VERSION
