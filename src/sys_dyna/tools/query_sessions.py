from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any, Callable

from .base import Tool, ToolDefinition, ToolError, ToolResult


MAX_LIMIT = 50
DEFAULT_LIMIT = 20
EXCERPT_LEN = 400


DEFINITION = ToolDefinition(
    name="query_sessions",
    description=(
        "Search past chat sessions stored in the data warehouse. "
        "Returns a list of sessions whose chat log matches all given keywords "
        "(AND match), with optional filters for model name, user, and minimum "
        "creation date. Use this first when the user asks about prior analyses."
    ),
    parameters={
        "type": "object",
        "properties": {
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Keywords ANDed against the chat log (case-insensitive).",
            },
            "model_name": {
                "type": "string",
                "description": "Filter by simulation model name.",
            },
            "user_id": {
                "type": "string",
                "description": "Restrict to sessions created by this user_id.",
            },
            "since": {
                "type": "string",
                "description": "ISO8601 timestamp; only sessions created at/after this point are returned.",
            },
            "limit": {
                "type": "integer",
                "description": f"Max rows to return (1..{MAX_LIMIT}).",
            },
        },
        "required": ["keywords"],
    },
)


class QuerySessionsTool(Tool):
    """F-02: parameterised SQL search. Free SQL is intentionally not supported."""

    definition = DEFINITION

    def __init__(self, connection_factory: Callable[[], sqlite3.Connection]) -> None:
        self._connect = connection_factory

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        keywords = arguments.get("keywords") or []
        if not isinstance(keywords, list) or not keywords:
            raise ToolError("invalid_argument", "keywords must be a non-empty list of strings")
        if not all(isinstance(k, str) and k.strip() for k in keywords):
            raise ToolError("invalid_argument", "each keyword must be a non-empty string")

        model_name = arguments.get("model_name")
        user_id = arguments.get("user_id")
        since = arguments.get("since")
        limit_raw = arguments.get("limit", DEFAULT_LIMIT)
        try:
            limit = int(limit_raw)
        except (TypeError, ValueError) as e:
            raise ToolError("invalid_argument", "limit must be an integer") from e
        limit = max(1, min(limit, MAX_LIMIT))

        where: list[str] = []
        params: list[Any] = []
        for kw in keywords:
            where.append("LOWER(chat_log) LIKE ?")
            params.append(f"%{kw.lower()}%")
        if model_name:
            where.append("model_name = ?")
            params.append(model_name)
        if user_id:
            where.append("user_id = ?")
            params.append(user_id)
        if since is not None:
            if not isinstance(since, str):
                raise ToolError("invalid_argument", "since must be an ISO8601 string")
            try:
                datetime.fromisoformat(since.replace("Z", "+00:00"))
            except ValueError as e:
                raise ToolError(
                    "invalid_argument", "since must be a valid ISO8601 timestamp"
                ) from e
            where.append("created_at >= ?")
            params.append(since)

        sql = (
            "SELECT session_id, created_at, user_id, model_name, chat_log "
            "FROM sessions "
            f"WHERE {' AND '.join(where)} "
            "ORDER BY created_at DESC "
            "LIMIT ?"
        )
        bind = [*params, limit]

        conn = self._connect()
        try:
            rows = conn.execute(sql, bind).fetchall()
        finally:
            conn.close()

        results: list[dict[str, Any]] = []
        for r in rows:
            excerpt = _summarise_chat_log(r["chat_log"])
            results.append(
                {
                    "session_id": r["session_id"],
                    "created_at": r["created_at"],
                    "user_id": r["user_id"],
                    "model_name": r["model_name"],
                    "chat_excerpt": excerpt,
                }
            )

        return ToolResult(payload={"sessions": results, "count": len(results)})


def _summarise_chat_log(raw: str | None) -> str:
    """The chat log is stored as JSON. Try to extract human-readable text."""
    if not raw:
        return ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw[:EXCERPT_LEN]
    if isinstance(data, list):
        parts: list[str] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            role = entry.get("role", "?")
            content = entry.get("content", "")
            parts.append(f"[{role}] {content}")
        joined = " | ".join(parts)
        return joined[:EXCERPT_LEN]
    return str(data)[:EXCERPT_LEN]
