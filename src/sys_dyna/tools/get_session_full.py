from __future__ import annotations

import sqlite3
from typing import Any, Callable

from ..repository import sessions as sessions_repo
from ..repository import simulation_results as sim_repo
from .base import Tool, ToolDefinition, ToolError, ToolResult


DEFINITION = ToolDefinition(
    name="get_session_full",
    description=(
        "Fetch the full chat log, final variable state, and related simulation "
        "result metadata for a single past session. Use this after locating a "
        "candidate session via query_sessions."
    ),
    parameters={
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "Target session identifier.",
            },
        },
        "required": ["session_id"],
    },
)


class GetSessionFullTool(Tool):
    """F-03"""

    definition = DEFINITION

    def __init__(self, connection_factory: Callable[[], sqlite3.Connection]) -> None:
        self._connect = connection_factory

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        session_id = arguments.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ToolError("invalid_argument", "session_id must be a non-empty string")

        conn = self._connect()
        try:
            record = sessions_repo.get(conn, session_id)
            if record is None:
                raise ToolError("not_found", f"session {session_id} does not exist")
            sim_meta = sim_repo.list_metadata_for_session(conn, session_id)
        finally:
            conn.close()

        payload = {
            "session_id": record.session_id,
            "user_id": record.user_id,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "model_name": record.model_name,
            "chat_log": [m.to_dict() for m in record.chat_log],
            "final_state": record.final_state,
            "simulation_results": sim_meta,
        }
        return ToolResult(payload=payload)
