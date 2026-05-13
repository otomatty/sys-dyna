from __future__ import annotations

import sqlite3
from typing import Any, Callable

from ..repository import simulation_results as sim_repo
from .base import Tool, ToolDefinition, ToolError, ToolResult


DEFINITION = ToolDefinition(
    name="get_simulation_results",
    description=(
        "Retrieve the numerical (time-series) simulation results for a past "
        "session. Optionally restrict to specific variables. Prefer this over "
        "get_session_full when only numbers are needed for comparison."
    ),
    parameters={
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "Target session identifier.",
            },
            "variable_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "If provided, return only these variables.",
            },
        },
        "required": ["session_id"],
    },
)


class GetSimulationResultsTool(Tool):
    """F-04"""

    definition = DEFINITION

    def __init__(self, connection_factory: Callable[[], sqlite3.Connection]) -> None:
        self._connect = connection_factory

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        session_id = arguments.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ToolError("invalid_argument", "session_id must be a non-empty string")
        variable_names = arguments.get("variable_names")
        if variable_names is not None:
            if not isinstance(variable_names, list) or not all(
                isinstance(v, str) for v in variable_names
            ):
                raise ToolError(
                    "invalid_argument",
                    "variable_names must be a list of strings when provided",
                )

        conn = self._connect()
        try:
            record = sim_repo.get_latest_for_session(conn, session_id)
        finally:
            conn.close()

        if record is None:
            raise ToolError(
                "not_found",
                f"no simulation results recorded for session {session_id}",
            )

        data = record.time_series_data
        if variable_names:
            filtered = {k: v for k, v in data.items() if k in set(variable_names)}
            missing = [v for v in variable_names if v not in data]
        else:
            filtered = data
            missing = []

        payload = {
            "result_id": record.result_id,
            "session_id": record.session_id,
            "created_at": record.created_at,
            "variables": filtered,
            "missing_variables": missing,
        }
        return ToolResult(payload=payload)
