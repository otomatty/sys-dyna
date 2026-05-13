from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable

from ..db.connection import _connect  # type: ignore[attr-defined]
from .base import Tool, ToolDefinition
from .get_session_full import GetSessionFullTool
from .get_simulation_results import GetSimulationResultsTool
from .query_sessions import QuerySessionsTool


def build_default_tools(db_path: Path) -> dict[str, Tool]:
    """Construct the three production tools (F-02 / F-03 / F-04) bound to a DB path."""

    factory: Callable[[], sqlite3.Connection] = lambda: _connect(db_path)
    return {
        "query_sessions": QuerySessionsTool(factory),
        "get_session_full": GetSessionFullTool(factory),
        "get_simulation_results": GetSimulationResultsTool(factory),
    }


def get_tool_definitions(tools: dict[str, Tool]) -> list[ToolDefinition]:
    return [t.definition for t in tools.values()]
