from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ToolCallLog:
    session_id: str
    tool_name: str
    tool_input: dict[str, Any] | None
    tool_output: Any
    called_at: str
    duration_ms: int
    log_id: str = field(default_factory=lambda: str(uuid.uuid4()))


def record(conn: sqlite3.Connection, log: ToolCallLog) -> None:
    conn.execute(
        """
        INSERT INTO tool_call_logs (
            log_id, session_id, tool_name, tool_input, tool_output, called_at, duration_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            log.log_id,
            log.session_id,
            log.tool_name,
            json.dumps(log.tool_input, ensure_ascii=False) if log.tool_input is not None else None,
            json.dumps(log.tool_output, ensure_ascii=False, default=str) if log.tool_output is not None else None,
            log.called_at,
            log.duration_ms,
        ),
    )


def list_for_session(conn: sqlite3.Connection, session_id: str) -> list[ToolCallLog]:
    rows = conn.execute(
        """
        SELECT log_id, session_id, tool_name, tool_input, tool_output, called_at, duration_ms
        FROM tool_call_logs WHERE session_id = ? ORDER BY called_at ASC
        """,
        (session_id,),
    ).fetchall()
    out: list[ToolCallLog] = []
    for r in rows:
        out.append(
            ToolCallLog(
                log_id=r["log_id"],
                session_id=r["session_id"],
                tool_name=r["tool_name"],
                tool_input=json.loads(r["tool_input"]) if r["tool_input"] else None,
                tool_output=json.loads(r["tool_output"]) if r["tool_output"] else None,
                called_at=r["called_at"],
                duration_ms=r["duration_ms"],
            )
        )
    return out


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
