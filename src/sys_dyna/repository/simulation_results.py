from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class SimulationResultRecord:
    result_id: str
    session_id: str
    time_series_data: dict[str, list[dict[str, Any]]]
    created_at: str


def upsert(conn: sqlite3.Connection, record: SimulationResultRecord) -> None:
    conn.execute(
        """
        INSERT INTO simulation_results (result_id, session_id, time_series_data, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(result_id) DO UPDATE SET
            session_id       = excluded.session_id,
            time_series_data = excluded.time_series_data,
            created_at       = excluded.created_at
        """,
        (
            record.result_id,
            record.session_id,
            json.dumps(record.time_series_data, ensure_ascii=False),
            record.created_at,
        ),
    )


def list_metadata_for_session(
    conn: sqlite3.Connection, session_id: str
) -> list[dict[str, str]]:
    rows = conn.execute(
        """
        SELECT result_id, created_at FROM simulation_results
        WHERE session_id = ? ORDER BY created_at DESC
        """,
        (session_id,),
    ).fetchall()
    return [{"result_id": r["result_id"], "created_at": r["created_at"]} for r in rows]


def get_latest_for_session(
    conn: sqlite3.Connection, session_id: str
) -> SimulationResultRecord | None:
    row = conn.execute(
        """
        SELECT result_id, session_id, time_series_data, created_at
        FROM simulation_results
        WHERE session_id = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    return SimulationResultRecord(
        result_id=row["result_id"],
        session_id=row["session_id"],
        time_series_data=json.loads(row["time_series_data"]),
        created_at=row["created_at"],
    )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
