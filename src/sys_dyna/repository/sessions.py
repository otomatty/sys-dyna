from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ChatMessage:
    role: str  # "user" | "assistant" | "tool"
    content: str
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content, "ts": self.ts}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChatMessage":
        return cls(role=d["role"], content=d["content"], ts=d.get("ts", ""))


@dataclass
class SessionRecord:
    session_id: str
    user_id: str
    created_at: str
    updated_at: str
    model_name: str | None
    chat_log: list[ChatMessage]
    final_state: dict[str, Any] | None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def upsert(conn: sqlite3.Connection, session: SessionRecord) -> None:
    chat_log_json = json.dumps([m.to_dict() for m in session.chat_log], ensure_ascii=False)
    final_state_json = (
        json.dumps(session.final_state, ensure_ascii=False) if session.final_state is not None else None
    )
    conn.execute(
        """
        INSERT INTO sessions (session_id, user_id, created_at, updated_at, model_name, chat_log, final_state)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            updated_at  = excluded.updated_at,
            model_name  = excluded.model_name,
            chat_log    = excluded.chat_log,
            final_state = excluded.final_state
        """,
        (
            session.session_id,
            session.user_id,
            session.created_at,
            session.updated_at,
            session.model_name,
            chat_log_json,
            final_state_json,
        ),
    )


def get(conn: sqlite3.Connection, session_id: str) -> SessionRecord | None:
    row = conn.execute(
        """
        SELECT session_id, user_id, created_at, updated_at, model_name, chat_log, final_state
        FROM sessions WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    return _row_to_record(row)


def create_empty(
    conn: sqlite3.Connection,
    session_id: str,
    user_id: str,
    model_name: str | None,
) -> SessionRecord:
    now = _now_iso()
    rec = SessionRecord(
        session_id=session_id,
        user_id=user_id,
        created_at=now,
        updated_at=now,
        model_name=model_name,
        chat_log=[],
        final_state=None,
    )
    upsert(conn, rec)
    return rec


def append_messages(
    conn: sqlite3.Connection,
    session_id: str,
    messages: list[ChatMessage],
) -> None:
    rec = get(conn, session_id)
    if rec is None:
        raise KeyError(f"session not found: {session_id}")
    rec.chat_log.extend(messages)
    rec.updated_at = _now_iso()
    upsert(conn, rec)


def _row_to_record(row: sqlite3.Row) -> SessionRecord:
    chat_log_raw = json.loads(row["chat_log"]) if row["chat_log"] else []
    return SessionRecord(
        session_id=row["session_id"],
        user_id=row["user_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        model_name=row["model_name"],
        chat_log=[ChatMessage.from_dict(m) for m in chat_log_raw],
        final_state=json.loads(row["final_state"]) if row["final_state"] else None,
    )
