from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class UserRow:
    user_id: str
    display_name: str
    department: str | None


def upsert(conn: sqlite3.Connection, user: UserRow) -> None:
    conn.execute(
        """
        INSERT INTO users (user_id, display_name, department)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            display_name = excluded.display_name,
            department   = excluded.department
        """,
        (user.user_id, user.display_name, user.department),
    )


def get(conn: sqlite3.Connection, user_id: str) -> UserRow | None:
    row = conn.execute(
        "SELECT user_id, display_name, department FROM users WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    if row is None:
        return None
    return UserRow(
        user_id=row["user_id"],
        display_name=row["display_name"],
        department=row["department"],
    )
