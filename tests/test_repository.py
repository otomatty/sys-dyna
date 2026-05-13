from __future__ import annotations

from sys_dyna.db.connection import _connect
from sys_dyna.repository import sessions as sessions_repo
from sys_dyna.repository import simulation_results as sim_repo
from sys_dyna.repository import tool_call_logs as logs_repo
from sys_dyna.repository import users as users_repo


def test_users_upsert_get(db_path):
    conn = _connect(db_path)
    try:
        users_repo.upsert(conn, users_repo.UserRow("alice", "Alice", "Eng"))
        fetched = users_repo.get(conn, "alice")
        assert fetched is not None
        assert fetched.display_name == "Alice"

        users_repo.upsert(conn, users_repo.UserRow("alice", "Alice Updated", None))
        fetched = users_repo.get(conn, "alice")
        assert fetched is not None
        assert fetched.display_name == "Alice Updated"
        assert fetched.department is None
    finally:
        conn.close()


def test_sessions_round_trip(db_path):
    conn = _connect(db_path)
    try:
        users_repo.upsert(conn, users_repo.UserRow("alice", "Alice", "Eng"))
        rec = sessions_repo.create_empty(conn, "s1", "alice", "MarketingMix_v3")
        assert rec.chat_log == []

        sessions_repo.append_messages(
            conn,
            "s1",
            [
                sessions_repo.ChatMessage(role="user", content="hello"),
                sessions_repo.ChatMessage(role="assistant", content="world"),
            ],
        )
        fetched = sessions_repo.get(conn, "s1")
        assert fetched is not None
        assert len(fetched.chat_log) == 2
        assert fetched.chat_log[0].content == "hello"
        assert fetched.chat_log[1].role == "assistant"
    finally:
        conn.close()


def test_simulation_results_filtering(seeded_db_path):
    conn = _connect(seeded_db_path)
    try:
        rec = sim_repo.get_latest_for_session(conn, "sess-2025-12-ad-revenue")
        assert rec is not None
        assert "revenue" in rec.time_series_data
        meta = sim_repo.list_metadata_for_session(conn, "sess-2025-12-ad-revenue")
        assert len(meta) == 1
    finally:
        conn.close()


def test_tool_call_log_round_trip(seeded_db_path):
    conn = _connect(seeded_db_path)
    try:
        log = logs_repo.ToolCallLog(
            session_id="sess-2025-12-ad-revenue",
            tool_name="query_sessions",
            tool_input={"keywords": ["広告"]},
            tool_output={"count": 1, "sessions": []},
            called_at=logs_repo.now_iso(),
            duration_ms=42,
        )
        logs_repo.record(conn, log)
        rows = logs_repo.list_for_session(conn, "sess-2025-12-ad-revenue")
        assert len(rows) == 1
        assert rows[0].tool_name == "query_sessions"
        assert rows[0].tool_input == {"keywords": ["広告"]}
    finally:
        conn.close()
