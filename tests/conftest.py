from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from sys_dyna.config import PROJECT_ROOT
from sys_dyna.db.connection import _connect, init_schema  # type: ignore[attr-defined]
from sys_dyna.repository import sessions as sessions_repo
from sys_dyna.repository import simulation_results as sim_repo
from sys_dyna.repository import users as users_repo


SEED_USERS = PROJECT_ROOT / "data" / "seed_users.json"
SEED_SESSIONS = PROJECT_ROOT / "data" / "seed_sessions.json"
SEED_SIM_RESULTS = PROJECT_ROOT / "data" / "seed_simulation_results.json"


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    p = tmp_path / "sys_dyna_test.db"
    init_schema(p)
    return p


@pytest.fixture
def seeded_db_path(db_path: Path) -> Path:
    conn = _connect(db_path)
    try:
        for u in json.loads(SEED_USERS.read_text(encoding="utf-8")):
            users_repo.upsert(
                conn,
                users_repo.UserRow(
                    user_id=u["user_id"],
                    display_name=u["display_name"],
                    department=u.get("department"),
                ),
            )
        for s in json.loads(SEED_SESSIONS.read_text(encoding="utf-8")):
            chat_log = [
                sessions_repo.ChatMessage(role=m["role"], content=m["content"], ts=m.get("ts", ""))
                for m in s["chat_log"]
            ]
            sessions_repo.upsert(
                conn,
                sessions_repo.SessionRecord(
                    session_id=s["session_id"],
                    user_id=s["user_id"],
                    created_at=s["created_at"],
                    updated_at=s["updated_at"],
                    model_name=s.get("model_name"),
                    chat_log=chat_log,
                    final_state=s.get("final_state"),
                ),
            )
        for r in json.loads(SEED_SIM_RESULTS.read_text(encoding="utf-8")):
            sim_repo.upsert(
                conn,
                sim_repo.SimulationResultRecord(
                    result_id=r["result_id"],
                    session_id=r["session_id"],
                    time_series_data=r["time_series_data"],
                    created_at=r["created_at"],
                ),
            )
    finally:
        conn.close()
    return db_path


@pytest.fixture
def connection_factory(seeded_db_path: Path):
    def _factory() -> sqlite3.Connection:
        return _connect(seeded_db_path)

    return _factory
