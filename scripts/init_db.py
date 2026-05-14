"""Initialise the local SQLite database and load seed data.

Usage:
    python -m scripts.init_db [--reset]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sys_dyna.config import PROJECT_ROOT, get_settings
from sys_dyna.db.connection import _connect, init_schema  # type: ignore[attr-defined]
from sys_dyna.repository import sessions as sessions_repo
from sys_dyna.repository import simulation_results as sim_repo
from sys_dyna.repository import users as users_repo


SEED_USERS = PROJECT_ROOT / "data" / "seed_users.json"
SEED_SESSIONS = PROJECT_ROOT / "data" / "seed_sessions.json"
SEED_SIM_RESULTS = PROJECT_ROOT / "data" / "seed_simulation_results.json"


def _load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialise sys-dyna SQLite DB")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing DB file before recreating.",
    )
    args = parser.parse_args()

    settings = get_settings()
    db_path = settings.db_path
    if args.reset and db_path.exists():
        db_path.unlink()
        print(f"removed existing DB: {db_path}")

    init_schema(db_path)
    print(f"schema applied at: {db_path}")

    conn = _connect(db_path)
    try:
        users = _load_json(SEED_USERS)
        for u in users:
            users_repo.upsert(
                conn,
                users_repo.UserRow(
                    user_id=u["user_id"],
                    display_name=u["display_name"],
                    department=u.get("department"),
                ),
            )
        print(f"seeded users: {len(users)}")

        sessions = _load_json(SEED_SESSIONS)
        for s in sessions:
            chat_log = [
                sessions_repo.ChatMessage(
                    role=m["role"], content=m["content"], ts=m.get("ts", "")
                )
                for m in s.get("chat_log", [])
            ]
            sessions_repo.upsert(
                conn,
                sessions_repo.SessionRecord(
                    session_id=s["session_id"],
                    user_id=s["user_id"],
                    created_at=s["created_at"],
                    updated_at=s.get("updated_at", s["created_at"]),
                    model_name=s.get("model_name"),
                    chat_log=chat_log,
                    final_state=s.get("final_state"),
                ),
            )
        print(f"seeded sessions: {len(sessions)}")

        sims = _load_json(SEED_SIM_RESULTS)
        for r in sims:
            sim_repo.upsert(
                conn,
                sim_repo.SimulationResultRecord(
                    result_id=r["result_id"],
                    session_id=r["session_id"],
                    time_series_data=r["time_series_data"],
                    created_at=r["created_at"],
                ),
            )
        print(f"seeded simulation_results: {len(sims)}")
    finally:
        conn.close()

    print("done.")


if __name__ == "__main__":
    main()
