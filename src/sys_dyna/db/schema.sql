-- Mirrors the Snowflake DDL described in the basic design doc section 7.
-- Columns named "variant" in the doc are stored as JSON text here.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    user_id      TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    department   TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(user_id),
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    model_name  TEXT,
    chat_log    TEXT NOT NULL,
    final_state TEXT
);

CREATE INDEX IF NOT EXISTS ix_sessions_user       ON sessions(user_id);
CREATE INDEX IF NOT EXISTS ix_sessions_created_at ON sessions(created_at);
CREATE INDEX IF NOT EXISTS ix_sessions_model      ON sessions(model_name);

CREATE TABLE IF NOT EXISTS simulation_results (
    result_id        TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL REFERENCES sessions(session_id),
    time_series_data TEXT NOT NULL,
    created_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_results_session ON simulation_results(session_id);

CREATE TABLE IF NOT EXISTS tool_call_logs (
    log_id      TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(session_id),
    tool_name   TEXT NOT NULL,
    tool_input  TEXT,
    tool_output TEXT,
    called_at   TEXT NOT NULL,
    duration_ms INTEGER
);

CREATE INDEX IF NOT EXISTS ix_tool_logs_session ON tool_call_logs(session_id);
CREATE INDEX IF NOT EXISTS ix_tool_logs_tool    ON tool_call_logs(tool_name);
