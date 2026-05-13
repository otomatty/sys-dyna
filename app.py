from __future__ import annotations

import uuid

import streamlit as st

from sys_dyna.auth import get_current_user
from sys_dyna.config import get_settings
from sys_dyna.db.connection import _connect, init_schema  # type: ignore[attr-defined]
from sys_dyna.llm.client import LLMMessage
from sys_dyna.llm.mock_client import MockGeminiClient
from sys_dyna.orchestrator import AgenticSearchOrchestrator
from sys_dyna.repository import sessions as sessions_repo
from sys_dyna.repository import users as users_repo
from sys_dyna.repository.sessions import ChatMessage
from sys_dyna.repository.users import UserRow
from sys_dyna.tools import build_default_tools
from sys_dyna.ui.chat import render_history
from sys_dyna.ui.sidebar import render_sidebar


st.set_page_config(page_title="SD x LLM 社内分析ツール", page_icon=None, layout="wide")


def _bootstrap() -> None:
    """Run on every script invocation; idempotent."""
    settings = get_settings()
    init_schema(settings.db_path)

    user = get_current_user()
    conn = _connect(settings.db_path)
    try:
        users_repo.upsert(
            conn,
            UserRow(
                user_id=user.user_id,
                display_name=user.display_name,
                department=user.department,
            ),
        )

        if "session_id" not in st.session_state:
            session_id = str(uuid.uuid4())
            sessions_repo.create_empty(
                conn,
                session_id=session_id,
                user_id=user.user_id,
                model_name=settings.model_name,
            )
            st.session_state.session_id = session_id
            st.session_state.chat_history = []  # list[ChatMessage]
            st.session_state.invocations_by_turn = []  # list[list[ToolInvocation]]
    finally:
        conn.close()


def _to_llm_history(history: list[ChatMessage]) -> list[LLMMessage]:
    out: list[LLMMessage] = []
    for m in history:
        if m.role in ("user", "assistant"):
            out.append(LLMMessage(role=m.role, content=m.content))  # type: ignore[arg-type]
    return out


def main() -> None:
    _bootstrap()

    settings = get_settings()
    user = get_current_user()
    session_id: str = st.session_state.session_id
    history: list[ChatMessage] = st.session_state.chat_history
    invocations_by_turn = st.session_state.invocations_by_turn

    render_sidebar(
        user=user,
        session_id=session_id,
        model_name=settings.model_name,
        invocations_by_turn=invocations_by_turn,
    )

    st.title("システムダイナミクス × LLM 社内分析ツール")
    st.caption(
        "過去セッションを Agentic Search で参照しながら回答します。質問を入力してください。"
    )

    render_history(history)

    user_input = st.chat_input("質問を入力 (例: 広告費を1.5倍にしたら売上はどうなる? 過去に似た分析あった?)")
    if not user_input:
        return

    with st.chat_message("user"):
        st.markdown(user_input)

    history.append(ChatMessage(role="user", content=user_input))

    tools = build_default_tools(settings.db_path)
    orchestrator = AgenticSearchOrchestrator(
        llm=MockGeminiClient(model_name=settings.model_name),
        tools=tools,
    )

    with st.spinner("Agentic Search 実行中..."):
        result = orchestrator.run_turn(
            session_id=session_id,
            user_text=user_input,
            history=_to_llm_history(history[:-1]),
        )

    history.append(ChatMessage(role="assistant", content=result.text))
    invocations_by_turn.append(result.invocations)

    conn = _connect(settings.db_path)
    try:
        sessions_repo.upsert(
            conn,
            sessions_repo.SessionRecord(
                session_id=session_id,
                user_id=user.user_id,
                created_at=_get_or_keep_created_at(conn, session_id),
                updated_at=_now_iso(),
                model_name=settings.model_name,
                chat_log=list(history),
                final_state=None,
            ),
        )
    finally:
        conn.close()

    with st.chat_message("assistant"):
        st.markdown(result.text)
        if result.hit_loop_limit:
            st.warning("ツール呼び出しの上限に達したため、現在の情報で応答しました。")
        if result.hit_turn_timeout:
            st.warning("ターンのタイムアウトに達しました。")
        if result.invocations:
            st.caption(f"このターンで {len(result.invocations)} 件のツール呼び出しを実行しました。")


def _get_or_keep_created_at(conn, session_id: str) -> str:
    rec = sessions_repo.get(conn, session_id)
    if rec is not None:
        return rec.created_at
    return _now_iso()


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
