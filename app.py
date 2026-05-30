from __future__ import annotations

import uuid
from dataclasses import dataclass

import streamlit as st

from sys_dyna.auth import get_current_user
from sys_dyna.config import get_settings
from sys_dyna.graph import build_planner, build_runner
from sys_dyna.simulation import get_model
from sys_dyna.ui.charts import render_simulation
from sys_dyna.ui.param_confirm import render_param_confirm


st.set_page_config(page_title="SD x LLM 社内分析ツール", layout="wide")


@dataclass
class ChatTurn:
    role: str
    content: str
    simulation: dict | None = None


@st.cache_resource(show_spinner=False)
def _get_runner():
    """Build the LangGraph runner once per process.

    Uses an in-memory checkpointer so the HITL interrupt can resume across
    Streamlit reruns within a process. In production this is swapped for the
    Postgres (Supabase) checkpointer — see docs/design_v2.md §5.2.
    """
    from langgraph.checkpoint.memory import MemorySaver

    settings = get_settings()
    planner = build_planner(
        gemini_api_key=settings.gemini_api_key,
        gemini_model=settings.gemini_model,
        temperature=settings.gemini_temperature,
        max_scenarios=settings.max_scenarios,
    )
    return build_runner(planner, checkpointer=MemorySaver())


def _bootstrap() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.history = []  # list[ChatTurn]
        st.session_state.pending = None  # TurnOutcome awaiting confirmation


def _render_sidebar(user, settings) -> None:
    with st.sidebar:
        st.subheader("セッション情報")
        st.text(f"ユーザー: {user.display_name}")
        if user.department:
            st.caption(f"部署: {user.department}")
        st.code(st.session_state.session_id, language="text")
        mode = "Gemini" if settings.gemini_api_key else "オフライン(ヒューリスティック)"
        st.caption(f"分析エンジン: {mode} / {settings.gemini_model}")


def _append(role: str, content: str, simulation: dict | None = None) -> None:
    st.session_state.history.append(ChatTurn(role, content, simulation))


def _handle_completed(outcome) -> None:
    _append("assistant", outcome.analysis or "(分析結果がありません)", outcome.simulation)


def main() -> None:
    _bootstrap()
    settings = get_settings()
    user = get_current_user()
    runner = _get_runner()
    session_id = st.session_state.session_id

    _render_sidebar(user, settings)

    st.title("システムダイナミクス × LLM 社内分析ツール")
    st.caption("シミュレーションを実行し、結果を Gemini が分析・説明します。")

    for turn in st.session_state.history:
        with st.chat_message(turn.role):
            st.markdown(turn.content)
            if turn.simulation:
                render_simulation(turn.simulation)

    # --- HITL: a pending confirmation takes priority over new input ---
    pending = st.session_state.pending
    if pending is not None:
        model = get_model(pending.selected_model_id) if pending.selected_model_id else None
        decision = render_param_confirm(pending.confirm, model)
        if decision is not None:
            st.session_state.pending = None
            if not decision.get("scenarios"):
                _append("assistant", "シミュレーションをキャンセルしました。")
                st.rerun()
            with st.spinner("シミュレーション実行中..."):
                outcome = runner.resume(session_id, decision)
            _handle_completed(outcome)
            st.rerun()
        return

    user_input = st.chat_input("質問を入力 (例: 広告費を1.5倍にしたら売上はどうなる?)")
    if not user_input:
        return

    _append("user", user_input)
    with st.spinner("解析中..."):
        outcome = runner.start(session_id, user_input, user_id=user.user_id)

    if outcome.status == "awaiting_confirmation":
        st.session_state.pending = outcome
    else:
        _handle_completed(outcome)
    st.rerun()


if __name__ == "__main__":
    main()
