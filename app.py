from __future__ import annotations

import uuid
from dataclasses import dataclass

import streamlit as st

from sys_dyna.auth import get_current_user
from sys_dyna.config import get_settings
from sys_dyna.graph import build_planner, build_runner
from sys_dyna.simulation import get_model
from sys_dyna.ui.charts import render_analysis_result, render_simulation
from sys_dyna.ui.param_confirm import render_analysis_confirm, render_param_confirm


st.set_page_config(page_title="SD x LLM 社内分析ツール", layout="wide")


@dataclass
class ChatTurn:
    role: str
    content: str
    simulation: dict | None = None
    # Monte Carlo / Bayesian-optimization result payload (SimulationAgent output).
    analysis: dict | None = None


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
        st.session_state.confirm_seq = 0  # unique key namespace per confirmation


def _render_sidebar(user, settings) -> None:
    with st.sidebar:
        st.subheader("セッション情報")
        st.text(f"ユーザー: {user.display_name}")
        if user.department:
            st.caption(f"部署: {user.department}")
        st.code(st.session_state.session_id, language="text")
        mode = "Gemini" if settings.gemini_api_key else "オフライン(ヒューリスティック)"
        st.caption(f"分析エンジン: {mode} / {settings.gemini_model}")


def _append(
    role: str,
    content: str,
    simulation: dict | None = None,
    analysis: dict | None = None,
) -> None:
    st.session_state.history.append(ChatTurn(role, content, simulation, analysis))


def _last_simulation_params(history: list[ChatTurn]) -> dict | None:
    """First scenario's params from the most recent simulation turn, if any."""
    for turn in reversed(history):
        if turn.simulation and turn.simulation.get("scenarios"):
            return dict(turn.simulation["scenarios"][0].get("params") or {})
    return None


def _handle_completed(outcome) -> None:
    _append(
        "assistant",
        outcome.analysis or "(分析結果がありません)",
        outcome.simulation,
        getattr(outcome, "simulation_analysis", None),
    )


def main() -> None:
    _bootstrap()
    settings = get_settings()
    user = get_current_user()
    runner = _get_runner()
    session_id = st.session_state.session_id

    _render_sidebar(user, settings)

    st.title("システムダイナミクス × LLM 社内分析ツール")
    st.caption("シミュレーションを実行し、結果を Gemini が分析・説明します。")

    for idx, turn in enumerate(st.session_state.history):
        with st.chat_message(turn.role):
            st.markdown(turn.content)
            if turn.simulation:
                render_simulation(turn.simulation, key_prefix=f"sim_{idx}")
            if turn.analysis:
                render_analysis_result(turn.analysis)

    # --- HITL: a pending confirmation takes priority over new input ---
    pending = st.session_state.pending
    if pending is not None:
        model = get_model(pending.selected_model_id) if pending.selected_model_id else None
        key_prefix = f"confirm_{st.session_state.confirm_seq}"
        # The advanced-analysis interrupt (confirm_analysis) carries a different
        # payload/decision shape than the scenario confirm, so route on type —
        # otherwise the analysis form renders no fields and its result is lost.
        if (pending.confirm or {}).get("type") == "confirm_analysis":
            decision = render_analysis_confirm(pending.confirm, model, key_prefix=key_prefix)
            cancelled = decision is not None and not decision.get("spec")
            cancel_message = "分析をキャンセルしました。"
            running_message = "分析を実行中..."
        else:
            decision = render_param_confirm(pending.confirm, model, key_prefix=key_prefix)
            cancelled = decision is not None and not decision.get("scenarios")
            cancel_message = "シミュレーションをキャンセルしました。"
            running_message = "シミュレーション実行中..."
        if decision is not None:
            # Always resume so the LangGraph thread leaves the interrupted state,
            # even on cancel — otherwise the next input on this session fails.
            try:
                with st.spinner("処理中..." if cancelled else running_message):
                    outcome = runner.resume(session_id, decision)
            except Exception as e:
                # Keep `pending` so the confirmation form survives a failed resume.
                _append("assistant", f"エラーが発生しました: {e}")
                st.rerun()
                return
            st.session_state.pending = None
            if cancelled:
                _append("assistant", cancel_message)
            else:
                _handle_completed(outcome)
            st.rerun()
        return

    user_input = st.chat_input("質問を入力 (例: 広告費を1.5倍にしたら売上はどうなる?)")
    if not user_input:
        return

    # Prior turns (excluding the input we're about to add) give the LLM
    # multi-turn context for follow-up questions.
    history = [{"role": t.role, "content": t.content} for t in st.session_state.history]
    # The most recent simulation's first scenario seeds follow-up edits so an
    # unchanged parameter keeps its previous value rather than reverting.
    base_params = _last_simulation_params(st.session_state.history)
    _append("user", user_input)
    try:
        with st.spinner("解析中..."):
            outcome = runner.start(
                session_id,
                user_input,
                user_id=user.user_id,
                history=history,
                base_params=base_params,
            )
    except Exception as e:
        _append("assistant", f"エラーが発生しました: {e}")
        st.rerun()
        return

    if outcome.status == "awaiting_confirmation":
        st.session_state.confirm_seq += 1  # fresh key namespace for this form
        st.session_state.pending = outcome
    else:
        _handle_completed(outcome)
    st.rerun()


if __name__ == "__main__":
    main()
