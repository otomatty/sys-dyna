from __future__ import annotations

import json

import streamlit as st

from ..auth import CurrentUser
from ..orchestrator import ToolInvocation


def render_sidebar(
    user: CurrentUser,
    session_id: str,
    model_name: str,
    invocations_by_turn: list[list[ToolInvocation]],
) -> None:
    with st.sidebar:
        st.subheader("セッション情報")
        st.text(f"ユーザー: {user.display_name}")
        if user.department:
            st.caption(f"部署: {user.department}")
        st.code(session_id, language="text")
        st.caption(f"モデル: {model_name}")

        st.divider()
        st.subheader("ツール呼び出し履歴")
        if not invocations_by_turn:
            st.caption("まだツールは呼び出されていません。")
            return

        for turn_idx, invocations in enumerate(invocations_by_turn, start=1):
            if not invocations:
                continue
            st.markdown(f"**ターン {turn_idx}** — {len(invocations)} 件")
            for i, inv in enumerate(invocations, start=1):
                status = "OK" if inv.error is None else "ERR"
                title = f"{i}. {inv.name} [{status}] ({inv.duration_ms} ms)"
                with st.expander(title, expanded=False):
                    st.markdown("**input**")
                    st.code(
                        json.dumps(inv.arguments, ensure_ascii=False, indent=2),
                        language="json",
                    )
                    st.markdown("**output**")
                    st.code(
                        json.dumps(inv.output, ensure_ascii=False, indent=2, default=str),
                        language="json",
                    )
                    if inv.error:
                        st.error(inv.error)
