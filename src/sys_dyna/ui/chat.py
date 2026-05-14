from __future__ import annotations

import streamlit as st

from ..repository.sessions import ChatMessage


def render_history(messages: list[ChatMessage]) -> None:
    for msg in messages:
        if msg.role not in ("user", "assistant"):
            continue
        with st.chat_message(msg.role):
            st.markdown(msg.content)
