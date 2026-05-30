from __future__ import annotations

from typing import Any

import streamlit as st

from ..simulation import ModelSpec


def render_param_confirm(
    confirm: dict[str, Any], model: ModelSpec | None, key_prefix: str = "confirm"
) -> dict[str, Any] | None:
    """Render the HITL parameter-confirmation form.

    Returns a ``decision`` dict (``{"scenarios": [...]}`` for the graph's
    ``Command(resume=...)``) once the user submits, else ``None`` while waiting.

    ``key_prefix`` must be unique per confirmation instance: widget keys persist
    in session_state, so a fixed prefix would make a later turn's form show the
    previous turn's edited values (or raise) instead of the new proposal.
    """
    proposed = confirm.get("scenarios", [])
    labels = {p.name: p.label for p in (model.params if model else [])}

    st.info("シミュレーションを実行する前にパラメータをご確認ください。必要なら修正できます。")
    with st.form(f"{key_prefix}_form"):
        edited: list[dict[str, Any]] = []
        for i, scenario in enumerate(proposed):
            st.markdown(f"**シナリオ {i + 1}: {scenario.get('name', '')}**")
            params = scenario.get("params", {})
            new_params: dict[str, float] = {}
            cols = st.columns(min(3, max(1, len(params))))
            for j, (key, value) in enumerate(params.items()):
                label = labels.get(key, key)
                with cols[j % len(cols)]:
                    new_params[key] = st.number_input(
                        label,
                        value=float(value),
                        key=f"{key_prefix}_p_{i}_{key}",
                        format="%.4f",
                    )
            edited.append({"name": scenario.get("name", f"scenario_{i + 1}"), "params": new_params})

        col_run, col_cancel = st.columns(2)
        run = col_run.form_submit_button("この内容で実行", type="primary")
        cancel = col_cancel.form_submit_button("キャンセル")

    if run:
        return {"scenarios": edited}
    if cancel:
        return {"scenarios": []}  # caller treats empty as cancellation
    return None
