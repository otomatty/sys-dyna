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
    specs = {p.name: p for p in (model.params if model else [])}

    st.info("シミュレーションを実行する前にパラメータをご確認ください。必要なら修正できます。")
    with st.form(f"{key_prefix}_form"):
        edited: list[dict[str, Any]] = []
        for i, scenario in enumerate(proposed):
            st.markdown(f"**シナリオ {i + 1}: {scenario.get('name', '')}**")
            params = scenario.get("params", {})
            new_params: dict[str, float] = {}
            cols = st.columns(min(3, max(1, len(params))))
            for j, (key, value) in enumerate(params.items()):
                spec = specs.get(key)
                label = spec.label if spec else key
                # Bound the widget to the ParamSpec range; clamp the proposed
                # value first so it never falls outside [min_value, max_value].
                value = spec.clamp(float(value)) if spec else float(value)
                with cols[j % len(cols)]:
                    new_params[key] = st.number_input(
                        label,
                        value=value,
                        min_value=(spec.min if spec else None),
                        max_value=(spec.max if spec else None),
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


_AGGREGATES = ("final", "initial", "mean", "min", "max", "sum")
_DIRECTIONS = ("maximize", "minimize")


def render_analysis_confirm(
    confirm: dict[str, Any], model: ModelSpec | None, key_prefix: str = "analysis"
) -> dict[str, Any] | None:
    """Render the HITL confirmation form for an advanced analysis.

    The ``confirm_analysis`` interrupt payload (``{"analysis_kind", "spec", ...}``)
    has a different shape from ``confirm_params`` — it carries an analysis *spec*
    (objective + distributions / search space), not scenarios. Returns a decision
    dict for ``Command(resume=...)``: ``{"spec": {...}}`` to run (the graph's
    ``confirm_analysis`` node reads the ``spec`` key) or ``{"spec": {}}`` to
    cancel (an empty spec routes the graph away from running). ``None`` while
    waiting for input.
    """
    kind = confirm.get("analysis_kind", "montecarlo")
    spec = dict(confirm.get("spec") or {})
    specs = {p.name: p for p in (model.params if model else [])}
    objective = dict(spec.get("objective") or {})

    title = "モンテカルロ分析" if kind == "montecarlo" else "ベイズ最適化"
    st.info(f"{title}の設定をご確認ください。必要なら修正できます。")

    def _label(name: str) -> str:
        return specs[name].label if name in specs else name

    with st.form(f"{key_prefix}_form"):
        new_spec = dict(spec)

        st.markdown(f"**目的関数**: `{objective.get('variable', '?')}`")
        ocols = st.columns(2)
        with ocols[0]:
            agg = objective.get("aggregate", "final")
            new_agg = st.selectbox(
                "集約方法",
                _AGGREGATES,
                index=_AGGREGATES.index(agg) if agg in _AGGREGATES else 0,
                key=f"{key_prefix}_agg",
            )
        with ocols[1]:
            direction = objective.get("direction", "maximize")
            new_dir = st.selectbox(
                "方向",
                _DIRECTIONS,
                index=_DIRECTIONS.index(direction) if direction in _DIRECTIONS else 0,
                key=f"{key_prefix}_dir",
            )
        new_spec["objective"] = {
            "variable": objective.get("variable"),
            "aggregate": new_agg,
            "direction": new_dir,
        }

        if kind == "montecarlo":
            st.markdown("**サンプリング分布**")
            new_dists: list[dict[str, Any]] = []
            for i, dist in enumerate(spec.get("distributions") or []):
                name = dist.get("name", "")
                dkind = dist.get("kind", "normal")
                st.caption(f"{_label(name)} ({name}) — {dkind}")
                nd: dict[str, Any] = {"name": name, "kind": dkind}
                cols = st.columns(3)
                if dkind in ("normal", "lognormal"):
                    with cols[0]:
                        nd["mean"] = st.number_input(
                            "mean", value=float(dist.get("mean") or 0.0),
                            key=f"{key_prefix}_d{i}_mean", format="%.4f",
                        )
                    with cols[1]:
                        nd["std"] = st.number_input(
                            "std", value=float(dist.get("std") or 0.0), min_value=0.0,
                            key=f"{key_prefix}_d{i}_std", format="%.4f",
                        )
                elif dkind in ("uniform", "triangular"):
                    with cols[0]:
                        nd["low"] = st.number_input(
                            "low", value=float(dist.get("low") or 0.0),
                            key=f"{key_prefix}_d{i}_low", format="%.4f",
                        )
                    with cols[1]:
                        nd["high"] = st.number_input(
                            "high", value=float(dist.get("high") or 0.0),
                            key=f"{key_prefix}_d{i}_high", format="%.4f",
                        )
                    if dkind == "triangular":
                        midpoint = (nd["low"] + nd["high"]) / 2.0
                        with cols[2]:
                            nd["mode"] = st.number_input(
                                "mode",
                                value=float(dist.get("mode") if dist.get("mode") is not None else midpoint),
                                key=f"{key_prefix}_d{i}_mode", format="%.4f",
                            )
                else:  # fixed
                    with cols[0]:
                        nd["mean"] = st.number_input(
                            "value",
                            value=float(dist.get("mean") if dist.get("mean") is not None else (dist.get("low") or 0.0)),
                            key=f"{key_prefix}_d{i}_val", format="%.4f",
                        )
                new_dists.append(nd)
            new_spec["distributions"] = new_dists
            new_spec["iterations"] = int(
                st.number_input(
                    "試行回数 (iterations)", value=int(spec.get("iterations") or 200),
                    min_value=1, step=1, key=f"{key_prefix}_iter",
                )
            )
        else:  # optimize
            st.markdown("**探索範囲**")
            new_ranges: list[dict[str, Any]] = []
            for i, rng in enumerate(spec.get("search_space") or []):
                name = rng.get("name", "")
                st.caption(f"{_label(name)} ({name})")
                cols = st.columns(3)
                with cols[0]:
                    low = st.number_input(
                        "low", value=float(rng.get("low") or 0.0),
                        key=f"{key_prefix}_r{i}_low", format="%.4f",
                    )
                with cols[1]:
                    high = st.number_input(
                        "high", value=float(rng.get("high") or 1.0),
                        key=f"{key_prefix}_r{i}_high", format="%.4f",
                    )
                with cols[2]:
                    log = st.checkbox("log", value=bool(rng.get("log", False)), key=f"{key_prefix}_r{i}_log")
                new_ranges.append({"name": name, "low": low, "high": high, "log": log})
            new_spec["search_space"] = new_ranges
            new_spec["n_trials"] = int(
                st.number_input(
                    "試行回数 (n_trials)", value=int(spec.get("n_trials") or 30),
                    min_value=1, step=1, key=f"{key_prefix}_trials",
                )
            )

        seed_val = spec.get("seed")
        use_seed = st.checkbox(
            "乱数シードを固定する（再現性）", value=seed_val is not None, key=f"{key_prefix}_use_seed"
        )
        seed_in = st.number_input(
            "seed", value=int(seed_val) if seed_val is not None else 0, step=1, key=f"{key_prefix}_seed"
        )
        if use_seed:
            new_spec["seed"] = int(seed_in)
        else:
            new_spec.pop("seed", None)

        col_run, col_cancel = st.columns(2)
        run = col_run.form_submit_button("この設定で実行", type="primary")
        cancel = col_cancel.form_submit_button("キャンセル")

    if run:
        return {"spec": new_spec}
    if cancel:
        return {"spec": {}}  # empty spec -> graph skips the analysis (cancellation)
    return None
