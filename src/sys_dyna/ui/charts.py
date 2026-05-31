from __future__ import annotations

from typing import Any


def scenario_variables(simulation: dict[str, Any]) -> list[str]:
    """Union of variable names present across all scenarios, order-preserving."""
    seen: list[str] = []
    for sc in simulation.get("scenarios", []):
        for var in sc.get("variables", {}):
            if var not in seen:
                seen.append(var)
    return seen


def to_long_frame(simulation: dict[str, Any], variable: str) -> Any:
    """Tidy (long) DataFrame for one variable: columns t, value, scenario.

    Pure/data-only so it can be unit-tested without Streamlit. Pandas is a
    transitive dependency of PySD so it is always available at runtime.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for sc in simulation.get("scenarios", []):
        name = sc.get("scenario", "?")
        for point in sc.get("variables", {}).get(variable, []):
            rows.append({"t": point["t"], "value": point["v"], "scenario": name})
    return pd.DataFrame(rows, columns=["t", "value", "scenario"])


def render_simulation(simulation: dict[str, Any] | None, key_prefix: str = "sim") -> None:
    """Render scenario time-series as line charts (one per variable).

    ``key_prefix`` must be unique per rendered result: the chat history can show
    several simulations in one run, so without distinct widget keys Streamlit
    raises a duplicate-element-ID error.
    """
    import altair as alt
    import streamlit as st

    if not simulation or not simulation.get("scenarios"):
        return

    variables = scenario_variables(simulation)
    if not variables:
        return

    default = "Sales" if "Sales" in variables else variables[0]
    chosen = st.multiselect(
        "表示する変数", variables, default=[default], key=f"{key_prefix}_vars"
    )
    for var in chosen:
        frame = to_long_frame(simulation, var)
        if frame.empty:
            continue
        st.caption(var)
        chart = (
            alt.Chart(frame)
            .mark_line(point=True)
            .encode(
                x=alt.X("t:Q", title="時間"),
                y=alt.Y("value:Q", title=var),
                color=alt.Color("scenario:N", title="シナリオ"),
            )
            .properties(height=280)
        )
        st.altair_chart(chart, width="stretch", key=f"{key_prefix}_chart_{var}")

    for warning in simulation.get("warnings", []):
        st.warning(warning)


def render_analysis_result(analysis: dict[str, Any] | None) -> None:
    """Render a Monte Carlo / Bayesian-optimization result payload.

    Uses non-widget elements only (metric/table/json), so several results in the
    chat history don't need unique widget keys to avoid duplicate-ID errors.
    """
    import streamlit as st

    if not analysis or analysis.get("error"):
        return

    kind = analysis.get("kind")
    if kind == "montecarlo":
        stats = analysis.get("stats", {})
        pct = stats.get("percentiles", {})
        cols = st.columns(4)
        cols[0].metric("平均", f"{stats.get('mean', 0.0):.2f}")
        cols[1].metric("標準偏差", f"{stats.get('std', 0.0):.2f}")
        cols[2].metric("P5", f"{pct.get('p5', 0.0):.2f}")
        cols[3].metric("P95", f"{pct.get('p95', 0.0):.2f}")
        sens = analysis.get("sensitivities") or {}
        if sens:
            st.caption("入力感度（目的値との相関）")
            st.table(
                {
                    "パラメータ": list(sens.keys()),
                    "相関": [round(float(v), 3) for v in sens.values()],
                }
            )
    elif kind == "optimize":
        st.metric("最適な目的値", f"{analysis.get('best_value', 0.0):.2f}")
        best = analysis.get("best_params") or {}
        if best:
            st.caption("最適パラメータ")
            st.table(
                {
                    "パラメータ": list(best.keys()),
                    "値": [round(float(v), 4) for v in best.values()],
                }
            )

    for warning in analysis.get("warnings", []):
        st.warning(warning)
    with st.expander("詳細データ"):
        st.json(analysis)
