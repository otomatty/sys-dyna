from __future__ import annotations

from typing import Callable

import pytest

from sys_dyna.agents import AnalysisError, SimulationAgent
from sys_dyna.simulation.models import (
    ModelRef,
    ModelSpec,
    ParamSpec,
    ScenarioResult,
    SimulationRun,
)


pytest.importorskip("numpy")


class FakeEngine:
    def __init__(self, fn: Callable[[dict], float]) -> None:
        self.fn = fn

    def run_scenarios(self, ref, scenarios, return_columns=None):  # type: ignore[no-untyped-def]
        run = SimulationRun(model_id=ref.model_id)
        for sc in scenarios:
            v = self.fn(sc.params)
            run.scenarios.append(
                ScenarioResult(
                    scenario=sc.name,
                    params=dict(sc.params),
                    variables={"Sales": [{"t": 0.0, "v": 0.0}, {"t": 1.0, "v": v}]},
                )
            )
        return run


_SPEC = ModelSpec(
    model_id="sales_growth",
    name="fake",
    description="d",
    ref=ModelRef(model_id="sales_growth", source="catalog", path="/dev/null"),
    params=(
        ParamSpec(name="ad_spend", label="広告費", default=100.0, min=0.0),
        ParamSpec(name="churn_rate", label="解約率", default=0.05, min=0.0, max=1.0),
    ),
    output_variables=("Sales",),
)


def _lookup(model_id: str) -> ModelSpec | None:
    return _SPEC if model_id == "sales_growth" else None


def _agent(fn: Callable[[dict], float]) -> SimulationAgent:
    return SimulationAgent(FakeEngine(fn), model_lookup=_lookup)


def test_agent_exposes_both_tools() -> None:
    agent = _agent(lambda p: 1.0)
    assert set(agent.tools) == {"monte_carlo", "bayesian_optimization"}
    assert agent.supports("montecarlo") and agent.supports("optimize")
    names = {d.name for d in agent.tool_definitions()}
    assert names == {"monte_carlo", "bayesian_optimization"}


def test_agent_runs_monte_carlo() -> None:
    agent = _agent(lambda p: p["ad_spend"])
    payload = agent.run(
        "montecarlo",
        {
            "model_id": "sales_growth",
            "distributions": [{"name": "ad_spend", "kind": "normal", "mean": 100.0, "std": 5.0}],
            "objective": {"variable": "Sales"},
            "iterations": 200,
            "seed": 3,
        },
    )
    assert payload["kind"] == "montecarlo"
    assert payload["stats"]["mean"] == pytest.approx(100.0, abs=2.0)
    assert "summary" in payload and "モンテカルロ" in payload["summary"]


def test_agent_runs_optimization() -> None:
    pytest.importorskip("optuna")
    agent = _agent(lambda p: -((p["ad_spend"] - 120.0) ** 2))
    payload = agent.run(
        "optimize",
        {
            "model_id": "sales_growth",
            "search_space": [{"name": "ad_spend", "low": 0.0, "high": 300.0}],
            "objective": {"variable": "Sales", "direction": "maximize"},
            "n_trials": 50,
            "seed": 0,
        },
    )
    assert payload["kind"] == "optimize"
    assert payload["best_params"]["ad_spend"] == pytest.approx(120.0, abs=10.0)
    assert "ベイズ最適化" in payload["summary"]


def test_objective_defaults_to_first_output_variable() -> None:
    # No objective in arguments -> falls back to the model's first output var.
    agent = _agent(lambda p: p["ad_spend"])
    payload = agent.run(
        "montecarlo",
        {
            "model_id": "sales_growth",
            "distributions": [{"name": "ad_spend", "kind": "uniform", "low": 50.0, "high": 150.0}],
            "seed": 1,
        },
    )
    assert payload["objective"]["variable"] == "Sales"


def test_unknown_model_raises_analysis_error() -> None:
    agent = _agent(lambda p: 1.0)
    with pytest.raises(AnalysisError) as ei:
        agent.run("montecarlo", {"model_id": "ghost", "distributions": [{"name": "x", "kind": "fixed", "mean": 1.0}]})
    assert ei.value.code == "not_found"


def test_unknown_analysis_kind_raises() -> None:
    agent = _agent(lambda p: 1.0)
    with pytest.raises(AnalysisError) as ei:
        agent.run("genetic", {"model_id": "sales_growth"})
    assert ei.value.code == "unknown_analysis"


def test_missing_distributions_raises() -> None:
    agent = _agent(lambda p: 1.0)
    with pytest.raises(AnalysisError) as ei:
        agent.run("montecarlo", {"model_id": "sales_growth"})
    assert ei.value.code == "invalid_argument"


def test_unexpected_tool_exception_is_normalized_to_analysis_error() -> None:
    # A non-SimulationError from deep in the engine must surface as AnalysisError
    # (not escape and crash the graph turn, which only catches AnalysisError).
    class BoomEngine:
        def run_scenarios(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

    agent = SimulationAgent(BoomEngine(), model_lookup=_lookup)
    with pytest.raises(AnalysisError) as ei:
        agent.run(
            "montecarlo",
            {
                "model_id": "sales_growth",
                "distributions": [{"name": "ad_spend", "kind": "fixed", "mean": 1.0}],
            },
        )
    assert ei.value.code == "analysis_failed"
    assert "boom" in ei.value.message
