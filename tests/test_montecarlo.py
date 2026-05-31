from __future__ import annotations

from typing import Callable

import pytest

from sys_dyna.simulation import Objective, SimulationError
from sys_dyna.simulation.analysis import MonteCarloAnalysis, ParamDistribution
from sys_dyna.simulation.models import ModelRef, Scenario, ScenarioResult, SimulationRun


pytest.importorskip("numpy")


class FakeEngine:
    """Computes a deterministic ``Sales`` series from a parameter function.

    Replaces PySD so the analysis layer can be tested without the SD stack: the
    series is just two points whose final value is ``fn(params)``.
    """

    def __init__(self, fn: Callable[[dict], float]) -> None:
        self.fn = fn
        self.calls = 0

    def run_scenarios(self, ref, scenarios, return_columns=None):  # type: ignore[no-untyped-def]
        run = SimulationRun(model_id=ref.model_id)
        for sc in scenarios:
            self.calls += 1
            v = self.fn(sc.params)
            run.scenarios.append(
                ScenarioResult(
                    scenario=sc.name,
                    params=dict(sc.params),
                    variables={"Sales": [{"t": 0.0, "v": 0.0}, {"t": 1.0, "v": v}]},
                )
            )
        return run


_REF = ModelRef(model_id="fake", source="catalog", path="/dev/null")


def test_monte_carlo_aggregates_objective_distribution() -> None:
    # Sales == ad_spend, so the objective distribution mirrors the input one.
    engine = FakeEngine(lambda p: p["ad_spend"])
    mc = MonteCarloAnalysis(engine)
    obj = Objective("Sales", "final", "maximize")
    result = mc.run(
        _REF,
        base_params={"ad_spend": 100.0},
        distributions=[ParamDistribution("ad_spend", "normal", mean=100.0, std=10.0)],
        objective=obj,
        iterations=500,
        seed=7,
    )

    assert result.iterations == 500
    assert engine.calls == 500
    # Mean/std of Sales track the normal(100, 10) input within sampling error.
    assert result.stats.mean == pytest.approx(100.0, abs=2.0)
    assert result.stats.std == pytest.approx(10.0, abs=2.0)
    p = result.stats.percentiles
    assert p["p5"] < p["p50"] < p["p95"]
    # The only varied input is perfectly correlated with the objective.
    assert result.sensitivities["ad_spend"] == pytest.approx(1.0, abs=1e-6)
    assert sum(result.histogram["counts"]) == 500


def test_monte_carlo_is_reproducible_with_seed() -> None:
    engine = FakeEngine(lambda p: p["ad_spend"] * 2)
    mc = MonteCarloAnalysis(engine)
    obj = Objective("Sales")
    kwargs = dict(
        base_params={"ad_spend": 50.0},
        distributions=[ParamDistribution("ad_spend", "uniform", low=0.0, high=100.0)],
        objective=obj,
        iterations=100,
        seed=42,
    )
    a = mc.run(_REF, **kwargs)  # type: ignore[arg-type]
    b = mc.run(_REF, **kwargs)  # type: ignore[arg-type]
    assert a.stats.mean == b.stats.mean
    assert a.stats.percentiles == b.stats.percentiles


def test_clamp_keeps_samples_in_range() -> None:
    seen: list[float] = []

    def fn(p: dict) -> float:
        seen.append(p["churn_rate"])
        return p["churn_rate"]

    engine = FakeEngine(fn)
    mc = MonteCarloAnalysis(engine)
    mc.run(
        _REF,
        base_params={"churn_rate": 0.05},
        # Wide spread; clamp must pin every draw into [0, 1].
        distributions=[ParamDistribution("churn_rate", "normal", mean=0.5, std=2.0)],
        objective=Objective("Sales"),
        iterations=200,
        seed=1,
        clamp=lambda name, v: min(max(v, 0.0), 1.0),
    )
    assert seen and all(0.0 <= v <= 1.0 for v in seen)


def test_iterations_capped_and_warned() -> None:
    engine = FakeEngine(lambda p: 1.0)
    mc = MonteCarloAnalysis(engine, max_iterations=10)
    result = mc.run(
        _REF,
        base_params={},
        distributions=[ParamDistribution("ad_spend", "uniform", low=0.0, high=1.0)],
        objective=Objective("Sales"),
        iterations=1000,
    )
    assert result.iterations == 10
    assert any("capped" in w for w in result.warnings)


def test_no_distributions_raises() -> None:
    mc = MonteCarloAnalysis(FakeEngine(lambda p: 1.0))
    with pytest.raises(SimulationError) as ei:
        mc.run(_REF, {}, [], Objective("Sales"))
    assert ei.value.code == "no_distributions"


def test_triangular_requires_high_greater_than_low() -> None:
    # numpy's triangular needs right > left; reject hi == lo at construction.
    with pytest.raises(SimulationError) as ei:
        ParamDistribution("x", "triangular", low=5.0, high=5.0)
    assert ei.value.code == "invalid_distribution"
    # uniform with equal bounds is still allowed (degenerate but valid).
    ParamDistribution("x", "uniform", low=5.0, high=5.0)
