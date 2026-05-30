from __future__ import annotations

from typing import Callable

import pytest

from sys_dyna.simulation import Objective, SimulationError
from sys_dyna.simulation.analysis import BayesianOptimization, ParamRange
from sys_dyna.simulation.models import ModelRef, ScenarioResult, SimulationRun


pytest.importorskip("optuna")


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
                    variables={"Sales": [{"t": 0.0, "v": v}]},
                )
            )
        return run


_REF = ModelRef(model_id="fake", source="catalog", path="/dev/null")


def test_optimization_finds_maximum() -> None:
    # Concave objective peaking at x == 3 -> maximiser should converge near 3.
    engine = FakeEngine(lambda p: -((p["x"] - 3.0) ** 2))
    bo = BayesianOptimization(engine)
    result = bo.optimize(
        _REF,
        base_params={},
        search_space=[ParamRange("x", low=0.0, high=10.0)],
        objective=Objective("Sales", "final", "maximize"),
        n_trials=60,
        seed=0,
    )
    assert result.n_trials == 60
    assert result.best_params["x"] == pytest.approx(3.0, abs=0.5)
    assert result.best_value == pytest.approx(0.0, abs=0.5)
    assert len(result.history) == 60
    assert "x" in result.history[0]["params"]


def test_optimization_minimizes_when_asked() -> None:
    # Sales == x; minimising should push x toward its lower bound.
    engine = FakeEngine(lambda p: p["x"])
    bo = BayesianOptimization(engine)
    result = bo.optimize(
        _REF,
        base_params={},
        search_space=[ParamRange("x", low=1.0, high=5.0)],
        objective=Objective("Sales", direction="minimize"),
        n_trials=40,
        seed=1,
    )
    assert result.best_params["x"] == pytest.approx(1.0, abs=0.3)


def test_trials_capped_and_warned() -> None:
    engine = FakeEngine(lambda p: p["x"])
    bo = BayesianOptimization(engine, max_trials=5)
    result = bo.optimize(
        _REF,
        {},
        [ParamRange("x", 0.0, 1.0)],
        Objective("Sales"),
        n_trials=100,
    )
    assert result.n_trials == 5
    assert any("capped" in w for w in result.warnings)


def test_no_search_space_raises() -> None:
    bo = BayesianOptimization(FakeEngine(lambda p: 1.0))
    with pytest.raises(SimulationError) as ei:
        bo.optimize(_REF, {}, [], Objective("Sales"))
    assert ei.value.code == "no_search_space"


def test_clamp_keeps_reported_params_within_model_bounds() -> None:
    # Sales == x (maximise). The range allows up to 100, but a clamp restricts
    # the model to <= 10. best_params/history must reflect values actually run
    # (i.e. clamped), not Optuna's raw out-of-range suggestions.
    engine = FakeEngine(lambda p: p["x"])
    bo = BayesianOptimization(engine)
    result = bo.optimize(
        _REF,
        {},
        [ParamRange("x", low=0.0, high=100.0)],
        Objective("Sales", direction="maximize"),
        n_trials=30,
        seed=0,
        clamp=lambda name, v: min(v, 10.0),
    )
    assert result.best_params["x"] <= 10.0 + 1e-6
    assert all(h["params"]["x"] <= 10.0 + 1e-6 for h in result.history)
    # The clamp caps the achievable objective at 10.
    assert result.best_value == pytest.approx(10.0, abs=0.5)


def test_clamp_collapsed_range_stays_exactly_on_bound() -> None:
    # A range wholly above the model max collapses both ends onto the bound
    # (1.0). Optuna must report exactly 1.0 and never a value past it — the
    # previous +1e-9 widening let suggestions cross the declared limit.
    engine = FakeEngine(lambda p: p["x"])
    bo = BayesianOptimization(engine)
    result = bo.optimize(
        _REF,
        {},
        [ParamRange("x", low=2.0, high=3.0)],
        Objective("Sales", direction="maximize"),
        n_trials=10,
        seed=0,
        clamp=lambda name, v: min(v, 1.0),
    )
    assert result.best_params["x"] == pytest.approx(1.0)
    assert all(h["params"]["x"] <= 1.0 for h in result.history)
