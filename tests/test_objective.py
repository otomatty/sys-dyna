from __future__ import annotations

import pytest

from sys_dyna.simulation import Objective, SimulationError
from sys_dyna.simulation.models import ScenarioResult


def _result(values: list[float], variable: str = "Sales") -> ScenarioResult:
    series = [{"t": float(i), "v": float(v)} for i, v in enumerate(values)]
    return ScenarioResult(scenario="s", params={}, variables={variable: series})


def test_aggregates() -> None:
    r = _result([10.0, 20.0, 30.0])
    assert Objective("Sales", "final").scalar(r) == 30.0
    assert Objective("Sales", "initial").scalar(r) == 10.0
    assert Objective("Sales", "mean").scalar(r) == 20.0
    assert Objective("Sales", "min").scalar(r) == 10.0
    assert Objective("Sales", "max").scalar(r) == 30.0
    assert Objective("Sales", "sum").scalar(r) == 60.0


def test_sign_reflects_direction() -> None:
    assert Objective("Sales", direction="maximize").sign == 1.0
    assert Objective("Sales", direction="minimize").sign == -1.0


def test_unknown_variable_raises() -> None:
    with pytest.raises(SimulationError) as ei:
        Objective("Profit").scalar(_result([1.0]))
    assert ei.value.code == "unknown_variable"


def test_invalid_aggregate_and_direction_rejected() -> None:
    with pytest.raises(SimulationError):
        Objective("Sales", aggregate="median")  # type: ignore[arg-type]
    with pytest.raises(SimulationError):
        Objective("Sales", direction="both")  # type: ignore[arg-type]


def test_from_dict_uses_default_variable() -> None:
    obj = Objective.from_dict({"aggregate": "mean"}, default_variable="Sales")
    assert obj.variable == "Sales"
    assert obj.aggregate == "mean"
    assert obj.direction == "maximize"
    # Round-trips through to_payload.
    assert obj.to_payload() == {"variable": "Sales", "aggregate": "mean", "direction": "maximize"}


def test_from_dict_without_variable_or_default_raises() -> None:
    with pytest.raises(SimulationError):
        Objective.from_dict({}, default_variable=None)
