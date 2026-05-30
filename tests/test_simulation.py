from __future__ import annotations

import math

import pytest

from sys_dyna.simulation import (
    PySDEngine,
    Scenario,
    SimulationError,
    catalog_summary,
    get_model,
    list_models,
)
from sys_dyna.simulation.models import ModelRef


pytest.importorskip("pysd")


def test_catalog_exposes_starter_model() -> None:
    models = list_models()
    assert any(m.model_id == "sales_growth" for m in models)

    spec = get_model("sales_growth")
    assert spec is not None
    assert spec.default_params() == {
        "ad_spend": 100.0,
        "conversion": 0.5,
        "churn_rate": 0.05,
    }
    # Summary is LLM-facing and must stay compact.
    summary = catalog_summary()
    assert summary and set(summary[0]) == {"model_id", "name", "description"}


def test_get_model_unknown_returns_none() -> None:
    assert get_model("does_not_exist") is None


def test_run_base_scenario_is_equilibrium() -> None:
    spec = get_model("sales_growth")
    assert spec is not None
    engine = PySDEngine()
    run = engine.run_scenarios(spec.ref, [Scenario(name="base", params={})])

    assert len(run.scenarios) == 1
    result = run.scenarios[0]
    sales = result.series("Sales")
    assert sales, "Sales series should not be empty"
    # Base parameters are chosen so acquisition == churn -> Sales stays at 1000.
    assert math.isclose(sales[0]["v"], 1000.0, rel_tol=1e-6)
    assert math.isclose(sales[-1]["v"], 1000.0, rel_tol=1e-3)
    # Every point is JSON-safe and has the normalised shape.
    assert all(set(p) == {"t", "v"} for p in sales)


def test_param_override_grows_sales() -> None:
    spec = get_model("sales_growth")
    assert spec is not None
    engine = PySDEngine()
    run = engine.run_scenarios(
        spec.ref,
        [Scenario(name="ad_x1.5", params={"ad_spend": 150.0})],
    )
    final = run.scenarios[0].series("Sales")[-1]["v"]
    assert final > 1000.0


def test_scenario_comparison_independent_runs() -> None:
    """Multiple scenarios on one cached model must not leak stock state."""
    spec = get_model("sales_growth")
    assert spec is not None
    engine = PySDEngine()
    run = engine.run_scenarios(
        spec.ref,
        [
            Scenario(name="low", params={"ad_spend": 100.0}),
            Scenario(name="high", params={"ad_spend": 200.0}),
            Scenario(name="low_again", params={"ad_spend": 100.0}),
        ],
    )
    finals = {s.scenario: s.series("Sales")[-1]["v"] for s in run.scenarios}
    assert math.isclose(finals["low"], finals["low_again"], rel_tol=1e-6)
    assert finals["high"] > finals["low"]


def test_return_columns_filter() -> None:
    spec = get_model("sales_growth")
    assert spec is not None
    engine = PySDEngine()
    run = engine.run_scenarios(
        spec.ref,
        [Scenario(name="base", params={})],
        return_columns=["Sales"],
    )
    variables = run.scenarios[0].variables
    assert "Sales" in variables
    # PySD internal bookkeeping columns are always stripped.
    assert "INITIAL TIME" not in variables


def test_missing_model_file_raises() -> None:
    bad = ModelRef(model_id="ghost", source="catalog", path="/no/such/model.xmile")
    engine = PySDEngine()
    with pytest.raises(SimulationError) as ei:
        engine.run_scenarios(bad, [Scenario(name="x", params={})])
    assert ei.value.code == "model_not_found"


def test_empty_scenarios_raises() -> None:
    spec = get_model("sales_growth")
    assert spec is not None
    engine = PySDEngine()
    with pytest.raises(SimulationError) as ei:
        engine.run_scenarios(spec.ref, [])
    assert ei.value.code == "no_scenarios"
